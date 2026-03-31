import modal
from pathlib import Path
import json
import asyncio

app = modal.App("bdh-explainer-backend")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "fastapi",
        "uvicorn",
        "pathway",
        "torch",
        "numpy",
        "requests",
        "sse-starlette",
    )
    .run_commands("git clone https://github.com/pathwaycom/bdh.git /root/bdh")
)

# A volume to store trained checkpoints
volume = modal.Volume.from_name("bdh-checkpoints", create_if_missing=True)

@app.function(image=image, volumes={"/vol": volume}, timeout=3600, gpu="T4")
def train_tiny_model():
    """Trains a tiny character-level BDH model and saves the weights to volume."""
    import sys
    sys.path.append("/root/bdh")
    import bdh
    import torch
    import torch.nn.functional as F
    import os
    import numpy as np
    import requests

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Configuration for a really tiny model that trains in seconds
    BDH_CONFIG = bdh.BDHConfig(n_layer=2, n_embd=64, n_head=2, mlp_internal_dim_multiplier=16, vocab_size=256)
    model = bdh.BDH(BDH_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # download tiny shakespeare
    input_file_path = "/vol/input.txt"
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)

    # very simple batch generator
    data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
    BLOCK_SIZE = 64
    BATCH_SIZE = 16
    MAX_ITERS = 200

    model.train()
    for step in range(MAX_ITERS):
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([torch.from_numpy((data[i : i + BLOCK_SIZE]).astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + BLOCK_SIZE]).astype(np.int64)) for i in ix]).to(device)
        
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 50 == 0:
            print(f"Step {step}, Loss {loss.item()}")

    checkpoint_path = "/vol/bdh_tiny.pt"
    torch.save(model.state_dict(), checkpoint_path)
    volume.commit()
    print(f"Model saved to {checkpoint_path}")
    return checkpoint_path


@app.function(image=image, volumes={"/vol": volume})
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Query, Body
    from fastapi.middleware.cors import CORSMiddleware
    from sse_starlette.sse import EventSourceResponse
    import pathway as pw
    import sys
    sys.path.append("/root/bdh")
    import bdh
    import torch
    import torch.nn.functional as F
    import json
    import asyncio
    import collections
    import numpy as np

    web_app = FastAPI(title="BDH Explainer API")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    device = torch.device("cpu")
    BDH_CONFIG = bdh.BDHConfig(n_layer=2, n_embd=64, n_head=2, mlp_internal_dim_multiplier=16, vocab_size=256)
    model = bdh.BDH(BDH_CONFIG).to(device)
    model.eval()

    checkpoint_path = "/vol/bdh_tiny.pt"
    try:
        volume.reload()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Successfully loaded actual trained weights from Volume!")
    except Exception as e:
        print("Could not load weights.", e)

    neuron_semantics = {}

    @web_app.get("/stream")
    async def stream_activations(prompt: str = Query(default="Hey")):
        async def event_generator():
            for token_char in prompt:
                await asyncio.sleep(0.3) 

                idx = torch.tensor(bytearray(token_char, "utf-8"), dtype=torch.long, device=device).unsqueeze(0)
                
                with torch.no_grad():
                    B, T = idx.size()
                    D = BDH_CONFIG.n_embd
                    nh = BDH_CONFIG.n_head
                    N = D * BDH_CONFIG.mlp_internal_dim_multiplier // nh

                    x = model.embed(idx).unsqueeze(1)
                    x = model.ln(x)
                    
                    x_latent = x @ model.encoder
                    x_sparse = F.relu(x_latent) 
                    yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
                    yKV = model.ln(yKV)
                    
                    y_latent = yKV @ model.encoder_v
                    y_sparse = F.relu(y_latent)

                    head_0_x_sparse = x_sparse[0, 0, 0, :].numpy().tolist()
                    head_0_y_sparse = y_sparse[0, 0, 0, :].numpy().tolist()

                    sliced_x = head_0_x_sparse[:64]
                    sliced_y = head_0_y_sparse[:64]

                    # Track interpretability: which neuron fired for this character?
                    active_indices = [i for i, val in enumerate(sliced_x) if val > 0]
                    for i in active_indices:
                        if i not in neuron_semantics:
                            neuron_semantics[i] = {}
                        if token_char not in neuron_semantics[i]:
                            neuron_semantics[i][token_char] = 0
                        neuron_semantics[i][token_char] += 1

                    # Get top semantic label for active neurons
                    top_labels = {}
                    for i in active_indices:
                        if i in neuron_semantics and neuron_semantics[i]:
                            best_char = max(neuron_semantics[i].items(), key=lambda item: item[1])[0]
                            top_labels[f"n-{i}"] = f"'{best_char}' detector"

                    payload = {
                        "token": token_char,
                        "layer": 0,
                        "x_sparse": [{"id": f"n-{i}", "value": val} for i, val in enumerate(sliced_x)],
                        "y_sparse": [{"id": f"n-{i}", "value": val} for i, val in enumerate(sliced_y)],
                        "semantics": top_labels
                    }

                yield json.dumps(payload)
            yield "[DONE]"
        return EventSourceResponse(event_generator())

    @web_app.post("/reinforce")
    async def reinforce(payload: dict = Body(...)):
        """
        Inference-Time Continuous Learning
        Executes a real-time weight update (Hebbian simulation via SGD) without needing a full dataset.
        """
        prompt = payload.get("prompt", "a")
        correct_token = payload.get("correct_token", "b")

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
        
        input_text = prompt
        target_text = input_text[1:] + correct_token
        
        ix = torch.tensor(bytearray(input_text, "utf-8"), dtype=torch.long, device=device).unsqueeze(0)
        iy = torch.tensor(bytearray(target_text, "utf-8"), dtype=torch.long, device=device).unsqueeze(0)
        
        logits, loss = model(ix, iy)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.eval()
        
        torch.save(model.state_dict(), checkpoint_path)
        volume.commit()
        
        return {"status": "success", "loss": float(loss.item())}

    return web_app
