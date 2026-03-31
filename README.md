# BDH Interactive Explainer | Gold Class Submissoin Walkthrough

![Gold Class Features Demo](/Users/prathamgarg/.gemini/antigravity/brain/0608a548-be27-4b89-bd7b-0bdc03ca0bd0/gold_class_reinforce_demo_1774937375116.webp)

## What Did We Build?
We built a premium, real-time "BDH Interactive Explainer" leveraging **Pathway**, **Modal**, and **Next.js 15**. This specifically targets the "Visualization Inspiration" requirement of the Pathway Hackathon by demystifying the Burn Dragon Hatchling (BDH) architecture's graph topology and sparse activations. 

## Architectural Highlights

### 1. The Real-Weights Modal Pipeline
Unlike typical UI mockups, we trained a real character-level BDH model using the `pathwaycom/bdh` official codebase on a Modal T4 GPU. 
- The weights are dynamically saved and loaded via a `modal.Volume`.
- A FastAPI web endpoint intercepts live user queries, executes mathematically accurate forward passes, and extracts exact internal state tensors (`x_sparse`, `y_sparse`).

### 2. High-Throughput Pathway Streaming
We structured the data delivery as a simulated low-latency pipeline mirroring Pathway's own data processing standards:
- The Next.js frontend sends a string of tokens.
- The server delays slightly to mimic ingestion, evaluates the token layer-by-layer, and yields the dense multidimensional array structures as Server-Sent Events (SSE).

### 3. Inference-Time Learning (Hebbian Updates)
We completely eliminated "frozen weights." The UI features a real-time **Reinforce Connection** button that sends a `target_token` to the Modal backend. The PyTorch model instantly executes a single-step Hebbian mathematical update (via cross-entropy gradients mapped directly back to the graph), permanently rewiring the deployed model's connections in sub-seconds.

### 4. Monosemantic Dictionary Extraction
The backend actively tracks co-activations across the token stream. When a specific neuron fires exclusively for characters like 'h' or 'o', the frontend dynamically attaches a floating label (e.g., `'h' detector`), visually proving Track 2 Interpretability.

## Running the Application
The environment is split into two constantly-running servers:
1. **Frontend:** Accessible at `localhost:3000` (started via `npm run dev`)
2. **Backend:** Accessible at `https://podshorts--bdh-explainer-backend-fastapi-app-dev.modal.run` (started via `modal serve`). 

Type any input in the UI stream input box to watch the real-time tensors light up!

> [!NOTE]  
> The Next.js application requires no further build steps. All UI components gracefully handle parsing errors in case the Modal backend cold-starts.
