"use client";

import { useMemo, useEffect, useRef, useState, useCallback } from "react";
import dynamic from "next/dynamic";
import { Network, Maximize2, Minimize2 } from "lucide-react";

// Next.js dynamic import for client-side rendering only
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

interface Activation {
  id: string;
  value: number;
}

export function GraphBrain({
  bdhActivations,
  isReinforcing,
  semantics,
  topologyLinks,
}: {
  bdhActivations: Activation[];
  isReinforcing: boolean;
  semantics?: Record<string, string>;
  topologyLinks: Array<{source: string, target: string, weight: number}>;
}) {
  const fgRef = useRef<any>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 });
  const containerRef = useRef<HTMLDivElement>(null);
  const [mounted, setMounted] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Resize observer to keep the canvas responsive
  useEffect(() => {
    if (!containerRef.current) return;
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    };
    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, [isFullscreen]); // re-run when full screen toggles!

  // Generate Topology data directly seeded from the Backend's mathematical tensor edges
  const graphData = useMemo(() => {
    const nodes = Array.from({ length: 64 }, (_, i) => ({
      id: `n-${i}`,
      group: i % 10 === 0 ? 1 : 2,
      label: `Neuron ${i}`,
    }));

    // Filter to top 50 strongest connections
    const topLinks = topologyLinks
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 50)
      .map(link => ({...link}));

    if (topLinks.length < 5) {
      for (let i = 0; i < 64; i += 3) {
        topLinks.push({ source: `n-${i}`, target: `n-${Math.min(63, i+5)}`, weight: 0.1 });
      }
    }

    return { nodes, links: topLinks };
  }, [topologyLinks]);

  // Adjust D3 physical forces for a very clean, structured network (less bouncy, well-spread)
  useEffect(() => {
    if (mounted && fgRef.current) {
      fgRef.current.d3Force('charge').strength(isFullscreen ? -2000 : -1200); // Massive repulsion to separate nodes wide apart
      fgRef.current.d3Force('link').distance(isFullscreen ? 300 : 180);       // Stretch the resting link length significantly
      fgRef.current.d3Force('center').strength(0.05); // Slight centralization to keep it in frame
    }
  }, [mounted, graphData, isFullscreen]);

  // Set the "fire" state based on active neurons to change Node appearance
  const activeNeuronIds = useMemo(() => {
    return bdhActivations.filter((a) => a.value > 0).map((a) => a.id);
  }, [bdhActivations]);

  // Node Canvas Rendering
  const paintNode = useCallback(
    (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const isActive = activeNeuronIds.includes(node.id);
      const semanticLabel = semantics?.[node.id];
      
      const size = node.group === 1 ? 6 : 4; 
      const glowScale = isActive ? 1.5 : 1;
      
      ctx.beginPath();
      ctx.arc(node.x, node.y, size * glowScale, 0, 2 * Math.PI, false);
      
      if (isActive) {
        ctx.fillStyle = "#10b981"; 
        ctx.shadowColor = "rgba(16, 185, 129, 0.9)";
        ctx.shadowBlur = 15;
      } else {
        ctx.fillStyle = "#4b5563"; 
        ctx.strokeStyle = "#9ca3af"; 
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
      ctx.fill();

      if (isActive && semanticLabel) {
        const fontSize = Math.max(12, 14 / globalScale);
        ctx.font = `bold ${fontSize}px sans-serif`;
        
        ctx.fillStyle = "#34d399"; 
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(semanticLabel, node.x, node.y + size + 4);
      }
    },
    [activeNeuronIds, semantics]
  );

  return (
    <div 
      className={`glass flex flex-col relative overflow-hidden transition-all duration-300 ${
        isFullscreen 
          ? "fixed inset-0 z-50 rounded-none w-screen h-screen bg-black/95 backdrop-blur-xl p-8" 
          : "rounded-xl p-4 min-h-[500px] h-full ring-1 ring-white/10"
      }`} 
      ref={containerRef}
    >
      
      <div className="flex flex-col sm:flex-row items-center justify-between mb-4 z-10 w-full">
        <label className={`font-semibold text-emerald-400 flex items-center gap-2 ${isFullscreen ? "text-2xl" : "text-sm"}`}>
          <Network size={isFullscreen ? 28 : 18} /> Emergent Topology Graph
        </label>
        
        <div className="flex items-center gap-6">
          <span className="text-xs font-mono text-gray-300 bg-gray-800/40 px-3 py-1 rounded-md border border-gray-600/30">
            {isReinforcing ? "Hebbian Reinforcement Active" : "Scale-Free Hub Structures"}
          </span>
          <button 
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="text-gray-400 hover:text-white bg-white/5 hover:bg-white/10 p-2 rounded-lg transition-colors border border-white/10 flex items-center gap-2"
          >
            {isFullscreen ? <><Minimize2 size={20} /> Exit Fullscreen</> : <><Maximize2 size={16} /> Expand</>}
          </button>
        </div>
      </div>

      <div className="flex-1 w-full relative z-0 flex items-center justify-center">
        {mounted && graphData.nodes.length > 0 && (
          <ForceGraph2D
            ref={fgRef}
            width={dimensions.width - (isFullscreen ? 64 : 32)} 
            height={dimensions.height - (isFullscreen ? 120 : 80)}
            graphData={graphData}
            nodeCanvasObject={paintNode}
            linkColor={(link: any) => {
              const srcActive = activeNeuronIds.includes(link.source.id || link.source);
              const tgtActive = activeNeuronIds.includes(link.target.id || link.target);
              
              if (srcActive && tgtActive) return "rgba(16, 185, 129, 0.9)"; 
              if (srcActive || tgtActive) return "rgba(16, 185, 129, 0.4)"; 
              return isFullscreen ? "rgba(200, 200, 200, 0.2)" : "rgba(200, 200, 200, 0.4)"; 
            }}
            linkWidth={(link: any) => {
              const srcActive = activeNeuronIds.includes(link.source.id || link.source);
              const tgtActive = activeNeuronIds.includes(link.target.id || link.target);
              
              const baseWidth = Math.max(0.5, (link.weight || 0.1) * 1.5); 
              if (srcActive && tgtActive && isReinforcing) return baseWidth + 2; 
              if (srcActive && tgtActive) return baseWidth + 0.8;
              return baseWidth;
            }}
            linkDirectionalParticles={(link: any) => {
              const srcActive = activeNeuronIds.includes(link.source.id || link.source);
              const tgtActive = activeNeuronIds.includes(link.target.id || link.target);
              return (srcActive && tgtActive) ? 3 : 0;
            }}
            linkDirectionalParticleSpeed={0.015}
            linkDirectionalParticleColor={() => "rgba(255, 255, 255, 0.8)"}
            d3AlphaDecay={0.01} 
            d3VelocityDecay={0.5}
            cooldownTicks={300}
            backgroundColor="transparent"
          />
        )}
      </div>

      {isReinforcing && (
        <div className="absolute inset-x-0 bottom-8 pointer-events-none animate-pulse flex items-center justify-center z-20">
          <div className="bg-gray-900/80 text-emerald-300 px-6 py-2 rounded-lg font-bold shadow-lg backdrop-blur-sm border border-emerald-500/50 text-base md:text-lg">
            Applying Synaptic Reinforcement Matrix Updates...
          </div>
        </div>
      )}
    </div>
  );
}
