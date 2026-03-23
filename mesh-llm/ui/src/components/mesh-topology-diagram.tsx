import { type CSSProperties, useEffect, useMemo, useRef, useState } from 'react';
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
  type ReactFlowInstance,
} from '@xyflow/react';
import { Badge } from './ui/badge';
import { Cpu, Gauge, Network, Sparkles, Wifi } from 'lucide-react';

import { cn } from '../lib/utils';

type Peer = {
  role: string;
  vram_gb: number;
};

type StatusPayload = {
  is_client: boolean;
  my_vram_gb: number;
  model_name: string;
  peers: Peer[];
};

type TopologyNode = {
  id: string;
  vram: number;
  self: boolean;
  host: boolean;
  client: boolean;
  serving: string;
  servingModels: string[];
  statusLabel: string;
  latencyMs?: number | null;
};

type ThemeMode = 'auto' | 'light' | 'dark';

type PositionedTopologyNode = TopologyNode & {
  x: number;
  y: number;
  bucket: 'center' | 'serving' | 'worker' | 'client';
};

type TopologyNodeInfo = {
  role: string;
  statusLabel: string;
  latencyMs?: number | null;
  loadedModel: string;
  loadedModels: string[];
  vramGb: number;
  vramSharePct: number;
};

type TopologyFlowNodeData = {
  node: PositionedTopologyNode;
  info: TopologyNodeInfo;
  selected: boolean;
  sameModelAsCurrent: boolean;
};

function shortName(name: string) {
  return (name || '').replace(/-Q\w+$/, '').replace(/-Instruct/, '');
}

function formatLatency(value?: number | null) {
  if (value == null || !Number.isFinite(Number(value))) return 'n/a';
  const ms = Math.round(Number(value));
  if (ms <= 0) return '<1 ms';
  return `${ms} ms`;
}

function topologyStatusClass(status: string) {
  if (status === 'Serving' || status === 'Serving (split)') {
    return 'border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300';
  }
  if (status === 'Client') {
    return 'border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300';
  }
  if (status === 'Host') {
    return 'border-indigo-500/40 bg-indigo-500/10 text-indigo-700 dark:text-indigo-300';
  }
  if (status === 'Idle' || status === 'Standby') {
    return 'border-zinc-500/40 bg-zinc-500/10 text-zinc-700 dark:text-zinc-300';
  }
  return 'border-border bg-muted text-foreground';
}

function TopologyFlowNode({ data }: NodeProps<TopologyFlowNodeData>) {
  const isCenter = data.node.bucket === 'center';
  const dotClass = isCenter ? 'bg-primary border-primary' : 'bg-muted border-border';
  const statusClass = topologyStatusClass(data.info.statusLabel);
  const dotCenterY = 22;
  const edgeHandleStyle = {
    opacity: 0,
    width: 1,
    height: 1,
    border: 0,
    pointerEvents: 'none' as const,
    left: '50%',
    top: dotCenterY,
    transform: 'translate(-50%, -50%)',
  };

  return (
    <div className="relative w-[246px] pt-2">
      <Handle type="target" position={Position.Top} style={edgeHandleStyle} />
      <Handle type="source" position={Position.Top} style={edgeHandleStyle} />

      <div className={cn('mx-auto h-7 w-7 rounded-full border-2', dotClass)} />
      <div className="mt-1 flex items-center justify-center gap-1 text-[10px] leading-3 text-foreground">
        <span className="break-all">{data.node.id}</span>
        {data.node.self ? (
          <Badge
            variant="outline"
            className="h-4 rounded-full border-sky-500/45 bg-sky-500/10 px-1.5 text-[9px] font-medium text-sky-700 dark:border-sky-400/55 dark:bg-sky-400/15 dark:text-sky-200"
          >
            You
          </Badge>
        ) : null}
      </div>

      <div
        className={cn(
          'mt-1 rounded-md border bg-card p-2',
          data.sameModelAsCurrent ? 'border-emerald-500/60 dark:border-emerald-400/70' : data.selected ? 'border-ring' : 'border-border/90',
          data.selected ? 'ring-1 ring-ring/50' : null,
        )}
      >
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            {data.info.loadedModels.length > 1 ? (
              <div className="flex flex-col gap-0.5">
                {data.info.loadedModels.map((model) => (
                  <div key={model} className="inline-flex items-center gap-1 text-[11px] font-medium leading-4">
                    <Sparkles className="h-3 w-3 shrink-0 text-muted-foreground" />
                    <span className="min-w-0 truncate" title={model}>{shortName(model)}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="inline-flex items-center gap-1 text-[11px] font-medium leading-4">
                <Sparkles className="h-3 w-3 shrink-0 text-muted-foreground" />
                <span className="min-w-0 truncate" title={data.info.loadedModels[0] || data.info.loadedModel}>{data.info.loadedModel}</span>
              </div>
            )}
          </div>
          <Badge className={cn('h-5 shrink-0 rounded-full px-2 text-[9px] font-medium', statusClass)}>
            {data.info.statusLabel}
          </Badge>
        </div>

        <div className="mt-2 flex flex-wrap gap-1.5 text-[10px] leading-3">
          <div className="inline-flex items-center gap-1 rounded-full border bg-muted/30 px-2 py-1">
            <Network className="h-3 w-3 text-muted-foreground" />
            <span className="text-muted-foreground">Role</span>
            <span className="font-medium">{data.info.role}</span>
          </div>
          {!data.node.self ? (
            <div className="inline-flex items-center gap-1 rounded-full border bg-muted/30 px-2 py-1">
              <Wifi className="h-3 w-3 text-muted-foreground" />
              <span className="text-muted-foreground">Latency</span>
              <span className="font-medium">{formatLatency(data.info.latencyMs)}</span>
            </div>
          ) : null}
          <div className="inline-flex items-center gap-1 rounded-full border bg-muted/30 px-2 py-1">
            <Cpu className="h-3 w-3 text-muted-foreground" />
            <span className="text-muted-foreground">VRAM</span>
            <span className="font-medium">{data.info.vramGb.toFixed(1)} GB</span>
          </div>
          <div className="inline-flex items-center gap-1 rounded-full border bg-muted/30 px-2 py-1">
            <Gauge className="h-3 w-3 text-muted-foreground" />
            <span className="text-muted-foreground">Share</span>
            <span className="whitespace-nowrap font-medium">{data.info.vramSharePct}%</span>
          </div>
        </div>
      </div>
    </div>
  );
}

const topologyNodeTypes = { topologyNode: TopologyFlowNode };

export default function MeshTopologyDiagram({
  status,
  nodes,
  selectedModel,
  themeMode,
  fullscreen = false,
  heightClass,
  containerStyle,
}: {
  status: StatusPayload | null;
  nodes: TopologyNode[];
  selectedModel: string;
  themeMode: ThemeMode;
  fullscreen?: boolean;
  heightClass?: string;
  containerStyle?: CSSProperties;
}) {
  if (!status || !nodes.length) {
    return (
      <div className="flex h-full w-full items-center justify-center text-sm text-muted-foreground">
        No topology data yet.
      </div>
    );
  }

  return (
    <MeshTopologyFlow
      status={status}
      nodes={nodes}
      selectedModel={selectedModel}
      themeMode={themeMode}
      fullscreen={fullscreen}
      heightClass={heightClass}
      containerStyle={containerStyle}
    />
  );
}

function MeshTopologyFlow({
  status,
  nodes,
  selectedModel,
  themeMode,
  fullscreen,
  heightClass,
  containerStyle,
}: {
  status: StatusPayload;
  nodes: TopologyNode[];
  selectedModel: string;
  themeMode: ThemeMode;
  fullscreen: boolean;
  heightClass?: string;
  containerStyle?: CSSProperties;
}) {
  const center = nodes.find((node) => node.host) || nodes.find((node) => node.self) || nodes[0];
  const others = nodes.filter((node) => node.id !== center.id).sort((a, b) => (b.vram - a.vram) || a.id.localeCompare(b.id));
  const focusModel = selectedModel || status.model_name || '';
  const serving = others.filter((node) => !node.client && !!node.serving && (!focusModel || node.serving === focusModel));
  const servingIds = new Set(serving.map((node) => node.id));
  const clients = others.filter((node) => node.client);
  const workers = others.filter((node) => !node.client && !servingIds.has(node.id));

  const total = nodes.length;
  const nodeRadius = total >= 500 ? 3.6 : total >= 280 ? 4.8 : total >= 160 ? 6.2 : total >= 90 ? 7.4 : total >= 50 ? 8.8 : 10.4;
  const positioned = layoutTopologyNodes(center, serving, workers, clients, nodeRadius);
  const clientEdgeStride = total > 320 ? 6 : total > 220 ? 4 : total > 120 ? 2 : 1;
  const meshVramGb = nodes.filter((node) => !node.client).reduce((sum, node) => sum + Math.max(0, node.vram), 0);
  const currentNodeServingModel = useMemo(() => {
    const current = nodes.find((node) => node.self);
    if (!current || current.client || !current.serving || current.serving === '(idle)') return '';
    return current.serving;
  }, [nodes]);

  const [selectedNodeId, setSelectedNodeId] = useState(center.id);

  useEffect(() => {
    setSelectedNodeId((prev) => (nodes.some((node) => node.id === prev) ? prev : center.id));
  }, [nodes, center.id]);

  const nodeInfoById = useMemo(() => {
    const output = new Map<string, TopologyNodeInfo>();
    for (const node of nodes) {
      const servingModel = !node.client && node.serving && node.serving !== '(idle)' ? node.serving : '';
      const servingModels = !node.client ? node.servingModels.filter((model) => model && model !== '(idle)') : [];
      const role = node.client ? 'Client' : node.host ? 'Host' : servingModel ? 'Worker' : 'Idle';
      const vramSharePct = !node.client && meshVramGb > 0 ? Math.round((Math.max(0, node.vram) / meshVramGb) * 100) : 0;
      output.set(node.id, {
        role,
        statusLabel: node.statusLabel,
        latencyMs: node.latencyMs ?? null,
        loadedModel: node.client ? 'n/a' : servingModels.length > 0 ? servingModels.map(shortName).join(', ') : servingModel ? shortName(servingModel) : 'idle',
        loadedModels: node.client ? [] : servingModels.length > 0 ? servingModels : (servingModel ? [servingModel] : []),
        vramGb: Math.max(0, node.vram),
        vramSharePct,
      });
    }
    return output;
  }, [nodes, meshVramGb]);

  const flowColorMode = themeMode === 'auto'
    ? (typeof document !== 'undefined' && document.documentElement.classList.contains('dark') ? 'dark' : 'light')
    : themeMode;
  const flowLayoutKey = useMemo(
    () => `${fullscreen ? 'fs' : 'std'}:${positioned.map((node) => node.id).sort().join(',')}`,
    [fullscreen, positioned],
  );
  const flowContainerRef = useRef<HTMLDivElement | null>(null);
  const flowInstanceRef = useRef<ReactFlowInstance | null>(null);
  const [containerReady, setContainerReady] = useState(false);
  const fitViewOptions = useMemo(() => ({ padding: 0.12, maxZoom: 1.45 }), []);
  const fitDuration = fullscreen ? 220 : 0;

  useEffect(() => {
    if (!flowInstanceRef.current) return;
    const fit = () => {
      flowInstanceRef.current?.fitView({ ...fitViewOptions, duration: fitDuration });
    };
    const frame = window.requestAnimationFrame(fit);
    const timeout = window.setTimeout(fit, 180);
    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(timeout);
    };
  }, [fitDuration, fitViewOptions, flowLayoutKey]);

  useEffect(() => {
    const container = flowContainerRef.current;
    if (!container) return;

    const update = () => {
      const rect = container.getBoundingClientRect();
      const ready = rect.width > 8 && rect.height > 8;
      setContainerReady(ready);
      if (ready) {
        flowInstanceRef.current?.fitView({ ...fitViewOptions, duration: 0 });
      }
    };

    update();
    const observer = new ResizeObserver(update);
    observer.observe(container);
    return () => observer.disconnect();
  }, [fitViewOptions, flowLayoutKey, containerStyle, heightClass]);

  const flowNodes = useMemo<Node<TopologyFlowNodeData>[]>(() => {
    return positioned.map((node) => ({
      id: node.id,
      type: 'topologyNode',
      position: { x: node.x, y: node.y },
      origin: [0.5, 0],
      data: {
        node,
        info: nodeInfoById.get(node.id) ?? {
          role: 'Node',
          statusLabel: 'n/a',
          latencyMs: null,
          loadedModel: 'idle',
          loadedModels: [],
          vramGb: 0,
          vramSharePct: 0,
        },
        selected: node.id === selectedNodeId,
        sameModelAsCurrent: !!currentNodeServingModel && !node.client && node.serving === currentNodeServingModel,
      },
      draggable: false,
      selectable: false,
      connectable: false,
      zIndex: node.id === center.id ? 10 : 1,
    }));
  }, [positioned, nodeInfoById, selectedNodeId, center.id, currentNodeServingModel]);

  const flowEdges = useMemo<Edge[]>(() => {
    const outer = positioned.filter((node) => node.id !== center.id);
    return outer
      .filter((node, index) => !(node.bucket === 'client' && index % clientEdgeStride !== 0))
      .map((node) => {
        const stroke =
          node.bucket === 'serving'
            ? 'rgba(34,197,94,0.35)'
            : node.bucket === 'worker'
              ? 'rgba(56,189,248,0.3)'
              : 'rgba(148,163,184,0.22)';
        return {
          id: `edge-${center.id}-${node.id}`,
          source: center.id,
          target: node.id,
          type: 'straight',
          className: `mesh-edge mesh-edge--${node.bucket}`,
          animated: false,
          style: {
            stroke,
            strokeWidth: node.bucket === 'client' ? 1.8 : 2.4,
            strokeDasharray: node.bucket === 'client' ? '2 8' : '2 6',
          },
        };
      });
  }, [positioned, center.id, clientEdgeStride]);

  return (
    <div
      className={cn(
        'overflow-hidden rounded-lg border',
        heightClass ?? 'h-[360px] md:h-[420px] lg:h-[460px] xl:h-[520px]',
      )}
      ref={flowContainerRef}
      style={containerStyle}
    >
      {containerReady ? (
        <ReactFlow
          key={flowLayoutKey}
          className="h-full w-full"
          style={{ width: '100%', height: '100%' }}
          nodes={flowNodes}
          edges={flowEdges}
          nodeTypes={topologyNodeTypes}
          colorMode={flowColorMode}
          fitView
          fitViewOptions={fitViewOptions}
          minZoom={0.2}
          maxZoom={1.6}
          zoomOnScroll={false}
          zoomOnPinch={false}
          panOnScroll={false}
          panOnDrag
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          onInit={(instance) => {
            flowInstanceRef.current = instance;
            window.requestAnimationFrame(() => {
              instance.fitView({ ...fitViewOptions, duration: fitDuration });
            });
          }}
          onNodeClick={(_, node) => setSelectedNodeId(node.id)}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="hsl(var(--border))" />
          <Controls showInteractive={false} position="bottom-right" />
        </ReactFlow>
      ) : (
        <div className="flex h-full w-full items-center justify-center text-sm text-muted-foreground">
          Preparing topology view...
        </div>
      )}
    </div>
  );
}

function layoutTopologyNodes(
  center: TopologyNode,
  serving: TopologyNode[],
  workers: TopologyNode[],
  clients: TopologyNode[],
  nodeRadius: number,
): PositionedTopologyNode[] {
  const all: Array<PositionedTopologyNode> = [
    ...serving.map((node) => ({ ...node, bucket: 'serving' as const, x: 0, y: 0 })),
    ...workers.map((node) => ({ ...node, bucket: 'worker' as const, x: 0, y: 0 })),
    ...clients.map((node) => ({ ...node, bucket: 'client' as const, x: 0, y: 0 })),
  ];

  const positioned: PositionedTopologyNode[] = [{ ...center, x: 0, y: 0, bucket: 'center' }];
  const peerCount = all.length;
  if (peerCount === 0) return positioned;

  const baseRadius = peerCount <= 3
    ? 190
    : peerCount <= 6
      ? 220
      : peerCount <= 10
        ? 250
        : Math.max(200, nodeRadius * 9 + 96);
  const ringSpacing = peerCount <= 10
    ? 120
    : Math.max(102, nodeRadius * 7 + 58);
  const minArcLength = Math.max(110, nodeRadius * 7 + 54);

  if (peerCount <= 10) {
    for (let index = 0; index < peerCount; index += 1) {
      const angle = -Math.PI / 2 + ((2 * Math.PI * index) / peerCount);
      const node = all[index];
      positioned.push({
        ...node,
        x: Math.cos(angle) * baseRadius,
        y: Math.sin(angle) * baseRadius,
      });
    }
    return positioned;
  }

  let offset = 0;
  let ring = 0;
  while (offset < peerCount) {
    const radius = baseRadius + ring * ringSpacing;
    const capacity = Math.max(8, Math.floor((2 * Math.PI * radius) / minArcLength));
    const take = Math.min(capacity, peerCount - offset);
    const phase = ring % 2 === 0 ? 0 : (Math.PI / Math.max(6, take));
    for (let index = 0; index < take; index += 1) {
      const angle = -Math.PI / 2 + phase + ((2 * Math.PI * index) / take);
      const node = all[offset + index];
      positioned.push({
        ...node,
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
      });
    }
    offset += take;
    ring += 1;
  }

  return positioned;
}
