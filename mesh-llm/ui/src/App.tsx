import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import {
  BarChart3,
  Bot,
  Check,
  Circle,
  Copy,
  Cpu,
  Gauge,
  Loader2,
  Network,
  Send,
  Sparkles,
  User,
  Wifi,
} from 'lucide-react';

import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from './components/ui/accordion';
import { Badge } from './components/ui/badge';
import { Button } from './components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Input } from './components/ui/input';
import { Popover, PopoverContent, PopoverTrigger } from './components/ui/popover';
import { ScrollArea } from './components/ui/scroll-area';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Separator } from './components/ui/separator';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './components/ui/table';
import { Tabs, TabsList, TabsTrigger } from './components/ui/tabs';
import { Textarea } from './components/ui/textarea';
import { cn } from './lib/utils';

type MeshModel = {
  name: string;
  status: 'warm' | 'cold' | string;
  node_count: number;
  size_gb: number;
};

type Peer = {
  id: string;
  role: string;
  models: string[];
  vram_gb: number;
  serving?: string | null;
};

type StatusPayload = {
  node_id: string;
  token: string;
  node_status: string;
  is_host: boolean;
  is_client: boolean;
  llama_ready: boolean;
  model_name: string;
  api_port: number;
  my_vram_gb: number;
  model_size_gb: number;
  mesh_name?: string | null;
  peers: Peer[];
  mesh_models: MeshModel[];
};

type LiveMetrics = {
  requests_inflight?: number;
  requests_total?: number;
  requests_ok?: number;
  requests_err?: number;
  local_bytes_tx?: number;
  local_bytes_rx?: number;
  p50_ttft_ms?: number;
  p95_ttft_ms?: number;
  p50_tokens_per_sec?: number;
  p95_tokens_per_sec?: number;
};

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  reasoning?: string;
  model?: string;
  stats?: string;
  error?: boolean;
};

type LiveSnapshot = {
  ts_minute: number;
  node_id: string;
  active_requests: number;
  active_requests_peak: number;
  requests: number;
  errors: number;
  requests_local: number;
  requests_remote: number;
  request_time_ms_total: number;
  utilization_pct: number;
  tunnel_bytes_total: number;
};

type NodeMetricRow = {
  ts_minute: number;
  source_node_id: string;
  requests: number;
  errors: number;
  requests_local: number;
  requests_remote: number;
  request_time_ms_total: number;
  utilization_pct: number;
  active_requests_peak: number;
  latency_p50_ms?: number | null;
  latency_p95_ms?: number | null;
  latency_p99_ms?: number | null;
  ttft_p50_ms?: number | null;
  ttft_p95_ms?: number | null;
  completion_tokens_total: number;
  tps_avg?: number | null;
  tps_p50?: number | null;
  tps_p95?: number | null;
  tunnel_bytes_total: number;
  observed_at: number;
};

type RollupMetricRow = {
  ts_minute: number;
  node_count: number;
  requests: number;
  errors: number;
  requests_local: number;
  requests_remote: number;
  request_time_ms_total: number;
  utilization_pct_avg_nodes: number;
  active_requests_peak_max: number;
  tunnel_bytes_total: number;
  latency_p95_ms_avg_nodes?: number | null;
  latency_p95_ms_max_nodes?: number | null;
  latency_p50_ms_exact?: number | null;
  latency_p95_ms_exact?: number | null;
  latency_p99_ms_exact?: number | null;
};

type BenchmarkRunRow = {
  ts: number;
  source_node_id: string;
  model: string;
  probe_name: string;
  probe_type: string;
  stream: boolean;
  success: boolean;
  status_code?: number | null;
  latency_ms?: number | null;
  ttft_ms?: number | null;
  completion_tokens?: number | null;
  tokens_per_sec?: number | null;
  error_kind?: string | null;
  observed_at: number;
};

type TelemetryEventsPayload = {
  live: LiveSnapshot;
  nodes: NodeMetricRow[];
  rollup: RollupMetricRow[];
  node_history: NodeMetricRow[];
  benchmarks: BenchmarkRunRow[];
};

type TopSection = 'chat' | 'mesh' | 'metrics';

type TopologyNode = {
  id: string;
  vram: number;
  self: boolean;
  host: boolean;
  client: boolean;
  serving: string;
};

type SeriesPoint = {
  ts_minute: number;
  value: number;
};

export function App() {
  const [section, setSection] = useState<TopSection>('chat');
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [metrics, setMetrics] = useState<LiveMetrics | null>(null);
  const [telemetry, setTelemetry] = useState<TelemetryEventsPayload | null>(null);
  const [telemetryError, setTelemetryError] = useState<string | null>(null);
  const [metricsMinutes, setMetricsMinutes] = useState('180');
  const [benchmarkFilter, setBenchmarkFilter] = useState<'all' | 'ok' | 'fail'>('all');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [reasoningOpen, setReasoningOpen] = useState<Record<string, boolean>>({});
  const chatScrollRef = useRef<HTMLDivElement>(null);

  const warmModels = useMemo(() => {
    const list = (status?.mesh_models ?? []).filter((m) => m.status === 'warm').map((m) => m.name);
    if (!list.length && status?.model_name) list.push(status.model_name);
    return list;
  }, [status]);

  useEffect(() => {
    if (!warmModels.length) return;
    if (!selectedModel || !warmModels.includes(selectedModel)) setSelectedModel(warmModels[0]);
  }, [warmModels, selectedModel]);

  useEffect(() => {
    let stop = false;

    const loadStatus = () => {
      fetch('/api/status')
        .then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json() as Promise<StatusPayload>;
        })
        .then((data) => {
          if (stop) return;
          setStatus(data);
          setStatusError(null);
        })
        .catch((err: Error) => {
          if (!stop) setStatusError(err.message);
        });
    };

    const loadMetrics = () => {
      fetch('/api/metrics/live')
        .then((r) => (r.ok ? (r.json() as Promise<LiveMetrics>) : null))
        .then((data) => {
          if (!stop && data) setMetrics(data);
        })
        .catch(() => undefined);
    };

    loadStatus();
    loadMetrics();

    const statusEvents = new EventSource('/api/events');
    statusEvents.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as StatusPayload;
        setStatus(payload);
        setStatusError(null);
      } catch {
        // ignore malformed status event
      }
    };
    statusEvents.onerror = () => setStatusError('Status stream disconnected. Retrying...');

    const metricsPoll = window.setInterval(loadMetrics, 5000);
    return () => {
      stop = true;
      window.clearInterval(metricsPoll);
      statusEvents.close();
    };
  }, []);

  useEffect(() => {
    let closed = false;
    setTelemetryError(null);
    const es = new EventSource(`/api/metrics/events?minutes=${encodeURIComponent(metricsMinutes)}&limit=300`);

    es.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as TelemetryEventsPayload;
        if (!closed) {
          setTelemetry(payload);
          setTelemetryError(null);
        }
      } catch {
        // ignore malformed payloads
      }
    };
    es.onerror = () => {
      if (!closed) setTelemetryError('Telemetry stream disconnected. Retrying...');
    };

    return () => {
      closed = true;
      es.close();
    };
  }, [metricsMinutes]);

  useEffect(() => {
    const el = chatScrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, isSending]);

  const canChat = !!status && (status.llama_ready || (status.is_client && warmModels.length > 0));
  const nodeCount = (status?.peers.length ?? 0) + (status?.node_id ? 1 : 0);
  const availableModelCount = Math.max(
    (status?.mesh_models ?? []).filter((m) => m.status === 'warm').length,
    warmModels.length,
  );

  async function sendMessage(text: string) {
    const trimmed = text.trim();
    if (!trimmed || !status || isSending) return;

    const model = selectedModel || status.model_name;
    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: trimmed, model };
    const assistantId = crypto.randomUUID();
    const assistantMessage: ChatMessage = { id: assistantId, role: 'assistant', content: '', model };
    const historyForRequest = [...messages, userMessage];

    setMessages([...historyForRequest, assistantMessage]);
    setInput('');
    setIsSending(true);

    const reqStart = performance.now();

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          messages: historyForRequest.map((m) => ({ role: m.role, content: m.content })),
          stream: true,
          stream_options: { include_usage: true },
        }),
      });

      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      let full = '';
      let reasoning = '';
      let completionTokens: number | null = null;
      let firstTokenAt: number | null = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (!data || data === '[DONE]') continue;
          try {
            const chunk = JSON.parse(data) as {
              usage?: { completion_tokens?: number };
              choices?: Array<{ delta?: { content?: string; reasoning_content?: string } }>;
            };
            const delta = chunk.choices?.[0]?.delta;
            if (Number.isFinite(chunk.usage?.completion_tokens)) completionTokens = chunk.usage!.completion_tokens!;
            const contentDelta = delta?.content ?? '';
            const reasoningDelta = delta?.reasoning_content ?? '';
            if (!contentDelta && !reasoningDelta) continue;
            if (firstTokenAt == null) firstTokenAt = performance.now();
            full += contentDelta;
            reasoning += reasoningDelta;
            setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, content: full, reasoning: reasoning || undefined } : m)));
          } catch {
            // ignore malformed chunk
          }
        }
      }

      const endAt = performance.now();
      const genStart = firstTokenAt ?? reqStart;
      const genSecs = Math.max(0.001, (endAt - genStart) / 1000);
      const ttftMs = Math.max(0, Math.round((firstTokenAt ?? endAt) - reqStart));
      const tokenCount = Number.isFinite(completionTokens) ? completionTokens! : Math.max(1, Math.round(Math.max(full.length, 1) / 4));
      const tps = tokenCount / genSecs;
      const stats = `${tokenCount} tok · ${tps.toFixed(1)} tok/s · TTFT ${ttftMs}ms`;

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: m.content || '(empty response)', reasoning: m.reasoning || undefined, stats }
            : m,
        ),
      );

      void fetch('/api/metrics/chat-sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ttft_ms: ttftMs, completion_tokens: tokenCount, tokens_per_sec: tps }),
      }).catch(() => undefined);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, content: `Error: ${message}`, error: true } : m)));
    } finally {
      setIsSending(false);
    }
  }

  const topologyNodes = useMemo<TopologyNode[]>(() => {
    if (!status) return [];
    const nodes: TopologyNode[] = [];
    if (status.node_id) {
      nodes.push({
        id: status.node_id,
        vram: status.my_vram_gb || 0,
        self: true,
        host: status.is_host,
        client: status.is_client,
        serving: status.model_name || '',
      });
    }
    for (const p of status.peers ?? []) {
      nodes.push({
        id: p.id,
        vram: p.vram_gb,
        self: false,
        host: /^Host/.test(p.role),
        client: p.role === 'Client',
        serving: p.serving || '',
      });
    }
    return nodes;
  }, [status]);

  const filteredBenchmarks = useMemo(() => {
    const rows = telemetry?.benchmarks ?? [];
    if (benchmarkFilter === 'ok') return rows.filter((r) => r.success);
    if (benchmarkFilter === 'fail') return rows.filter((r) => !r.success);
    return rows;
  }, [telemetry, benchmarkFilter]);

  function handleSubmit() {
    if (!canChat) return;
    void sendMessage(input);
  }

  return (
    <div className="h-screen overflow-hidden bg-grid [background-size:18px_18px]">
      <div className="mx-auto flex h-full w-full max-w-[1680px] flex-col gap-4 overflow-hidden p-4 md:p-5">
        <Card className="border-border/80 bg-card/85 backdrop-blur">
          <CardContent className="flex flex-wrap items-center gap-3 p-4">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary/10 text-primary">
              <Bot className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <div className="truncate text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">mesh-llm console</div>
              <div className="truncate text-sm text-foreground">
                {status?.mesh_name ? `Mesh ${status.mesh_name}` : 'Local Dashboard'}
                {status?.node_id ? ` · Node ${status.node_id}` : ''}
                {status ? ` · ${status.is_host ? 'Host' : status.is_client ? 'Client' : 'Worker'}` : ''}
              </div>
            </div>
            <Tabs value={section} onValueChange={(v) => setSection(v as TopSection)} className="ml-auto">
              <TabsList className="grid h-10 grid-cols-3 rounded-xl border border-border/80 bg-black/50 p-1 shadow-inner shadow-black/20">
                <TabsTrigger
                  value="chat"
                  className="rounded-lg px-4 text-sm data-[state=active]:bg-background data-[state=active]:text-primary data-[state=active]:shadow-none"
                >
                  Chat
                </TabsTrigger>
                <TabsTrigger
                  value="mesh"
                  className="rounded-lg px-4 text-sm data-[state=active]:bg-background data-[state=active]:text-primary data-[state=active]:shadow-none"
                >
                  Mesh
                </TabsTrigger>
                <TabsTrigger
                  value="metrics"
                  className="rounded-lg px-4 text-sm data-[state=active]:bg-background data-[state=active]:text-primary data-[state=active]:shadow-none"
                >
                  Metrics
                </TabsTrigger>
              </TabsList>
            </Tabs>
            <div className="flex flex-wrap items-center gap-2">
              <Badge>
                <Network className="mr-1 h-3.5 w-3.5" />
                {`${nodeCount} node${nodeCount === 1 ? '' : 's'}`}
              </Badge>
              <Badge>
                <Sparkles className="mr-1 h-3.5 w-3.5" />
                {`${availableModelCount} model${availableModelCount === 1 ? '' : 's'}`}
              </Badge>
              <InvitePopover token={status?.token ?? ''} selectedModel={selectedModel || warmModels[0] || status?.model_name || ''} />
              <ApiEndpointPopover port={status?.api_port ?? null} />
              <StatusBadge ready={!!status?.llama_ready} />
              {statusError ? <Badge className="border-amber-500/30 bg-amber-500/10 text-amber-200">{statusError}</Badge> : null}
            </div>
          </CardContent>
        </Card>

        <div className="min-h-0 flex-1 overflow-hidden">
          {section === 'chat' ? (
            <ChatPage
              inviteToken={status?.token ?? ''}
              warmModels={warmModels}
              selectedModel={selectedModel}
              setSelectedModel={setSelectedModel}
              messages={messages}
              reasoningOpen={reasoningOpen}
              setReasoningOpen={setReasoningOpen}
              chatScrollRef={chatScrollRef}
              input={input}
              setInput={setInput}
              isSending={isSending}
              canChat={canChat}
              onSubmit={handleSubmit}
            />
          ) : null}

          {section === 'mesh' ? (
            <MeshPage status={status} metrics={metrics} topologyNodes={topologyNodes} selectedModel={selectedModel || status?.model_name || ''} />
          ) : null}

          {section === 'metrics' ? (
            <MetricsPage
              telemetry={telemetry}
              telemetryError={telemetryError}
              metricsMinutes={metricsMinutes}
              setMetricsMinutes={setMetricsMinutes}
              benchmarkFilter={benchmarkFilter}
              setBenchmarkFilter={setBenchmarkFilter}
              filteredBenchmarks={filteredBenchmarks}
            />
          ) : null}
        </div>
      </div>
    </div>
  );
}

function ChatPage(props: {
  inviteToken: string;
  warmModels: string[];
  selectedModel: string;
  setSelectedModel: (v: string) => void;
  messages: ChatMessage[];
  reasoningOpen: Record<string, boolean>;
  setReasoningOpen: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
  chatScrollRef: React.RefObject<HTMLDivElement>;
  input: string;
  setInput: (v: string) => void;
  isSending: boolean;
  canChat: boolean;
  onSubmit: () => void;
}) {
  const {
    inviteToken,
    warmModels,
    selectedModel,
    setSelectedModel,
    messages,
    reasoningOpen,
    setReasoningOpen,
    chatScrollRef,
    input,
    setInput,
    isSending,
    canChat,
    onSubmit,
  } = props;

  return (
    <div className="h-full min-h-0">
      <Card className="flex h-full min-h-0 flex-col overflow-hidden border-border/80 bg-card/85 backdrop-blur">
        <CardHeader className="pb-3">
          <div className="flex flex-wrap items-center gap-3">
            <CardTitle className="text-base">Chat</CardTitle>
            <Badge className="border-emerald-500/30 bg-emerald-500/10 text-emerald-300">Streaming Chat UI</Badge>
            <div className="ml-auto flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Model</span>
              <Select value={selectedModel || warmModels[0] || ''} onValueChange={setSelectedModel} disabled={!warmModels.length}>
                <SelectTrigger className="h-8 w-[220px]">
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  {warmModels.map((model) => (
                    <SelectItem key={model} value={model}>
                      {shortName(model)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardHeader>
        <Separator />
        <CardContent className="flex min-h-0 flex-1 flex-col p-0">
          <div ref={chatScrollRef} className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-4 md:px-6">
            {messages.length === 0 ? (
              <InviteFriendEmptyState inviteToken={inviteToken} selectedModel={selectedModel || warmModels[0] || ''} />
            ) : null}

            {messages.map((message) => (
              <ChatBubble
                key={message.id}
                message={message}
                reasoningOpen={!!reasoningOpen[message.id]}
                onReasoningToggle={(open) => setReasoningOpen((prev) => ({ ...prev, [message.id]: open }))}
              />
            ))}

            {isSending ? (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="h-3.5 w-3.5 animate-spin" /> Streaming response...
              </div>
            ) : null}
          </div>
          <Separator />
          <div className="space-y-3 p-4">
            <Card className="shadow-none">
              <CardContent className="space-y-3 p-3">
                <Textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      onSubmit();
                    }
                  }}
                  rows={4}
                  placeholder={props.canChat ? 'Send a prompt to the mesh...' : 'Waiting for a warm model...'}
                  disabled={!props.canChat || isSending}
                  className="min-h-[112px] resize-none border-0 p-1 shadow-none focus-visible:ring-0"
                />
                <div className="flex items-center justify-between gap-2">
                  <div className="text-xs text-muted-foreground">Enter to send. Shift+Enter for newline.</div>
                  <Button onClick={onSubmit} disabled={!props.canChat || !input.trim() || isSending}>
                    {isSending ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Send className="mr-2 h-4 w-4" />}
                    Send
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function InviteFriendEmptyState({ inviteToken, selectedModel }: { inviteToken: string; selectedModel: string }) {
  const [copied, setCopied] = useState(false);
  const command = inviteToken && selectedModel ? `mesh-llm --join ${inviteToken} --model ${selectedModel}` : '';

  async function copy() {
    if (!command) return;
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <Card className="mx-auto max-w-3xl border-dashed shadow-none">
      <CardContent className="space-y-4 p-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
            <Sparkles className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold">Invite a friend to Mesh LLM</h2>
            <p className="text-sm text-muted-foreground">
              Open models exist. Excess compute exists. What&apos;s missing is coordination.
            </p>
          </div>
        </div>

        <div className="space-y-3 text-sm leading-6 text-muted-foreground">
          <p>
            Mesh LLM is a shared network for open AI inference. Instead of relying only on centralized infrastructure,
            people can contribute idle compute and gain access to collective capacity.
          </p>
          <p>
            Invite someone to join this mesh and load the current model. More participation increases available capacity
            and makes the network stronger.
          </p>
        </div>

        <div className="rounded-lg border border-border/70 bg-muted/20 p-3">
          <div className="mb-2 text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">Join Command</div>
          {command ? (
            <div className="flex items-center gap-2">
              <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap rounded bg-background px-2 py-1.5 font-mono text-xs text-foreground">
                {command}
              </code>
              <Button type="button" variant="secondary" size="sm" onClick={() => void copy()}>
                {copied ? <Check className="mr-1 h-3.5 w-3.5" /> : <Copy className="mr-1 h-3.5 w-3.5" />}
                {copied ? 'Copied' : 'Copy'}
              </Button>
            </div>
          ) : (
            <div className="text-xs text-muted-foreground">
              Start or join a mesh and select a model to generate an invite command.
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function MeshPage({
  status,
  metrics,
  topologyNodes,
  selectedModel,
}: {
  status: StatusPayload | null;
  metrics: LiveMetrics | null;
  topologyNodes: TopologyNode[];
  selectedModel: string;
}) {
  return (
    <div className="grid h-full min-h-0 gap-4 xl:grid-cols-[minmax(0,1.2fr)_420px]">
      <Card className="flex min-h-0 flex-col overflow-hidden border-border/80 bg-card/85 backdrop-blur">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Network className="h-4 w-4 text-primary" />
            <CardTitle className="text-base">Mesh Topology</CardTitle>
            <Badge className="ml-auto">{topologyNodes.length} nodes</Badge>
          </div>
        </CardHeader>
        <Separator />
        <CardContent className="min-h-0 flex-1 p-4">
          <div className="grid h-full min-h-0 gap-4 lg:grid-rows-[auto_minmax(0,1fr)]">
            <Card className="shadow-none">
              <CardContent className="p-4">
                <MeshTopologyDiagram status={status} nodes={topologyNodes} selectedModel={selectedModel} />
              </CardContent>
            </Card>
            <div className="grid min-h-0 gap-4 lg:grid-cols-2">
              <Card className="min-h-0 shadow-none">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Peers</CardTitle>
                </CardHeader>
                <CardContent className="min-h-0 pt-0">
                  <ScrollArea className="h-[28rem] pr-3 lg:h-full">
                    {(status?.peers.length ?? 0) > 0 ? (
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>ID</TableHead>
                            <TableHead>Role</TableHead>
                            <TableHead className="text-right">VRAM</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {(status?.peers ?? []).map((peer) => (
                            <TableRow key={peer.id}>
                              <TableCell className="font-mono text-xs">{peer.id}</TableCell>
                              <TableCell>{peer.role}</TableCell>
                              <TableCell className="text-right">{peer.vram_gb.toFixed(1)} GB</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    ) : (
                      <EmptyPanel text="No peers connected." />
                    )}
                  </ScrollArea>
                </CardContent>
              </Card>
              <Card className="min-h-0 shadow-none">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Models</CardTitle>
                </CardHeader>
                <CardContent className="min-h-0 pt-0">
                  <ScrollArea className="h-[28rem] pr-3 lg:h-full">
                    <div className="space-y-2 pr-2">
                      {(status?.mesh_models.length ?? 0) > 0 ? (
                        status!.mesh_models.map((model) => (
                          <div key={model.name} className="rounded-lg border border-border/70 bg-muted/20 p-3">
                            <div className="flex items-center gap-2">
                              <Circle className={cn('h-3.5 w-3.5 fill-current', model.status === 'warm' ? 'text-emerald-500' : 'text-zinc-600')} />
                              <div className="min-w-0 flex-1">
                                <div className="truncate text-sm font-medium">{shortName(model.name)}</div>
                                <div className="truncate text-xs text-muted-foreground">{model.name}</div>
                              </div>
                              <Badge className={model.status === 'warm' ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300' : ''}>{model.status}</Badge>
                            </div>
                            <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
                              <span>{model.node_count} node{model.node_count === 1 ? '' : 's'}</span>
                              <span>{model.size_gb.toFixed(1)} GB</span>
                            </div>
                          </div>
                        ))
                      ) : (
                        <EmptyPanel text="No mesh model metadata yet." />
                      )}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="flex min-h-0 flex-col overflow-hidden border-border/80 bg-card/85 backdrop-blur">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Mesh Summary</CardTitle>
        </CardHeader>
        <CardContent className="min-h-0 flex-1 space-y-4 pt-0">
          <div className="grid grid-cols-2 gap-3">
            <StatCard title="Nodes" value={`${topologyNodes.length}`} icon={<Network className="h-4 w-4" />} />
            <StatCard title="Warm Models" value={`${(status?.mesh_models ?? []).filter((m) => m.status === 'warm').length}`} icon={<Sparkles className="h-4 w-4" />} />
            <StatCard title="VRAM Mesh" value={`${meshGpuVram(status).toFixed(1)} GB`} icon={<Cpu className="h-4 w-4" />} />
            <StatCard title="Inflight" value={`${metrics?.requests_inflight ?? 0}`} icon={<Gauge className="h-4 w-4" />} />
          </div>
          <Separator />
          <div className="space-y-2">
            <SectionLabel>Status</SectionLabel>
            <MetricLine label="Node status" value={status?.node_status ?? 'n/a'} />
            <MetricLine label="Selected model" value={selectedModel ? shortName(selectedModel) : 'n/a'} />
            <MetricLine label="TTFT p95" value={fmtMs(metrics?.p95_ttft_ms)} />
            <MetricLine label="Token rate p95" value={fmtRate(metrics?.p95_tokens_per_sec)} />
            <MetricLine label="TX / RX" value={`${fmtBytes(metrics?.local_bytes_tx)} / ${fmtBytes(metrics?.local_bytes_rx)}`} />
          </div>
          <Separator />
          <div className="space-y-2">
            <SectionLabel>Invite Token</SectionLabel>
            <Input readOnly value={status?.token ?? ''} className="font-mono text-xs" />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function MetricsPage({
  telemetry,
  telemetryError,
  metricsMinutes,
  setMetricsMinutes,
  benchmarkFilter,
  setBenchmarkFilter,
  filteredBenchmarks,
}: {
  telemetry: TelemetryEventsPayload | null;
  telemetryError: string | null;
  metricsMinutes: string;
  setMetricsMinutes: (v: string) => void;
  benchmarkFilter: 'all' | 'ok' | 'fail';
  setBenchmarkFilter: (v: 'all' | 'ok' | 'fail') => void;
  filteredBenchmarks: BenchmarkRunRow[];
}) {
  const latestRollup = telemetry?.rollup?.[telemetry.rollup.length - 1] ?? null;
  const latestNodeRows = [...(telemetry?.nodes ?? [])].sort((a, b) => b.observed_at - a.observed_at);
  const ttftSeries = useMemo(
    () => aggregateNodeHistorySeries(telemetry?.node_history ?? [], (r) => r.ttft_p95_ms),
    [telemetry?.node_history],
  );
  const tpsSeries = useMemo(
    () => aggregateNodeHistorySeries(telemetry?.node_history ?? [], (r) => r.tps_p50),
    [telemetry?.node_history],
  );

  return (
    <div className="grid h-full min-h-0 gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
      <Card className="flex min-h-0 flex-col overflow-hidden border-border/80 bg-card/85 backdrop-blur">
        <CardHeader className="pb-3">
          <div className="flex flex-wrap items-center gap-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <BarChart3 className="h-4 w-4 text-primary" /> Metrics
            </CardTitle>
            <div className="ml-auto flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Range</span>
              <Select value={metricsMinutes} onValueChange={setMetricsMinutes}>
                <SelectTrigger className="h-9 w-[110px]"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="60">1h</SelectItem>
                  <SelectItem value="180">3h</SelectItem>
                  <SelectItem value="720">12h</SelectItem>
                  <SelectItem value="1440">24h</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          {telemetryError ? <div className="text-xs text-amber-300">{telemetryError}</div> : null}
        </CardHeader>
        <Separator />
        <CardContent className="min-h-0 flex-1 p-4">
          <ScrollArea className="h-full pr-3">
            <div className="space-y-4 pr-2">
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                <StatCard title="Requests" value={`${latestRollup?.requests ?? telemetry?.live.requests ?? 0}`} icon={<Gauge className="h-4 w-4" />} />
                <StatCard title="Errors" value={`${latestRollup?.errors ?? telemetry?.live.errors ?? 0}`} icon={<Circle className="h-4 w-4" />} />
                <StatCard title="Nodes" value={`${latestRollup?.node_count ?? telemetry?.nodes.length ?? 0}`} icon={<Network className="h-4 w-4" />} />
                <StatCard title="Util Avg" value={`${fmtPct(latestRollup?.utilization_pct_avg_nodes)}`} icon={<Cpu className="h-4 w-4" />} />
              </div>

              <div className="grid gap-4 xl:grid-cols-3">
                <MetricChartCard
                  title="Requests / min"
                  rows={telemetry?.rollup ?? []}
                  value={(r) => r.requests}
                  color="#60a5fa"
                  format={(n) => `${Math.round(n)}`}
                />
                <MetricChartCard
                  title="Latency p95 (exact)"
                  rows={telemetry?.rollup ?? []}
                  value={(r) => r.latency_p95_ms_exact ?? 0}
                  color="#34d399"
                  format={(n) => `${Math.round(n)}ms`}
                />
                <MetricChartCard
                  title="Tunnel bytes"
                  rows={telemetry?.rollup ?? []}
                  value={(r) => r.tunnel_bytes_total}
                  color="#f59e0b"
                  format={(n) => fmtBytes(n)}
                />
                <DualMetricChartCard
                  title="Requests Local vs Remote"
                  rows={telemetry?.rollup ?? []}
                  aValue={(r) => r.requests_local}
                  bValue={(r) => r.requests_remote}
                  aLabel="Local"
                  bLabel="Remote"
                  aColor="#22c55e"
                  bColor="#60a5fa"
                  format={(n) => `${Math.round(n)}`}
                />
                <MetricChartCard
                  title="TTFT p95 (avg nodes)"
                  rows={ttftSeries}
                  value={(r) => r.value}
                  color="#34d399"
                  format={(n) => `${Math.round(n)}ms`}
                />
                <MetricChartCard
                  title="Tokens / sec p50 (avg nodes)"
                  rows={tpsSeries}
                  value={(r) => r.value}
                  color="#22c55e"
                  format={(n) => n.toFixed(1)}
                />
                <MetricChartCard
                  title="Errors / min"
                  rows={telemetry?.rollup ?? []}
                  value={(r) => r.errors}
                  color="#ef4444"
                  format={(n) => `${Math.round(n)}`}
                />
                <MetricChartCard
                  title="Utilization % (avg nodes)"
                  rows={telemetry?.rollup ?? []}
                  value={(r) => r.utilization_pct_avg_nodes}
                  color="#10b981"
                  format={(n) => `${Math.round(n)}%`}
                />
              </div>

              <Card className="shadow-none">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Nodes (latest)</CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Node</TableHead>
                        <TableHead className="text-right">Req</TableHead>
                        <TableHead className="text-right">Err</TableHead>
                        <TableHead className="text-right">TTFT p95</TableHead>
                        <TableHead className="text-right">TPS p50</TableHead>
                        <TableHead className="text-right">Util</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {latestNodeRows.length ? latestNodeRows.map((r) => (
                        <TableRow key={`${r.source_node_id}-${r.observed_at}`}>
                          <TableCell className="font-mono text-xs">{r.source_node_id}</TableCell>
                          <TableCell className="text-right">{r.requests}</TableCell>
                          <TableCell className="text-right">{r.errors}</TableCell>
                          <TableCell className="text-right">{fmtMs(r.ttft_p95_ms)}</TableCell>
                          <TableCell className="text-right">{fmtRate(r.tps_p50)}</TableCell>
                          <TableCell className="text-right">{fmtPct(r.utilization_pct)}</TableCell>
                        </TableRow>
                      )) : (
                        <TableRow><TableCell colSpan={6} className="text-muted-foreground">No node telemetry yet.</TableCell></TableRow>
                      )}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>

              <Card className="shadow-none">
                <CardHeader className="pb-2">
                  <div className="flex flex-wrap items-center gap-3">
                    <CardTitle className="text-sm">Benchmarks</CardTitle>
                    <div className="ml-auto flex items-center gap-2">
                      <span className="text-xs text-muted-foreground">Filter</span>
                      <Select value={benchmarkFilter} onValueChange={(v) => setBenchmarkFilter(v as 'all' | 'ok' | 'fail')}>
                        <SelectTrigger className="h-8 w-[110px]"><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">All</SelectItem>
                          <SelectItem value="ok">Success</SelectItem>
                          <SelectItem value="fail">Failures</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="pt-0">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Time</TableHead>
                        <TableHead>Node</TableHead>
                        <TableHead>Probe</TableHead>
                        <TableHead>Model</TableHead>
                        <TableHead className="text-right">TTFT</TableHead>
                        <TableHead className="text-right">TPS</TableHead>
                        <TableHead>Status</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredBenchmarks.slice(0, 100).map((b, i) => (
                        <TableRow key={`${b.source_node_id}-${b.ts}-${i}`}>
                          <TableCell className="text-xs text-muted-foreground">{fmtTime(b.ts)}</TableCell>
                          <TableCell className="font-mono text-xs">{b.source_node_id}</TableCell>
                          <TableCell>{b.probe_name}</TableCell>
                          <TableCell className="max-w-[180px] truncate">{shortName(b.model)}</TableCell>
                          <TableCell className="text-right">{fmtMs(b.ttft_ms)}</TableCell>
                          <TableCell className="text-right">{fmtRate(b.tokens_per_sec)}</TableCell>
                          <TableCell>
                            <Badge className={b.success ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300' : 'border-red-500/30 bg-red-500/10 text-red-200'}>
                              {b.success ? 'ok' : (b.error_kind || 'fail')}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      ))}
                      {!filteredBenchmarks.length ? (
                        <TableRow><TableCell colSpan={7} className="text-muted-foreground">No benchmark runs in range.</TableCell></TableRow>
                      ) : null}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      <Card className="flex min-h-0 flex-col overflow-hidden border-border/80 bg-card/85 backdrop-blur">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Telemetry Summary</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 pt-0">
          <MetricLine label="Active requests" value={`${telemetry?.live.active_requests ?? 0}`} />
          <MetricLine label="Peak active" value={`${telemetry?.live.active_requests_peak ?? 0}`} />
          <MetricLine label="Requests local / remote" value={`${telemetry?.live.requests_local ?? 0} / ${telemetry?.live.requests_remote ?? 0}`} />
          <MetricLine label="Utilization" value={fmtPct(telemetry?.live.utilization_pct)} />
          <MetricLine label="Tunnel bytes" value={fmtBytes(telemetry?.live.tunnel_bytes_total)} />
          <MetricLine label="Rollup p95 exact" value={fmtMs(latestRollup?.latency_p95_ms_exact)} />
          <MetricLine label="Rollup p99 exact" value={fmtMs(latestRollup?.latency_p99_ms_exact)} />
          <MetricLine label="Rows (nodes/rollup/bench)" value={`${telemetry?.nodes.length ?? 0}/${telemetry?.rollup.length ?? 0}/${telemetry?.benchmarks.length ?? 0}`} />
        </CardContent>
      </Card>
    </div>
  );
}

function MeshTopologyDiagram({ status, nodes, selectedModel }: { status: StatusPayload | null; nodes: TopologyNode[]; selectedModel: string }) {
  if (!status || !nodes.length) return <EmptyPanel text="No topology data yet." />;

  const totalGpuVram = nodes.filter((n) => !n.client).reduce((s, n) => s + (n.vram || 0), 0);
  const ok = !!status.llama_ready;
  const W = 760;
  const nW = 180;
  const nH = 96;

  if (nodes.length === 1) {
    return (
      <svg viewBox={`0 0 ${W} ${nH + 20}`} className="w-full" role="img" aria-label="Mesh topology">
        {renderNodeBox({ x: W / 2 - nW / 2, y: 10, node: nodes[0], totalGpuVram, ok, selectedModel, status, w: nW, h: nH })}
      </svg>
    );
  }

  const host = nodes.find((n) => n.host) || nodes[0];
  const workers = nodes.filter((n) => n !== host);
  const hX = W / 2 - nW / 2;
  const hY = 14;
  const wY = nH + 70;
  const gap = Math.min(220, Math.max(190, (W - 40) / Math.max(1, workers.length)));
  const wStart = W / 2 - ((workers.length - 1) * gap) / 2 - nW / 2;

  return (
    <svg viewBox={`0 0 ${W} ${wY + nH + 20}`} className="w-full" role="img" aria-label="Mesh topology">
      {workers.map((w, i) => {
        const wx = wStart + i * gap;
        const x1 = hX + nW / 2;
        const y1 = hY + nH;
        const x2 = wx + nW / 2;
        const y2 = wY;
        const mx = (x1 + x2) / 2;
        const my = (y1 + y2) / 2;
        return (
          <g key={`edge-${w.id}`}>
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={ok ? '#1f3b2f' : '#293241'} strokeWidth={1.5} strokeDasharray={ok ? undefined : '4 4'} />
            {ok ? (
              <>
                <circle r="2" fill="#4ade80" opacity="0.8">
                  <animateMotion dur="1.8s" repeatCount="indefinite" path={`M${x1},${y1} L${x2},${y2}`} />
                </circle>
                <circle r="2" fill="#60a5fa" opacity="0.7">
                  <animateMotion dur="1.8s" repeatCount="indefinite" path={`M${x2},${y2} L${x1},${y1}`} />
                </circle>
              </>
            ) : null}
            <text x={mx} y={my - 4} textAnchor="middle" fill="#64748b" fontSize="9" fontFamily="monospace">
              {w.client ? 'QUIC · HTTP' : 'QUIC · RPC'}
            </text>
          </g>
        );
      })}

      {renderNodeBox({ x: hX, y: hY, node: host, totalGpuVram, ok, selectedModel, status, w: nW, h: nH })}
      {workers.map((w, i) => renderNodeBox({ x: wStart + i * gap, y: wY, node: w, totalGpuVram, ok, selectedModel, status, w: nW, h: nH }))}
    </svg>
  );
}

function renderNodeBox(args: {
  x: number;
  y: number;
  node: TopologyNode;
  totalGpuVram: number;
  ok: boolean;
  selectedModel: string;
  status: StatusPayload;
  w: number;
  h: number;
}) {
  const { x, y, node, totalGpuVram, ok, selectedModel, status, w, h } = args;
  const pct = node.client ? 0 : totalGpuVram > 0 ? Math.round((node.vram / totalGpuVram) * 100) : 100;
  const servingSel = !node.client && !!selectedModel && node.serving === selectedModel;
  const active = ok && servingSel;
  const fill = active ? '#0c1f17' : node.self ? '#0c1728' : '#0b1220';
  const stroke = active ? '#22c55e' : ok ? (node.host ? '#1f6b48' : node.client ? '#334155' : '#1d4f91') : '#233047';
  const role = node.client ? 'CLIENT' : node.serving ? shortName(node.serving) : 'IDLE';
  const roleColor = node.client ? '#94a3b8' : servingSel ? '#4ade80' : '#60a5fa';
  const label = node.self ? `${node.id} (you)` : node.id;
  const service = node.client ? 'API tunnel' : node.host ? 'llama-server' : 'rpc-server';
  const svcColor = active ? '#86efac' : '#64748b';
  const nodeModel = (status.mesh_models || []).find((m) => m.name === node.serving);
  const modelGB = nodeModel ? nodeModel.size_gb : node.self ? status.model_size_gb || 0 : 0;
  const usagePct = !node.client && node.vram > 0 && modelGB > 0 ? Math.min(100, Math.round((modelGB / node.vram) * 100)) : 0;
  const vramLabel = node.vram > 0 ? `${node.vram.toFixed(0)} GB` : '';
  const usageLabel = modelGB > 0 ? `${modelGB.toFixed(1)}GB model` : '';
  const barX = x + 16;
  const barW = w - 32;
  const barY = y + 62;
  const bar2Y = barY + 10;

  return (
    <g key={`node-${node.id}`}>
      <rect x={x} y={y} width={w} height={h} rx={10} fill={fill} stroke={stroke} strokeWidth={active ? 2 : 1.4} />
      <text x={x + w / 2} y={y + 16} textAnchor="middle" fill={roleColor} fontSize="9" fontFamily="monospace" fontWeight={600}>{role}</text>
      <text x={x + w / 2} y={y + 30} textAnchor="middle" fill="#bfdbfe" fontSize="10" fontFamily="monospace">{label}</text>
      <text x={x + w / 2} y={y + 43} textAnchor="middle" fill={svcColor} fontSize="8" fontFamily="monospace">{service}</text>

      <rect x={barX} y={barY} width={barW} height={5} rx={2} fill="#111827" />
      <rect x={barX} y={barY} width={(barW * pct) / 100} height={5} rx={2} fill={servingSel ? '#15803d' : '#1d4ed8'} />
      <rect x={barX} y={bar2Y} width={barW} height={5} rx={2} fill="#111827" />
      {usagePct > 0 ? <rect x={barX} y={bar2Y} width={(barW * usagePct) / 100} height={5} rx={2} fill={usagePct > 80 ? '#ea580c' : '#a16207'} /> : null}

      <text x={barX} y={y + h - 8} fill="#64748b" fontSize="7.5" fontFamily="monospace">
        {[vramLabel, usageLabel].filter(Boolean).join(' · ')}
      </text>
      <text x={barX + barW} y={y + h - 8} textAnchor="end" fill="#64748b" fontSize="7.5" fontFamily="monospace">
        {usagePct > 0 ? `${usagePct}% used` : pct > 0 && !node.client ? `${pct}%` : ''}
      </text>

      {node.host && ok ? (
        <g>
          <rect x={x + w - 54} y={y + 6} width={40} height={13} rx={4} fill="#072012" stroke="#166534" strokeWidth={0.8} />
          <text x={x + w - 34} y={y + 15} textAnchor="middle" fill="#86efac" fontSize="7.5" fontFamily="monospace">
            :{status.api_port || 9337}
          </text>
        </g>
      ) : null}
    </g>
  );
}

function MetricChartCard<T>({
  title,
  rows,
  value,
  color,
  format,
}: {
  title: string;
  rows: T[];
  value: (row: T) => number;
  color: string;
  format: (n: number) => string;
}) {
  const vals = rows.map(value).filter((v) => Number.isFinite(v));
  const latest = vals.length ? vals[vals.length - 1] : 0;
  return (
    <Card className="shadow-none">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-sm">{title}</CardTitle>
          <div className="text-xs text-muted-foreground">{format(latest)}</div>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <Sparkline values={vals} color={color} />
      </CardContent>
    </Card>
  );
}

function DualMetricChartCard<T>({
  title,
  rows,
  aValue,
  bValue,
  aLabel,
  bLabel,
  aColor,
  bColor,
  format,
}: {
  title: string;
  rows: T[];
  aValue: (row: T) => number;
  bValue: (row: T) => number;
  aLabel: string;
  bLabel: string;
  aColor: string;
  bColor: string;
  format: (n: number) => string;
}) {
  const aVals = rows.map(aValue).filter((v) => Number.isFinite(v));
  const bVals = rows.map(bValue).filter((v) => Number.isFinite(v));
  const aLatest = aVals.length ? aVals[aVals.length - 1] : 0;
  const bLatest = bVals.length ? bVals[bVals.length - 1] : 0;

  return (
    <Card className="shadow-none">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-sm">{title}</CardTitle>
          <div className="text-right text-xs text-muted-foreground">
            <div>{aLabel}: {format(aLatest)}</div>
            <div>{bLabel}: {format(bLatest)}</div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-2 pt-0">
        <DualSparkline aValues={aVals} bValues={bVals} aColor={aColor} bColor={bColor} />
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <LegendDot color={aColor} /> {aLabel}
          <LegendDot color={bColor} /> {bLabel}
        </div>
      </CardContent>
    </Card>
  );
}

function Sparkline({ values, color }: { values: number[]; color: string }) {
  const W = 320;
  const H = 72;
  if (!values.length) {
    return <div className="flex h-[72px] items-center justify-center text-xs text-muted-foreground">No data</div>;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(1, max - min);
  const points = values.map((v, i) => {
    const x = (i / Math.max(1, values.length - 1)) * (W - 8) + 4;
    const y = H - 6 - ((v - min) / range) * (H - 12);
    return `${x},${y}`;
  });
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-[72px] w-full" role="img" aria-label="sparkline">
      <rect x="0" y="0" width={W} height={H} rx="8" fill="#0b1220" />
      <polyline points={points.join(' ')} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

function DualSparkline({
  aValues,
  bValues,
  aColor,
  bColor,
}: {
  aValues: number[];
  bValues: number[];
  aColor: string;
  bColor: string;
}) {
  const W = 320;
  const H = 72;
  const maxLen = Math.max(aValues.length, bValues.length);
  if (!maxLen) return <div className="flex h-[72px] items-center justify-center text-xs text-muted-foreground">No data</div>;
  const a = normalizeLen(aValues, maxLen);
  const b = normalizeLen(bValues, maxLen);
  const all = [...a, ...b].filter((v) => Number.isFinite(v));
  const min = all.length ? Math.min(...all) : 0;
  const max = all.length ? Math.max(...all) : 1;
  const range = Math.max(1, max - min);
  const pts = (vals: number[]) =>
    vals.map((v, i) => {
      const x = (i / Math.max(1, vals.length - 1)) * (W - 8) + 4;
      const y = H - 6 - ((v - min) / range) * (H - 12);
      return `${x},${y}`;
    }).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-[72px] w-full" role="img" aria-label="dual sparkline">
      <rect x="0" y="0" width={W} height={H} rx="8" fill="#0b1220" />
      <polyline points={pts(a)} fill="none" stroke={aColor} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
      <polyline points={pts(b)} fill="none" stroke={bColor} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" opacity="0.9" />
    </svg>
  );
}

function LegendDot({ color }: { color: string }) {
  return <span className="inline-block h-2.5 w-2.5 rounded-full align-middle" style={{ backgroundColor: color }} />;
}

function normalizeLen(values: number[], len: number) {
  if (values.length === len) return values;
  if (values.length === 0) return Array.from({ length: len }, () => 0);
  if (values.length > len) return values.slice(values.length - len);
  const pad = Array.from({ length: len - values.length }, () => values[0]);
  return [...pad, ...values];
}

function ChatBubble({
  message,
  reasoningOpen,
  onReasoningToggle,
}: {
  message: ChatMessage;
  reasoningOpen: boolean;
  onReasoningToggle: (open: boolean) => void;
}) {
  const isUser = message.role === 'user';
  return (
    <div className={cn('flex', isUser ? 'justify-end' : 'justify-start')}>
      <div className="w-full max-w-[92%] md:max-w-[82%]">
        <div className="mb-1 flex items-center gap-2 px-1 text-xs text-muted-foreground">
          {isUser ? <User className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5" />}
          <span>{isUser ? 'You' : 'Assistant'}</span>
          {message.model ? <span>· {shortName(message.model)}</span> : null}
        </div>
        <div
          className={cn(
            'rounded-2xl border px-4 py-3 text-sm leading-6 shadow-sm whitespace-pre-wrap',
            isUser
              ? 'border-primary/15 bg-primary text-primary-foreground'
              : message.error
                ? 'border-red-500/30 bg-red-500/10 text-red-200'
                : 'border-border bg-card',
          )}
        >
          {message.content || (!isUser ? '...' : '')}
        </div>
        {message.reasoning ? (
          <Card className="mt-2 shadow-none">
            <CardContent className="p-3">
              <Accordion type="single" collapsible value={reasoningOpen ? 'reasoning' : ''} onValueChange={(v) => onReasoningToggle(v === 'reasoning')}>
                <AccordionItem value="reasoning" className="border-b-0">
                  <AccordionTrigger className="py-0 text-xs">Reasoning</AccordionTrigger>
                  <AccordionContent>
                    <div className="mt-2 whitespace-pre-wrap text-xs leading-5 text-muted-foreground">{message.reasoning}</div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </CardContent>
          </Card>
        ) : null}
        {message.stats ? <div className="mt-1 px-1 text-xs text-muted-foreground">{message.stats}</div> : null}
      </div>
    </div>
  );
}

function InvitePopover({ token, selectedModel }: { token: string; selectedModel: string }) {
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const hasToken = !!token;
  const joinCommand = hasToken && selectedModel ? `mesh-llm --join ${token} --model ${selectedModel}` : '';
  const clientCommand = hasToken ? `mesh-llm --client --join ${token}` : '';

  async function copyText(key: string, text: string) {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopiedKey(key);
      window.setTimeout(() => setCopiedKey((prev) => (prev === key ? null : prev)), 1500);
    } catch {
      setCopiedKey(null);
    }
  }

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="default" size="sm" disabled={!hasToken} className="shadow-sm">
          Invite
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-[28rem] space-y-3">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
            <Sparkles className="h-5 w-5" />
          </div>
          <div>
            <div className="text-sm font-semibold text-foreground">Invite a friend to Mesh LLM</div>
            <div className="mt-0.5 text-xs text-muted-foreground">
              Open models and excess compute already exist. Invite someone and coordinate capacity.
            </div>
          </div>
        </div>
        <Separator />
        {hasToken ? (
          <div className="space-y-3">
            <div className="rounded-lg border border-border/70 bg-muted/20 p-3">
              <div className="mb-2 text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">Join Command</div>
              <div className="flex items-center gap-2">
                <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap rounded bg-background px-2 py-1.5 font-mono text-xs text-foreground">
                  {joinCommand || `mesh-llm --join ${token}`}
                </code>
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={() => void copyText('join', joinCommand || `mesh-llm --join ${token}`)}
                >
                  {copiedKey === 'join' ? <Check className="mr-1 h-3.5 w-3.5" /> : <Copy className="mr-1 h-3.5 w-3.5" />}
                  {copiedKey === 'join' ? 'Copied' : 'Copy'}
                </Button>
              </div>
            </div>

            <div className="rounded-md border border-border/70 bg-muted/10 p-2">
              <div className="mb-1 text-xs font-medium text-muted-foreground">Client only (no GPU)</div>
              <div className="flex items-center gap-2">
                <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap rounded bg-background px-2 py-1 font-mono text-xs">
                  {clientCommand}
                </code>
                <Button type="button" variant="secondary" size="sm" onClick={() => void copyText('client', clientCommand)}>
                  {copiedKey === 'client' ? <Check className="mr-1 h-3.5 w-3.5" /> : <Copy className="mr-1 h-3.5 w-3.5" />}
                  {copiedKey === 'client' ? 'Copied' : 'Copy'}
                </Button>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-xs text-muted-foreground">Invite token not available yet. Start or host a mesh first.</div>
        )}
      </PopoverContent>
    </Popover>
  );
}

function ApiEndpointPopover({ port }: { port: number | null }) {
  const [copied, setCopied] = useState(false);
  const hasPort = Number.isFinite(port);
  const host =
    typeof window !== 'undefined' && window.location?.hostname
      ? window.location.hostname
      : 'localhost';
  const protocol =
    typeof window !== 'undefined' && window.location?.protocol
      ? window.location.protocol
      : 'http:';
  const baseUrl = hasPort ? `${protocol}//${host}:${port}/v1` : '';

  async function copy() {
    if (!baseUrl) return;
    try {
      await navigator.clipboard.writeText(baseUrl);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="secondary" size="sm" disabled={!hasPort}>
          API Endpoint
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-[26rem] space-y-3">
        <div>
          <div className="text-sm font-semibold text-foreground">OpenAI-Compatible API Endpoint</div>
          <div className="mt-1 text-xs text-muted-foreground">
            Use this as your base URL for tools and clients.
          </div>
        </div>
        <Separator />
        {hasPort ? (
          <div className="rounded-lg border border-border/70 bg-muted/20 p-3">
            <div className="mb-2 text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">
              Base URL
            </div>
            <div className="flex items-center gap-2">
              <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap rounded bg-background px-2 py-1.5 font-mono text-xs text-foreground">
                {baseUrl}
              </code>
              <Button type="button" variant="secondary" size="sm" onClick={() => void copy()}>
                {copied ? <Check className="mr-1 h-3.5 w-3.5" /> : <Copy className="mr-1 h-3.5 w-3.5" />}
                {copied ? 'Copied' : 'Copy'}
              </Button>
            </div>
          </div>
        ) : (
          <div className="text-xs text-muted-foreground">API port not available yet.</div>
        )}
      </PopoverContent>
    </Popover>
  );
}

function StatusBadge({ ready }: { ready: boolean }) {
  return (
    <Badge className={ready ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300' : ''}>
      <Circle className={cn('mr-1 h-2.5 w-2.5 fill-current', ready ? 'text-emerald-500' : 'text-zinc-500')} />
      {ready ? 'LLM Ready' : 'Starting'}
    </Badge>
  );
}

function SectionLabel({ children }: { children: ReactNode }) {
  return <div className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">{children}</div>;
}

function StatCard({ title, value, icon }: { title: string; value: string; icon: ReactNode }) {
  return (
    <Card className="shadow-none">
      <CardContent className="p-3">
        <div className="mb-2 flex items-center gap-2 text-muted-foreground">{icon}<span className="text-xs">{title}</span></div>
        <div className="text-sm font-semibold text-foreground">{value}</div>
      </CardContent>
    </Card>
  );
}

function MetricLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between rounded-md border border-border/70 bg-muted/40 px-3 py-2">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="font-mono text-xs text-foreground">{value}</span>
    </div>
  );
}

function EmptyPanel({ text }: { text: string }) {
  return (
    <Card className="shadow-none">
      <CardContent className="p-4 text-sm text-muted-foreground">{text}</CardContent>
    </Card>
  );
}

function meshGpuVram(status: StatusPayload | null) {
  if (!status) return 0;
  return (status.is_client ? 0 : status.my_vram_gb || 0) + (status.peers || []).filter((p) => p.role !== 'Client').reduce((s, p) => s + p.vram_gb, 0);
}

function aggregateNodeHistorySeries(
  rows: NodeMetricRow[],
  pick: (row: NodeMetricRow) => number | null | undefined,
): SeriesPoint[] {
  const byMinute = new Map<number, { sum: number; count: number }>();
  for (const row of rows) {
    const n = Number(pick(row));
    if (!Number.isFinite(n)) continue;
    const agg = byMinute.get(row.ts_minute) ?? { sum: 0, count: 0 };
    agg.sum += n;
    agg.count += 1;
    byMinute.set(row.ts_minute, agg);
  }
  return [...byMinute.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([ts_minute, agg]) => ({ ts_minute, value: agg.sum / Math.max(agg.count, 1) }));
}

function shortName(name: string) {
  return (name || '').replace(/-Q\w+$/, '').replace(/-Instruct/, '');
}

function fmtMs(value?: number | null) {
  return Number.isFinite(Number(value)) ? `${Math.round(Number(value))}ms` : 'n/a';
}

function fmtRate(value?: number | null) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(1) : 'n/a';
}

function fmtPct(value?: number | null) {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(0)}%` : 'n/a';
}

function fmtTime(tsSec?: number | null) {
  if (!Number.isFinite(Number(tsSec))) return 'n/a';
  return new Date(Number(tsSec) * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function fmtBytes(value?: number | null) {
  if (!Number.isFinite(Number(value))) return 'n/a';
  const n = Number(value);
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}
