import { type ReactNode, useEffect, useMemo, useState } from 'react';
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
} from '@xyflow/react';
import {
  BarChart3,
  Bot,
  Check,
  Circle,
  Copy,
  Cpu,
  Gauge,
  Link2,
  Loader2,
  MonitorCog,
  Moon,
  Network,
  Send,
  Sparkles,
  Sun,
  User,
  Wifi,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';

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
import { BrandIcon } from './components/brand-icon';
import { MeshLlmWordmark } from './components/mesh-llm-wordmark';
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

type ThemeMode = 'auto' | 'light' | 'dark';

const THEME_STORAGE_KEY = 'mesh-llm-theme';

function readThemeMode(): ThemeMode {
  if (typeof window === 'undefined') return 'auto';
  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  return stored === 'light' || stored === 'dark' || stored === 'auto' ? stored : 'auto';
}

function applyThemeMode(mode: ThemeMode) {
  if (typeof window === 'undefined') return;
  const media = window.matchMedia('(prefers-color-scheme: dark)');
  const dark = mode === 'dark' || (mode === 'auto' && media.matches);
  document.documentElement.classList.toggle('dark', dark);
  document.documentElement.style.colorScheme = mode === 'auto' ? 'light dark' : dark ? 'dark' : 'light';
}

function nextThemeMode(mode: ThemeMode): ThemeMode {
  if (mode === 'auto') return 'light';
  if (mode === 'light') return 'dark';
  return 'auto';
}

export function App() {
  const [section, setSection] = useState<TopSection>('chat');
  const [themeMode, setThemeMode] = useState<ThemeMode>(() => readThemeMode());
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
  const [apiCopied, setApiCopied] = useState(false);
  const chatScrollRef = useRef<HTMLDivElement>(null);

  const warmModels = useMemo(() => {
    const list = (status?.mesh_models ?? []).filter((m) => m.status === 'warm').map((m) => m.name);
    if (!list.length && status?.model_name) list.push(status.model_name);
    return list;
  }, [status]);

  const openApiUrl = useMemo(() => {
    if (!status?.api_port) return '';
    const host =
      typeof window !== 'undefined' && window.location?.hostname
        ? window.location.hostname
        : 'localhost';
    const protocol =
      typeof window !== 'undefined' && window.location?.protocol
        ? window.location.protocol
        : 'http:';
    return `${protocol}//${host}:${status.api_port}/v1`;
  }, [status?.api_port]);

  useEffect(() => {
    if (!warmModels.length) return;
    if (!selectedModel || !warmModels.includes(selectedModel)) setSelectedModel(warmModels[0]);
  }, [warmModels, selectedModel]);

  useEffect(() => {
    applyThemeMode(themeMode);
    window.localStorage.setItem(THEME_STORAGE_KEY, themeMode);
  }, [themeMode]);

  useEffect(() => {
    if (themeMode !== 'auto') return;
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = () => applyThemeMode('auto');
    media.addEventListener('change', onChange);
    return () => media.removeEventListener('change', onChange);
  }, [themeMode]);

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

    loadStatus();

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

    return () => {
      stop = true;
      statusEvents.close();
    };
  }, []);

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

  async function copyOpenApiUrl() {
    if (!openApiUrl) return;
    try {
      await navigator.clipboard.writeText(openApiUrl);
      setApiCopied(true);
      window.setTimeout(() => setApiCopied(false), 1400);
    } catch {
      setApiCopied(false);
    }
  }

  return (
    <div className="min-h-screen overflow-x-hidden overflow-y-auto bg-grid [background-size:18px_18px]">
      <div className="mx-auto flex min-h-screen w-full max-w-[1680px] flex-col gap-4 p-4 md:p-5">
        <Card className="relative overflow-hidden border-border/80 bg-card/85 backdrop-blur">
          <div className="pointer-events-none absolute -right-6 -top-8 text-primary/10">
            <BrandIcon className="h-28 w-28 rotate-12" />
          </div>
          <div className="pointer-events-none absolute -bottom-8 -left-6 text-accent/10">
            <BrandIcon className="h-24 w-24 -rotate-12" />
          </div>
          <CardContent className="relative flex flex-wrap items-center gap-3 p-4">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-border bg-background/80 text-primary shadow-sm">
              <BrandIcon className="h-6 w-6" />
            </div>
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold tracking-tight text-foreground">
                <MeshLlmWordmark />
              </div>
              <div className="mt-1 inline-flex max-w-full items-center gap-1.5 rounded-full border border-border/70 bg-background/70 px-2.5 py-1 text-[11px] text-muted-foreground">
                <span className="truncate">{status?.mesh_name ? `Mesh ${status.mesh_name}` : 'Local Dashboard'}</span>
                {status?.node_id ? <span aria-hidden="true">·</span> : null}
                {status?.node_id ? <span className="truncate font-mono">Node {status.node_id}</span> : null}
                {status ? <span aria-hidden="true">·</span> : null}
                {status ? <span>{status.is_host ? 'Host' : status.is_client ? 'Client' : 'Worker'}</span> : null}
              </div>
            </div>
            <Tabs value={section} onValueChange={(v) => setSection(v as TopSection)} className="ml-auto">
              <TabsList className="h-auto gap-2 rounded-none border-0 bg-transparent p-0 text-sm">
                <TabsTrigger
                  value="chat"
                  className="h-auto rounded-md px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-none"
                >
                  Chat
                </TabsTrigger>
                <TabsTrigger
                  value="mesh"
                  className="h-auto rounded-md px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-none"
                >
                  Mesh
                </TabsTrigger>
              </TabsList>
            </Tabs>
            <div className="flex flex-wrap items-center gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-9 bg-background/70"
                title={openApiUrl ? `Copy ${openApiUrl}` : 'API URL unavailable'}
                onClick={() => void copyOpenApiUrl()}
                disabled={!openApiUrl}
              >
                {apiCopied ? <Check className="mr-1 h-3.5 w-3.5" /> : <Link2 className="mr-1 h-3.5 w-3.5" />}
                {apiCopied ? 'Copied' : 'API URL'}
              </Button>
              <Button
                type="button"
                variant="outline"
                size="icon"
                className="h-9 w-9 bg-background/70"
                title={`Theme: ${themeMode} (click to cycle)`}
                aria-label={`Theme ${themeMode}. Click to cycle Auto, Light, Dark`}
                onClick={() => setThemeMode((prev) => nextThemeMode(prev))}
              >
                {themeMode === 'auto' ? <MonitorCog className="h-4 w-4" /> : null}
                {themeMode === 'light' ? <Sun className="h-4 w-4" /> : null}
                {themeMode === 'dark' ? <Moon className="h-4 w-4" /> : null}
              </Button>
              <Badge>
                <Network className="mr-1 h-3.5 w-3.5" />
                {`${nodeCount} node${nodeCount === 1 ? '' : 's'}`}
              </Badge>
              <Badge>
                <Sparkles className="mr-1 h-3.5 w-3.5" />
                {`${availableModelCount} model${availableModelCount === 1 ? '' : 's'}`}
              </Badge>
              <InvitePopover token={status?.token ?? ''} selectedModel={selectedModel || warmModels[0] || status?.model_name || ''} />
              <StatusBadge ready={!!status?.llama_ready} />
              {statusError ? <Badge className="border-amber-500/30 bg-amber-500/10 text-amber-200">{statusError}</Badge> : null}
            </div>
          </CardContent>
        </Card>

        <div className="flex-1">
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
            <MeshPage
              status={status}
              metrics={metrics}
              topologyNodes={topologyNodes}
              selectedModel={selectedModel || status?.model_name || ''}
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
  const [modelFilter, setModelFilter] = useState<'all' | 'warm' | 'cold'>('all');
  const filteredModels = useMemo(() => {
    const models = status?.mesh_models ?? [];
    return [...models]
      .filter((m) => (modelFilter === 'all' ? true : m.status === modelFilter))
      .sort((a, b) => (b.node_count - a.node_count) || a.name.localeCompare(b.name));
  }, [status?.mesh_models, modelFilter]);

  return (
    <div className="grid h-full min-h-0 gap-4">
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
          <div className="grid h-full min-h-0 gap-4 lg:grid-rows-[auto_320px_minmax(0,1fr)] xl:grid-rows-[auto_360px_minmax(0,1fr)]">
            <Card className="shadow-none">
              <CardContent className="space-y-3 p-3">
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
                  <div className="flex items-center gap-2 rounded-md border border-border/70 bg-muted/30 px-3 py-2">
                    <Network className="h-4 w-4 text-primary" />
                    <div className="text-xs">
                      <div className="text-muted-foreground">Nodes</div>
                      <div className="font-semibold text-foreground">{topologyNodes.length}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 rounded-md border border-border/70 bg-muted/30 px-3 py-2">
                    <Sparkles className="h-4 w-4 text-emerald-500" />
                    <div className="text-xs">
                      <div className="text-muted-foreground">Warm Models</div>
                      <div className="font-semibold text-foreground">{(status?.mesh_models ?? []).filter((m) => m.status === 'warm').length}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 rounded-md border border-border/70 bg-muted/30 px-3 py-2">
                    <Cpu className="h-4 w-4 text-cyan-500" />
                    <div className="text-xs">
                      <div className="text-muted-foreground">Mesh VRAM</div>
                      <div className="font-semibold text-foreground">{meshGpuVram(status).toFixed(1)} GB</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 rounded-md border border-border/70 bg-muted/30 px-3 py-2">
                    <Gauge className="h-4 w-4 text-amber-500" />
                    <div className="text-xs">
                      <div className="text-muted-foreground">Inflight</div>
                      <div className="font-semibold text-foreground">{metrics?.requests_inflight ?? 0}</div>
                    </div>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <Badge className="gap-1 border-border/70 bg-background/70">
                    <Circle className="h-3 w-3 fill-current" />
                    {status?.node_status ?? 'n/a'}
                  </Badge>
                  <Badge className="gap-1 border-border/70 bg-background/70">
                    <Sparkles className="h-3 w-3" />
                    {selectedModel ? shortName(selectedModel) : 'n/a'}
                  </Badge>
                  <Badge className="gap-1 border-border/70 bg-background/70">
                    <Gauge className="h-3 w-3" />
                    TTFT {fmtMs(metrics?.p95_ttft_ms)}
                  </Badge>
                  <Badge className="gap-1 border-border/70 bg-background/70">
                    <Wifi className="h-3 w-3" />
                    Rate {fmtRate(metrics?.p95_tokens_per_sec)}
                  </Badge>
                  <Badge className="gap-1 border-border/70 bg-background/70">
                    <Cpu className="h-3 w-3" />
                    {fmtBytes(metrics?.local_bytes_tx)} / {fmtBytes(metrics?.local_bytes_rx)}
                  </Badge>
                </div>
              </CardContent>
            </Card>
            <Card className="shadow-none overflow-hidden">
              <CardContent className="p-4">
                <MeshTopologyDiagram
                  status={status}
                  nodes={topologyNodes}
                  selectedModel={selectedModel}
                />
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
                  <div className="flex items-center gap-2">
                    <CardTitle className="text-sm">Models</CardTitle>
                    <div className="ml-auto flex items-center gap-2">
                      <span className="text-xs text-muted-foreground">Filter</span>
                      <Select value={modelFilter} onValueChange={(v) => setModelFilter(v as 'all' | 'warm' | 'cold')}>
                        <SelectTrigger className="h-8 w-[110px]">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">All</SelectItem>
                          <SelectItem value="warm">Warm</SelectItem>
                          <SelectItem value="cold">Cold</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="min-h-0 pt-0">
                  <ScrollArea className="h-[28rem] pr-3 lg:h-full">
                    <div className="space-y-2 pr-2">
                      {filteredModels.length > 0 ? (
                        filteredModels.map((model) => (
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
                        <EmptyPanel text={(status?.mesh_models.length ?? 0) > 0 ? `No ${modelFilter} models.` : 'No mesh model metadata yet.'} />
                      )}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>
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

type PositionedTopologyNode = TopologyNode & {
  x: number;
  y: number;
  bucket: 'center' | 'serving' | 'worker' | 'client';
};

type TopologyNodeInfo = {
  role: string;
  servingLabel: string;
  fullServing: string;
  vramSharePct: number;
  modelUsagePct: number;
  modelGb: number;
};

type TopologyFlowNodeData = {
  node: PositionedTopologyNode;
  info: TopologyNodeInfo;
  selected: boolean;
};

function TopologyFlowNode({ data }: NodeProps<TopologyFlowNodeData>) {
  const isCenter = data.node.bucket === 'center';
  const dotClass = isCenter
    ? 'bg-blue-500 border-blue-300'
    : data.node.bucket === 'serving'
      ? 'bg-emerald-500 border-emerald-300'
      : data.node.bucket === 'worker'
        ? 'bg-cyan-500 border-cyan-300'
        : 'bg-slate-400 border-slate-200';
  const vramWidth = Math.max(0, Math.min(100, data.info.vramSharePct));
  const modelWidth = Math.max(0, Math.min(100, data.info.modelUsagePct));

  return (
    <div className="w-[208px]">
      <Handle type="target" position={Position.Top} style={{ opacity: 0, width: 1, height: 1, border: 0, pointerEvents: 'none' }} />
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0, width: 1, height: 1, border: 0, pointerEvents: 'none' }} />

      <div className={cn('mx-auto h-7 w-7 rounded-full border-2 shadow-sm', dotClass)} />
      <div className="mt-1 break-all text-center font-mono text-[10px] leading-3 text-foreground">
        {data.node.self ? `${data.node.id} (you)` : data.node.id}
      </div>

      <div
        className={cn(
          'mt-1 rounded-md border bg-card/95 px-2 py-1.5 shadow-sm backdrop-blur',
          data.selected ? 'border-ring ring-1 ring-ring/50' : 'border-border/90',
        )}
      >
        <div className="truncate font-mono text-[10px] leading-3 text-foreground/90">
          {data.info.role} · {data.info.servingLabel}
        </div>

        <div className="mt-1 flex items-center gap-1">
          <span className="text-[9px] text-sky-500">●</span>
          <div className="h-1 flex-1 rounded bg-muted">
            <div className="h-1 rounded bg-sky-500" style={{ width: `${vramWidth}%` }} />
          </div>
          <span className="font-mono text-[10px] text-sky-600 dark:text-sky-400">{data.info.vramSharePct}%</span>
        </div>

        <div className="mt-0.5 flex items-center gap-1">
          <span className="text-[9px] text-amber-500">◆</span>
          <div className="h-1 flex-1 rounded bg-muted">
            <div className="h-1 rounded bg-amber-500" style={{ width: `${modelWidth}%` }} />
          </div>
          <span className="font-mono text-[10px] text-amber-600 dark:text-amber-400">{data.info.modelUsagePct}%</span>
        </div>
      </div>
    </div>
  );
}

const topologyNodeTypes = { topologyNode: TopologyFlowNode };

function MeshTopologyDiagram({
  status,
  nodes,
  selectedModel,
}: {
  status: StatusPayload | null;
  nodes: TopologyNode[];
  selectedModel: string;
}) {
  if (!status || !nodes.length) return <EmptyPanel text="No topology data yet." />;

  const center = nodes.find((n) => n.host) || nodes.find((n) => n.self) || nodes[0];
  const others = nodes.filter((n) => n.id !== center.id).sort((a, b) => (b.vram - a.vram) || a.id.localeCompare(b.id));
  const focusModel = selectedModel || status.model_name || '';
  const serving = others.filter((n) => !n.client && !!n.serving && (!focusModel || n.serving === focusModel));
  const servingIds = new Set(serving.map((n) => n.id));
  const clients = others.filter((n) => n.client);
  const workers = others.filter((n) => !n.client && !servingIds.has(n.id));

  const total = nodes.length;
  const nodeRadius = total >= 500 ? 3.6 : total >= 280 ? 4.8 : total >= 160 ? 6.2 : total >= 90 ? 7.4 : total >= 50 ? 8.8 : 10.4;
  const positioned = layoutTopologyNodes(center, serving, workers, clients, nodeRadius);
  const maxCoord = positioned.reduce((m, p) => Math.max(m, Math.hypot(p.x, p.y)), 0);
  const frame = Math.max(220, maxCoord + 230);
  const clientEdgeStride = total > 320 ? 6 : total > 220 ? 4 : total > 120 ? 2 : 1;
  const gpuNodeCount = nodes.filter((n) => !n.client).length;
  const meshVramGb = nodes.filter((n) => !n.client).reduce((sum, n) => sum + Math.max(0, n.vram), 0);
  const servingCount = nodes.filter((n) => !n.client && n.serving && n.serving !== '(idle)').length;

  const [selectedNodeId, setSelectedNodeId] = useState(center.id);

  useEffect(() => {
    setSelectedNodeId((prev) => (nodes.some((n) => n.id === prev) ? prev : center.id));
  }, [nodes, center.id]);

  const modelSizeByName = useMemo(() => new Map((status.mesh_models ?? []).map((m) => [m.name, m.size_gb])), [status.mesh_models]);
  const nodeInfoById = useMemo(() => {
    const out = new Map<string, TopologyNodeInfo>();
    for (const node of nodes) {
      const servingModel = !node.client && node.serving && node.serving !== '(idle)' ? node.serving : '';
      const role = node.client ? 'Client' : node.host ? 'Host' : servingModel ? 'Worker' : 'Idle';
      const modelGb = servingModel
        ? (modelSizeByName.get(servingModel) ?? (node.self ? status.model_size_gb || 0 : 0))
        : 0;
      const vramSharePct = !node.client && meshVramGb > 0 ? Math.round((Math.max(0, node.vram) / meshVramGb) * 100) : 0;
      const modelUsagePct = !node.client && node.vram > 0 && modelGb > 0
        ? Math.min(100, Math.round((modelGb / node.vram) * 100))
        : 0;
      out.set(node.id, {
        role,
        servingLabel: node.client ? 'CLIENT' : servingModel ? shortName(servingModel) : 'IDLE',
        fullServing: servingModel,
        vramSharePct,
        modelUsagePct,
        modelGb,
      });
    }
    return out;
  }, [nodes, modelSizeByName, status.model_size_gb, meshVramGb]);
  const selectedInfo = nodeInfoById.get(selectedNodeId) ?? nodeInfoById.get(center.id);

  const flowNodes = useMemo<Node<TopologyFlowNodeData>[]>(() => {
    return positioned.map((p) => ({
      id: p.id,
      type: 'topologyNode',
      position: { x: p.x + frame, y: p.y + frame },
      origin: [0.5, 0],
      data: {
        node: p,
        info: nodeInfoById.get(p.id) ?? {
          role: 'Node',
          servingLabel: 'IDLE',
          fullServing: '',
          vramSharePct: 0,
          modelUsagePct: 0,
          modelGb: 0,
        },
        selected: p.id === selectedNodeId,
      },
      draggable: false,
      selectable: false,
      connectable: false,
      zIndex: p.id === center.id ? 10 : 1,
    }));
  }, [positioned, frame, nodeInfoById, selectedNodeId, center.id]);

  const flowEdges = useMemo<Edge[]>(() => {
    const outer = positioned.filter((p) => p.id !== center.id);
    return outer
      .filter((p, idx) => !(p.bucket === 'client' && idx % clientEdgeStride !== 0))
      .map((p) => {
        const stroke =
          p.bucket === 'serving'
            ? 'rgba(34,197,94,0.35)'
            : p.bucket === 'worker'
              ? 'rgba(56,189,248,0.3)'
              : 'rgba(148,163,184,0.22)';
        return {
          id: `edge-${center.id}-${p.id}`,
          source: center.id,
          target: p.id,
          type: 'straight',
          animated: status.llama_ready,
          style: {
            stroke,
            strokeWidth: 1.2,
            strokeDasharray: p.bucket === 'client' ? '4 5' : undefined,
          },
        };
      });
  }, [positioned, center.id, clientEdgeStride, status.llama_ready]);

  return (
    <div className="flex h-full flex-col gap-2">
      <div className="flex flex-wrap gap-2 text-[11px] text-muted-foreground">
        <Badge className="border-sky-500/30 bg-sky-500/10 text-sky-700 dark:text-sky-300">GPU nodes: {gpuNodeCount}</Badge>
        <Badge className="border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300">Serving: {servingCount}</Badge>
        <Badge className="border-blue-500/30 bg-blue-500/10 text-blue-700 dark:text-blue-300">Mesh VRAM: {meshVramGb.toFixed(1)} GB</Badge>
        <Badge className="border-zinc-500/30 bg-zinc-500/10 text-zinc-700 dark:text-zinc-300">Clients: {clients.length}</Badge>
        {focusModel ? (
          <Badge className="border-violet-500/30 bg-violet-500/10 text-violet-700 dark:text-violet-300">
            Focus: {shortName(focusModel)}
          </Badge>
        ) : null}
        <Badge className="ml-auto border-border/70 bg-background/70 text-muted-foreground">Pan: drag · Zoom: controls</Badge>
      </div>

      <div className="h-[220px] md:h-[240px] lg:h-[250px] xl:h-[280px] overflow-hidden rounded-lg border border-border/70 bg-muted/10">
        <ReactFlow
          nodes={flowNodes}
          edges={flowEdges}
          nodeTypes={topologyNodeTypes}
          fitView
          fitViewOptions={{ padding: 0.22, maxZoom: 1.05 }}
          minZoom={0.2}
          maxZoom={1.6}
          zoomOnScroll={false}
          zoomOnPinch={false}
          panOnScroll={false}
          panOnDrag
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          onNodeClick={(_, node) => setSelectedNodeId(node.id)}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="hsl(var(--border))" />
          <Controls showInteractive={false} position="bottom-right" />
        </ReactFlow>
      </div>
      <div className="grid gap-1 rounded-md border border-border/70 bg-muted/30 px-3 py-2 text-[11px] text-muted-foreground">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-foreground">{selectedNodeId}</span>
          {selectedInfo ? <Badge className="h-5 border-border/70 bg-background/70 px-1.5 text-[10px]">{selectedInfo.role}</Badge> : null}
          {selectedInfo ? <span>{selectedInfo.fullServing ? shortName(selectedInfo.fullServing) : selectedInfo.servingLabel}</span> : null}
        </div>
        <div className="flex flex-wrap gap-3">
          <span>VRAM share: {selectedInfo?.vramSharePct ?? 0}%</span>
          <span>Model usage: {selectedInfo?.modelUsagePct ?? 0}%</span>
          <span>Drag to pan, use control buttons to zoom.</span>
        </div>
      </div>
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
  const ringSpacing = nodeRadius * 8.4 + 62;
  const minChord = nodeRadius * 6.8 + 118;
  const positioned: PositionedTopologyNode[] = [{ ...center, x: 0, y: 0, bucket: 'center' }];
  const all = [
    ...serving.map((n) => ({ ...n, bucket: 'serving' as const })),
    ...workers.map((n) => ({ ...n, bucket: 'worker' as const })),
    ...clients.map((n) => ({ ...n, bucket: 'client' as const })),
  ];

  if (all.length > 0 && all.length <= 12) {
    const radius = 150 + ringSpacing + (all.length > 6 ? 20 : 0);
    for (let i = 0; i < all.length; i += 1) {
      const angle = -Math.PI / 2 + ((2 * Math.PI * i) / all.length);
      const node = all[i];
      positioned.push({
        ...node,
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
      });
    }
    return positioned;
  }

  let ring = 1;
  const groups: Array<{ key: 'serving' | 'worker' | 'client'; nodes: TopologyNode[]; phase: number }> = [
    { key: 'serving', nodes: [...serving], phase: 0 },
    { key: 'worker', nodes: [...workers], phase: Math.PI / 9 },
    { key: 'client', nodes: [...clients], phase: Math.PI / 4 },
  ];

  for (const group of groups) {
    let phase = group.phase;
    let offset = 0;
    while (offset < group.nodes.length) {
      const radius = 110 + ring * ringSpacing;
      const capacity = Math.max(8, Math.floor((2 * Math.PI * radius) / minChord));
      const take = Math.min(capacity, group.nodes.length - offset);
      for (let i = 0; i < take; i += 1) {
        const angle = -Math.PI / 2 + phase + ((2 * Math.PI * i) / take);
        const node = group.nodes[offset + i];
        positioned.push({
          ...node,
          x: Math.cos(angle) * radius,
          y: Math.sin(angle) * radius,
          bucket: group.key,
        });
      }
      offset += take;
      phase += 0.21;
      ring += 1;
    }
  }

  return positioned;
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

function MarkdownMessage({ content }: { content: string }) {
  return (
    <div
      className={cn(
        '[&_a]:underline [&_a]:underline-offset-2',
        '[&_blockquote]:my-2 [&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-3 [&_blockquote]:italic',
        '[&_code]:rounded [&_code]:bg-background/70 [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[0.9em]',
        '[&_h1]:mb-2 [&_h1]:mt-3 [&_h1]:text-base [&_h1]:font-semibold [&_h1:first-child]:mt-0',
        '[&_h2]:mb-2 [&_h2]:mt-3 [&_h2]:text-sm [&_h2]:font-semibold [&_h2:first-child]:mt-0',
        '[&_hr]:my-3 [&_hr]:border-border',
        '[&_li]:my-0.5',
        '[&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5',
        '[&_p]:my-2 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0',
        '[&_pre]:my-2 [&_pre]:overflow-x-auto [&_pre]:rounded-lg [&_pre]:border [&_pre]:border-border/70 [&_pre]:bg-background/80 [&_pre]:p-3',
        '[&_pre_code]:bg-transparent [&_pre_code]:p-0',
        '[&_table]:my-2 [&_table]:w-full [&_table]:border-collapse [&_table]:text-xs',
        '[&_td]:border [&_td]:border-border/60 [&_td]:px-2 [&_td]:py-1',
        '[&_th]:border [&_th]:border-border/60 [&_th]:bg-muted/40 [&_th]:px-2 [&_th]:py-1 [&_th]:text-left',
        '[&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5',
      )}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
        {content}
      </ReactMarkdown>
    </div>
  );
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
          {message.content ? <MarkdownMessage content={message.content} /> : !isUser ? '...' : ''}
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
