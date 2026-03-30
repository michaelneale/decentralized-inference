import { type CSSProperties, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
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
import {
  Bot,
  Braces,
  ChevronDown,
  Check,
  Copy,
  Cpu,
  Gauge,
  Gpu,
  Hash,
  ImagePlus,
  Laptop,
  Loader2,
  MessageSquarePlus,
  Maximize2,
  MemoryStick,
  Minimize2,
  Moon,
  Network,
  Pencil,
  RotateCcw,
  Send,
  Server,
  Square,
  Sparkles,
  Sun,
  Trash2,
  UserPlus,
  User,
  Wifi,
  X,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';

import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from './components/ui/accordion';
import { Alert, AlertDescription, AlertTitle } from './components/ui/alert';
import { Badge } from './components/ui/badge';
import { Button } from './components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  navigationMenuTriggerStyle,
} from './components/ui/navigation-menu';
import { Popover, PopoverContent, PopoverTrigger } from './components/ui/popover';
import { ScrollArea } from './components/ui/scroll-area';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Separator } from './components/ui/separator';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './components/ui/table';
import { Textarea } from './components/ui/textarea';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './components/ui/tooltip';
import { Sheet, SheetContent, SheetHeader, SheetTitle } from './components/ui/sheet';
import { BrandIcon } from './components/brand-icon';
import { MeshLlmWordmark } from './components/mesh-llm-wordmark';
import { cn } from './lib/utils';
import githubBlackLogo from './assets/icons/github-invertocat-black.svg';
import githubWhiteLogo from './assets/icons/github-invertocat-white.svg';

const DOCS_URL = 'https://docs.anarchai.org';
const FLY_DOMAINS = ['mesh-llm-console.fly.dev', 'www.mesh-llm.com', 'www.anarchai.org'];

type MeshModel = {
  name: string;
  status: 'warm' | 'cold' | string;
  node_count: number;
  size_gb: number;
  vision?: boolean;
};

type Peer = {
  id: string;
  role: string;
  models: string[];
  vram_gb: number;
  serving?: string | null;
  serving_models?: string[];
  rtt_ms?: number | null;
  hostname?: string;
  is_soc?: boolean;
  gpus?: { name: string; vram_bytes: number; bandwidth_gbps?: number }[];
};

type StatusPayload = {
  version?: string;
  latest_version?: string | null;
  node_id: string;
  token: string;
  node_status: string;
  is_host: boolean;
  is_client: boolean;
  llama_ready: boolean;
  model_name: string;
  serving_models?: string[];
  api_port: number;
  my_vram_gb: number;
  model_size_gb: number;
  mesh_name?: string | null;
  peers: Peer[];
  mesh_models: MeshModel[];
  inflight_requests: number;
  launch_pi?: string | null;
  launch_goose?: string | null;
  nostr_discovery?: boolean;
  my_hostname?: string;
  my_is_soc?: boolean;
  gpus?: { name: string; vram_bytes: number; bandwidth_gbps?: number }[];
};

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  reasoning?: string;
  model?: string;
  stats?: string;
  error?: boolean;
  /** Base64 data URL for attached image (vision) */
  image?: string;
};

type ChatConversation = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
};

type ChatState = {
  conversations: ChatConversation[];
  activeConversationId: string;
};

type ModelServingStat = {
  nodes: number;
  vramGb: number;
};

type TopSection = 'dashboard' | 'chat';
type AppRoute = {
  section: TopSection;
  chatId: string | null;
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
  hostname?: string;
  isSoc?: boolean;
  gpus?: { name: string; vram_bytes: number; bandwidth_gbps?: number }[];
};

type ThemeMode = 'auto' | 'light' | 'dark';

const THEME_STORAGE_KEY = 'mesh-llm-theme';
const DEFAULT_CHAT_TITLE = 'New chat';
const CHAT_DB_NAME = 'mesh-llm-chat-db';
const CHAT_DB_STORE = 'state';
const CHAT_DB_KEY = 'chat-state';
const CHAT_SAVE_DEBOUNCE_MS = 500;
const CHAT_MAX_CONVERSATIONS = 80;
const CHAT_MAX_MESSAGES_PER_CONVERSATION = 240;
const CHAT_MAX_TEXT_CHARS = 12000;

function sectionFromPathname(pathname: string): TopSection | null {
  if (pathname === '/dashboard' || pathname === '/dashboard/') return 'dashboard';
  if (pathname === '/chat' || pathname === '/chat/' || pathname.startsWith('/chat/')) return 'chat';
  return null;
}

function isMobileViewport(): boolean {
  return typeof window !== 'undefined' && window.innerWidth < 768;
}

function readRouteFromLocation(): AppRoute {
  if (typeof window === 'undefined') return { section: 'dashboard', chatId: null };
  const pathname = window.location.pathname;
  if (pathname === '/dashboard' || pathname === '/dashboard/') return { section: 'dashboard', chatId: null };
  if (pathname === '/chat' || pathname === '/chat/') return { section: 'chat', chatId: null };
  if (pathname.startsWith('/chat/')) {
    const raw = pathname.slice('/chat/'.length);
    const chatId = raw ? decodeURIComponent(raw.split('/')[0]) : null;
    return { section: 'chat', chatId };
  }
  // Default: chat on mobile, dashboard on desktop
  return { section: isMobileViewport() ? 'chat' : 'dashboard', chatId: null };
}

function pathnameForRoute(route: AppRoute): string {
  if (route.section === 'dashboard') return '/dashboard';
  return route.chatId ? `/chat/${encodeURIComponent(route.chatId)}` : '/chat';
}

function pushRoute(route: AppRoute) {
  if (typeof window === 'undefined') return;
  const nextPath = pathnameForRoute(route);
  if (window.location.pathname !== nextPath) window.history.pushState({}, '', nextPath);
}

function replaceRoute(route: AppRoute) {
  if (typeof window === 'undefined') return;
  const nextPath = pathnameForRoute(route);
  if (window.location.pathname !== nextPath) window.history.replaceState({}, '', nextPath);
}

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

function randomId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  return `id-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function deriveConversationTitle(input: string): string {
  const compact = input.replace(/\s+/g, ' ').trim();
  if (!compact) return DEFAULT_CHAT_TITLE;
  return compact.length > 52 ? `${compact.slice(0, 52).trimEnd()}...` : compact;
}

function createConversation(seed?: Partial<Pick<ChatConversation, 'title' | 'messages'>>): ChatConversation {
  const now = Date.now();
  return {
    id: randomId(),
    title: seed?.title || DEFAULT_CHAT_TITLE,
    createdAt: now,
    updatedAt: now,
    messages: seed?.messages || [],
  };
}

function createInitialChatState(): ChatState {
  return { conversations: [], activeConversationId: '' };
}

function clampText(text: string | undefined, maxChars = CHAT_MAX_TEXT_CHARS): string | undefined {
  if (typeof text !== 'string') return undefined;
  if (!text.length) return undefined;
  return text.length > maxChars ? `${text.slice(0, maxChars).trimEnd()}...` : text;
}

function sanitizeMessages(raw: unknown): ChatMessage[] {
  if (!Array.isArray(raw)) return [];
  const sanitized = raw.flatMap((item) => {
    if (!item || typeof item !== 'object') return [];
    const role = (item as { role?: unknown }).role;
    const content = (item as { content?: unknown }).content;
    if ((role !== 'user' && role !== 'assistant') || typeof content !== 'string') return [];
    return [{
      id: typeof (item as { id?: unknown }).id === 'string' ? (item as { id: string }).id : randomId(),
      role,
      content: clampText(content, CHAT_MAX_TEXT_CHARS) ?? '',
      reasoning: clampText(typeof (item as { reasoning?: unknown }).reasoning === 'string' ? (item as { reasoning: string }).reasoning : undefined),
      model: clampText(typeof (item as { model?: unknown }).model === 'string' ? (item as { model: string }).model : undefined, 256),
      stats: clampText(typeof (item as { stats?: unknown }).stats === 'string' ? (item as { stats: string }).stats : undefined, 256),
      error: Boolean((item as { error?: unknown }).error),
    }];
  });
  return sanitized.slice(-CHAT_MAX_MESSAGES_PER_CONVERSATION);
}

function sanitizeChatState(raw: unknown): ChatState {
  const fallback = createInitialChatState();
  if (!raw || typeof raw !== 'object') return fallback;

  const parsed = raw as { conversations?: unknown; activeConversationId?: unknown };
  if (!Array.isArray(parsed.conversations)) return fallback;

  const now = Date.now();
  const conversations = parsed.conversations
    .flatMap((item) => {
      if (!item || typeof item !== 'object') return [];
      const obj = item as Record<string, unknown>;
      const id = typeof obj.id === 'string' && obj.id ? obj.id : randomId();
      const titleRaw = typeof obj.title === 'string' && obj.title.trim() ? obj.title : DEFAULT_CHAT_TITLE;
      return [{
        id,
        title: clampText(titleRaw, 140) ?? DEFAULT_CHAT_TITLE,
        createdAt: typeof obj.createdAt === 'number' ? obj.createdAt : now,
        updatedAt: typeof obj.updatedAt === 'number' ? obj.updatedAt : now,
        messages: sanitizeMessages(obj.messages),
      }];
    })
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, CHAT_MAX_CONVERSATIONS);

  return {
    conversations,
    // On reopen, always start at the most recently updated chat.
    activeConversationId: conversations[0]?.id ?? '',
  };
}

function openChatDb(): Promise<IDBDatabase | null> {
  if (typeof window === 'undefined' || !('indexedDB' in window)) return Promise.resolve(null);
  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open(CHAT_DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(CHAT_DB_STORE)) db.createObjectStore(CHAT_DB_STORE, { keyPath: 'id' });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Failed to open chat DB'));
  });
}

function requestToPromise<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('IndexedDB request failed'));
  });
}

function transactionToPromise(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB transaction failed'));
    tx.onabort = () => reject(tx.error ?? new Error('IndexedDB transaction aborted'));
  });
}

async function readChatStateFromDb(): Promise<ChatState | null> {
  let db: IDBDatabase | null = null;
  try {
    db = await openChatDb();
    if (!db) return null;
    const tx = db.transaction(CHAT_DB_STORE, 'readonly');
    const store = tx.objectStore(CHAT_DB_STORE);
    const record = await requestToPromise(store.get(CHAT_DB_KEY) as IDBRequest<{ id: string; state?: unknown } | undefined>);
    await transactionToPromise(tx);
    if (!record?.state) return null;
    return sanitizeChatState(record.state);
  } catch (err) {
    console.warn('Failed to read chat state from IndexedDB:', err);
    return null;
  } finally {
    db?.close();
  }
}

async function writeChatStateToDb(state: ChatState): Promise<void> {
  let db: IDBDatabase | null = null;
  try {
    db = await openChatDb();
    if (!db) return;
    const tx = db.transaction(CHAT_DB_STORE, 'readwrite');
    const store = tx.objectStore(CHAT_DB_STORE);
    store.put({ id: CHAT_DB_KEY, state, updatedAt: Date.now() });
    await transactionToPromise(tx);
  } catch (err) {
    console.warn('Failed to write chat state to IndexedDB:', err);
  } finally {
    db?.close();
  }
}

async function loadPersistedChatState(): Promise<ChatState> {
  const fromDb = await readChatStateFromDb();
  return fromDb ?? createInitialChatState();
}

function findLastUserMessageIndex(messages: ChatMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === 'user') return i;
  }
  return -1;
}

function updateConversationList(
  conversations: ChatConversation[],
  conversationId: string,
  updater: (conversation: ChatConversation) => ChatConversation,
): ChatConversation[] {
  const index = conversations.findIndex((conversation) => conversation.id === conversationId);
  if (index < 0) return conversations;
  const updated = updater(conversations[index]);
  if (index === 0) return [updated, ...conversations.slice(1)];
  return [updated, ...conversations.slice(0, index), ...conversations.slice(index + 1)];
}

export function App() {
  const [section, setSection] = useState<TopSection>(() => readRouteFromLocation().section);
  const [routedChatId, setRoutedChatId] = useState<string | null>(() => readRouteFromLocation().chatId);
  const [themeMode, setThemeMode] = useState<ThemeMode>(() => readThemeMode());
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [chatState, setChatState] = useState<ChatState>(() => createInitialChatState());
  const [chatStateHydrated, setChatStateHydrated] = useState(false);
  const [input, setInput] = useState('');
  const [pendingImage, setPendingImage] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [reasoningOpen, setReasoningOpen] = useState<Record<string, boolean>>({});
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const currentAbortRef = useRef<AbortController | null>(null);
  const activeConversationId = chatState.activeConversationId;
  const conversations = chatState.conversations;
  const activeConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === activeConversationId) ?? conversations[0],
    [conversations, activeConversationId],
  );
  const messages = activeConversation?.messages ?? [];

  const warmModels = useMemo(() => {
    const list = (status?.mesh_models ?? []).filter((m) => m.status === 'warm').map((m) => m.name);
    if (!list.length && status?.model_name) list.push(status.model_name);
    return list;
  }, [status]);
  const modelStatsByName = useMemo<Record<string, ModelServingStat>>(() => {
    const stats: Record<string, ModelServingStat> = {};
    for (const model of warmModels) stats[model] = { nodes: 0, vramGb: 0 };
    if (!status) return stats;

    const addServingNode = (modelName: string, vramGb: number) => {
      if (!stats[modelName]) stats[modelName] = { nodes: 0, vramGb: 0 };
      stats[modelName].nodes += 1;
      stats[modelName].vramGb += Math.max(0, vramGb || 0);
    };

    if (!status.is_client && status.model_name) addServingNode(status.model_name, status.my_vram_gb);
    for (const peer of status.peers ?? []) {
      if (peer.role === 'Client' || !peer.serving || peer.serving === '(idle)') continue;
      addServingNode(peer.serving, peer.vram_gb);
    }

    for (const model of status.mesh_models ?? []) {
      if (!stats[model.name]) continue;
      if (stats[model.name].nodes === 0) stats[model.name].nodes = Math.max(0, model.node_count || 0);
    }

    return stats;
  }, [status, warmModels]);
  const selectedChatModel = selectedModel || warmModels[0] || status?.model_name || '';
  const visionModels = useMemo(() => {
    const set = new Set<string>();
    for (const m of status?.mesh_models ?? []) {
      if (m.vision) set.add(m.name);
    }
    return set;
  }, [status?.mesh_models]);
  const selectedModelVision = useMemo(() => {
    if (selectedModel) return visionModels.has(selectedModel);
    return (status?.mesh_models ?? []).some((m) => m.status === 'warm' && m.vision);
  }, [status?.mesh_models, selectedModel, visionModels]);
  const selectedModelStat = selectedChatModel ? modelStatsByName[selectedChatModel] : undefined;
  const selectedModelNodeCount = selectedModelStat ? selectedModelStat.nodes : null;
  const selectedModelVramGb = selectedModelStat ? selectedModelStat.vramGb : null;

  const inviteWithModelName = selectedModel || warmModels[0] || status?.model_name || '';
  const inviteWithModelCommand = useMemo(() => {
    const token = status?.token ?? '';
    return token && inviteWithModelName ? `mesh-llm --join ${token} --model ${inviteWithModelName}` : '';
  }, [inviteWithModelName, status?.token]);
  const inviteToken = status?.token ?? '';
  const inviteClientCommand = useMemo(() => {
    const token = status?.token ?? '';
    return token ? `mesh-llm --client --join ${token}` : '';
  }, [status?.token]);
  const isLocalhost = typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
  const isFlyHosted = typeof window !== 'undefined' && FLY_DOMAINS.includes(window.location.hostname);
  const apiDirectUrl = useMemo(() => {
    if (!isLocalhost) return '';
    const port = status?.api_port ?? 9337;
    return `http://127.0.0.1:${port}/v1`;
  }, [status?.api_port, isLocalhost]);

  useEffect(() => {
    if (!warmModels.length) return;
    if (!selectedModel || (selectedModel !== 'auto' && !warmModels.includes(selectedModel))) setSelectedModel(warmModels.length > 1 ? 'auto' : warmModels[0]);
  }, [warmModels, selectedModel]);

  useEffect(() => {
    applyThemeMode(themeMode);
    window.localStorage.setItem(THEME_STORAGE_KEY, themeMode);
  }, [themeMode]);

  useEffect(() => {
    let cancelled = false;
    void loadPersistedChatState()
      .then((loaded) => {
        if (cancelled) return;
        setChatState(sanitizeChatState(loaded));
      })
      .catch((err) => {
        console.warn('Failed to hydrate chat state:', err);
      })
      .finally(() => {
        if (!cancelled) setChatStateHydrated(true);
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!chatStateHydrated) return;
    const timeout = window.setTimeout(() => {
      void writeChatStateToDb(sanitizeChatState(chatState));
    }, CHAT_SAVE_DEBOUNCE_MS);
    return () => window.clearTimeout(timeout);
  }, [chatState, chatStateHydrated]);

  useEffect(() => {
    if (themeMode !== 'auto') return;
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = () => applyThemeMode('auto');
    media.addEventListener('change', onChange);
    return () => media.removeEventListener('change', onChange);
  }, [themeMode]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const current = sectionFromPathname(window.location.pathname);
    if (current == null) replaceRoute({ section: 'dashboard', chatId: null });

    const onPopState = () => {
      const route = readRouteFromLocation();
      setSection(route.section);
      setRoutedChatId(route.chatId);
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  useEffect(() => {
    if (section !== 'chat' || !chatStateHydrated) return;
    if (routedChatId) {
      if (conversations.some((conversation) => conversation.id === routedChatId)) {
        setChatState((prev) => (prev.activeConversationId === routedChatId ? prev : { ...prev, activeConversationId: routedChatId }));
      } else {
        const fallbackId = conversations[0]?.id ?? null;
        setRoutedChatId(fallbackId);
        replaceRoute({ section: 'chat', chatId: fallbackId });
        setChatState((prev) => ({ ...prev, activeConversationId: fallbackId ?? '' }));
      }
      return;
    }
    if (activeConversation?.id) {
      setRoutedChatId(activeConversation.id);
      replaceRoute({ section: 'chat', chatId: activeConversation.id });
    }
  }, [section, chatStateHydrated, routedChatId, conversations, activeConversation?.id]);

  useEffect(() => {
    let stop = false;
    let statusEvents: EventSource | null = null;
    let reconnectTimer: number | null = null;
    let retryMs = 1000;
    const MAX_RETRY_MS = 15000;
    const reconnectStatusMessage = 'Trying to reconnect automatically. Live updates will resume shortly.';

    const clearReconnectTimer = () => {
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const closeStatusEvents = () => {
      if (!statusEvents) return;
      statusEvents.onopen = null;
      statusEvents.onmessage = null;
      statusEvents.onerror = null;
      statusEvents.close();
      statusEvents = null;
    };

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
          if (!stop) {
            setStatusError(reconnectStatusMessage);
            console.warn('Failed to fetch /api/status:', err.message);
          }
        });
    };

    const connectStatusEvents = () => {
      if (stop || statusEvents) return;

      let source: EventSource;
      try {
        source = new EventSource('/api/events');
      } catch (err) {
        const message = err instanceof Error ? err.message : 'failed to create EventSource';
        console.warn('Failed to connect status stream:', message);
        scheduleReconnect();
        return;
      }

      statusEvents = source;
      source.onopen = () => {
        if (stop) return;
        retryMs = 1000;
        setStatusError(null);
        loadStatus();
      };
      source.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as StatusPayload;
          setStatus(payload);
          setStatusError(null);
        } catch {
          // ignore malformed status event
        }
      };
      source.onerror = () => {
        if (stop) return;
        scheduleReconnect();
      };
    };

    const scheduleReconnect = () => {
      if (stop || reconnectTimer !== null) return;
      setStatusError(reconnectStatusMessage);
      console.warn('Connection lost. Reconnecting...');
      closeStatusEvents();
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;
        connectStatusEvents();
        retryMs = Math.min(retryMs * 2, MAX_RETRY_MS);
      }, retryMs);
    };

    loadStatus();
    connectStatusEvents();

    return () => {
      stop = true;
      clearReconnectTimer();
      closeStatusEvents();
    };
  }, []);

  useEffect(() => {
    const el = chatScrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, isSending, activeConversationId]);

  useEffect(() => () => currentAbortRef.current?.abort(), []);

  const canChat = !!status && (status.llama_ready || (status.is_client && warmModels.length > 0));
  const canRegenerate = canChat && !!activeConversation && findLastUserMessageIndex(activeConversation.messages) >= 0;

  function updateChatState(updater: (prev: ChatState) => ChatState) {
    setChatState((prev) => updater(prev));
  }

  async function streamAssistantReply(params: {
    conversationId: string;
    assistantId: string;
    model: string;
    historyForRequest: ChatMessage[];
  }) {
    const { conversationId, assistantId, model, historyForRequest } = params;
    const reqStart = performance.now();
    const controller = new AbortController();
    currentAbortRef.current = controller;

    try {
      const MAX_RETRIES = 3;
      const RETRY_DELAYS = [1000, 2000, 4000];
      const RETRYABLE = new Set([500, 502, 503]);
      let response: Response | null = null;

      for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
        response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: controller.signal,
          body: JSON.stringify({
            model,
            messages: historyForRequest.map((m) => ({
              role: m.role,
              content: m.image
                ? [
                    { type: 'text' as const, text: m.content },
                    { type: 'image_url' as const, image_url: { url: m.image } },
                  ]
                : m.content,
            })),
            stream: true,
            stream_options: { include_usage: true },
            chat_template_kwargs: { enable_thinking: false },
          }),
        });
        if (response.ok && response.body) break;
        if (!RETRYABLE.has(response.status) || attempt === MAX_RETRIES - 1) break;
        await new Promise((r) => setTimeout(r, RETRY_DELAYS[attempt]));
      }

      if (!response?.ok || !response?.body) throw new Error(`HTTP ${response?.status ?? 'unknown'}`);

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
            updateChatState((prev) => ({
              ...prev,
              conversations: updateConversationList(prev.conversations, conversationId, (conversation) => ({
                ...conversation,
                messages: conversation.messages.map((m) => (m.id === assistantId ? { ...m, content: full, reasoning: reasoning || undefined } : m)),
                updatedAt: Date.now(),
              })),
            }));
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

      updateChatState((prev) => ({
        ...prev,
        conversations: updateConversationList(prev.conversations, conversationId, (conversation) => ({
          ...conversation,
          messages: conversation.messages.map((m) =>
            m.id === assistantId
              ? { ...m, content: m.content || '(empty response)', reasoning: m.reasoning || undefined, stats }
              : m,
          ),
          updatedAt: Date.now(),
        })),
      }));
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        updateChatState((prev) => ({
          ...prev,
          conversations: updateConversationList(prev.conversations, conversationId, (conversation) => ({
            ...conversation,
            messages: conversation.messages.map((m) => (
              m.id === assistantId ? { ...m, content: m.content || '(stopped)' } : m
            )),
            updatedAt: Date.now(),
          })),
        }));
      } else {
        const message = err instanceof Error ? err.message : String(err);
        updateChatState((prev) => ({
          ...prev,
          conversations: updateConversationList(prev.conversations, conversationId, (conversation) => ({
            ...conversation,
            messages: conversation.messages.map((m) => (m.id === assistantId ? { ...m, content: `Error: ${message}`, error: true } : m)),
            updatedAt: Date.now(),
          })),
        }));
      }
    } finally {
      if (currentAbortRef.current === controller) currentAbortRef.current = null;
      setIsSending(false);
    }
  }

  async function sendMessage(text: string) {
    const trimmed = text.trim();
    if ((!trimmed && !pendingImage) || !status || isSending) return;

    // When sending an image with model=auto, route to a vision-capable model
    // (the server-side router doesn't sniff for images in the request body)
    let model = selectedModel || status.model_name;
    if (pendingImage && (!model || model === 'auto')) {
      const visionModel = warmModels.find((m) => visionModels.has(m));
      if (visionModel) model = visionModel;
    }
    const conversationId = activeConversation?.id ?? randomId();
    const userMessage: ChatMessage = { id: randomId(), role: 'user', content: trimmed, model, image: pendingImage ?? undefined };
    const assistantId = randomId();
    const assistantMessage: ChatMessage = { id: assistantId, role: 'assistant', content: '', model };
    const existingMessages = activeConversation?.messages ?? [];
    const historyForRequest = [...existingMessages, userMessage];
    const nextTitle = (activeConversation?.title === DEFAULT_CHAT_TITLE && existingMessages.length === 0) || !activeConversation
      ? deriveConversationTitle(trimmed)
      : activeConversation.title;

    updateChatState((prev) => ({
      ...(prev.conversations.some((conversation) => conversation.id === conversationId)
        ? {
            ...prev,
            conversations: updateConversationList(prev.conversations, conversationId, (conversation) => ({
              ...conversation,
              title: nextTitle,
              updatedAt: Date.now(),
              messages: [...historyForRequest, assistantMessage],
            })),
            activeConversationId: conversationId,
          }
        : {
            conversations: [{
              id: conversationId,
              title: nextTitle,
              createdAt: Date.now(),
              updatedAt: Date.now(),
              messages: [...historyForRequest, assistantMessage],
            }, ...prev.conversations],
            activeConversationId: conversationId,
          }),
    }));
    setSection('chat');
    setRoutedChatId(conversationId);
    pushRoute({ section: 'chat', chatId: conversationId });
    setInput('');
    setPendingImage(null);
    setIsSending(true);
    await streamAssistantReply({ conversationId, assistantId, model, historyForRequest });
  }

  async function regenerateLastResponse() {
    if (!status || isSending || !activeConversation) return;
    const model = selectedModel || status.model_name;
    const conversationId = activeConversation.id;
    const lastUserIndex = findLastUserMessageIndex(activeConversation.messages);
    if (lastUserIndex < 0) return;
    const historyForRequest = activeConversation.messages.slice(0, lastUserIndex + 1);
    const assistantId = randomId();
    const assistantMessage: ChatMessage = { id: assistantId, role: 'assistant', content: '', model };

    updateChatState((prev) => ({
      ...prev,
      conversations: updateConversationList(prev.conversations, conversationId, (conversation) => ({
        ...conversation,
        updatedAt: Date.now(),
        messages: [...historyForRequest, assistantMessage],
      })),
    }));
    setIsSending(true);
    await streamAssistantReply({ conversationId, assistantId, model, historyForRequest });
  }

  function stopStreaming() {
    currentAbortRef.current?.abort();
  }

  function createNewConversation() {
    const conversation = createConversation();
    updateChatState((prev) => ({
      conversations: [conversation, ...prev.conversations],
      activeConversationId: conversation.id,
    }));
    setSection('chat');
    setRoutedChatId(conversation.id);
    pushRoute({ section: 'chat', chatId: conversation.id });
    setReasoningOpen({});
    setInput('');
  }

  function selectConversation(conversationId: string) {
    updateChatState((prev) => {
      if (prev.activeConversationId === conversationId) return prev;
      if (!prev.conversations.some((conversation) => conversation.id === conversationId)) return prev;
      return { ...prev, activeConversationId: conversationId };
    });
    setSection('chat');
    setRoutedChatId(conversationId);
    pushRoute({ section: 'chat', chatId: conversationId });
    setReasoningOpen({});
    setInput('');
  }

  function renameConversation(conversationId: string, nextTitle: string) {
    const title = clampText(nextTitle.trim(), 140) || DEFAULT_CHAT_TITLE;
    updateChatState((prev) => ({
      ...prev,
      conversations: updateConversationList(prev.conversations, conversationId, (conversation) => ({
        ...conversation,
        title,
        updatedAt: Date.now(),
      })),
    }));
  }

  function deleteConversation(conversationId: string) {
    const remaining = conversations.filter((conversation) => conversation.id !== conversationId);
    const nextActiveId = activeConversationId === conversationId ? (remaining[0]?.id ?? null) : (activeConversationId || null);
    updateChatState((prev) => {
      return {
        conversations: prev.conversations.filter((conversation) => conversation.id !== conversationId),
        activeConversationId: prev.activeConversationId === conversationId ? (nextActiveId ?? '') : prev.activeConversationId,
      };
    });
    setSection('chat');
    setRoutedChatId(nextActiveId);
    replaceRoute({ section: 'chat', chatId: nextActiveId });
    setReasoningOpen({});
    setInput('');
  }

  function clearAllConversations() {
    updateChatState(() => ({ conversations: [], activeConversationId: '' }));
    setSection('chat');
    setRoutedChatId(null);
    replaceRoute({ section: 'chat', chatId: null });
    setReasoningOpen({});
    setInput('');
  }

  const peerStatusLabel = (peer: Peer): string => {
    if (peer.role === 'Client') return 'Client';
    if (peer.serving && peer.serving !== '(idle)') return 'Serving';
    if (peer.role === 'Host') return 'Host';
    return 'Idle';
  };

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
        servingModels: (status.serving_models && status.serving_models.length > 0)
          ? status.serving_models
          : (status.model_name ? [status.model_name] : []),
        statusLabel: status.node_status || (status.is_client ? 'Client' : status.is_host ? 'Host' : 'Idle'),
        latencyMs: null,
        hostname: status.my_hostname,
        isSoc: status.my_is_soc,
        gpus: status.gpus,
      });
    }
    for (const p of status.peers ?? []) {
      const pModels = (p.serving_models && p.serving_models.length > 0) ? p.serving_models : (p.serving ? [p.serving] : []);
      nodes.push({
        id: p.id,
        vram: p.vram_gb,
        self: false,
        host: /^Host/.test(p.role),
        client: p.role === 'Client',
        serving: p.serving || '',
        servingModels: pModels,
        statusLabel: peerStatusLabel(p),
        latencyMs: p.rtt_ms ?? null,
        hostname: p.hostname,
        isSoc: p.is_soc,
        gpus: p.gpus,
      });
    }
    return nodes;
  }, [status]);

  const sections: Array<{ key: TopSection; label: string }> = [
    { key: 'dashboard', label: 'Network' },
    { key: 'chat', label: 'Chat' },
  ];

  function navigateToSection(next: TopSection) {
    if (next === section) return;
    const nextChatId = next === 'chat' ? (activeConversation?.id ?? null) : null;
    pushRoute({ section: next, chatId: nextChatId });
    setSection(next);
    setRoutedChatId(nextChatId);
  }

  function handleSubmit() {
    if (!canChat) return;
    void sendMessage(input);
  }

  return (
    <div className="h-screen overflow-hidden bg-background [height:100svh] [padding-top:env(safe-area-inset-top)] [padding-bottom:env(safe-area-inset-bottom)]">
      <div className="flex h-full min-h-0 flex-col">
        <AppHeader
          sections={sections}
          section={section}
          setSection={navigateToSection}
          themeMode={themeMode}
          setThemeMode={setThemeMode}
          statusError={statusError}
          inviteWithModelCommand={inviteWithModelCommand}
          inviteWithModelName={inviteWithModelName}
          inviteClientCommand={inviteClientCommand}
          inviteToken={inviteToken}
          apiDirectUrl={apiDirectUrl}
          isPublicMesh={status?.nostr_discovery ?? false}
        />

        <main className="flex min-h-0 flex-1 flex-col overflow-hidden">
          {section === 'chat' ? (
            <div className="mx-auto flex min-h-0 min-w-0 w-full max-w-7xl flex-1 flex-col overflow-hidden p-2 md:p-4">
              <ChatPage
                inviteToken={status?.token ?? ''}
                isPublicMesh={status?.nostr_discovery ?? false}
                isFlyHosted={isFlyHosted}
                inflightRequests={status?.inflight_requests ?? 0}
                warmModels={warmModels}
                modelStatsByName={modelStatsByName}
                selectedModel={selectedModel}
                setSelectedModel={setSelectedModel}
                selectedModelNodeCount={selectedModelNodeCount}
                selectedModelVramGb={selectedModelVramGb}
                selectedModelVision={selectedModelVision}
                visionModels={visionModels}
                pendingImage={pendingImage}
                setPendingImage={setPendingImage}
                conversations={conversations}
                activeConversationId={activeConversationId}
                onConversationCreate={createNewConversation}
                onConversationSelect={selectConversation}
                onConversationRename={renameConversation}
                onConversationDelete={deleteConversation}
                onConversationsClear={clearAllConversations}
                messages={messages}
                reasoningOpen={reasoningOpen}
                setReasoningOpen={setReasoningOpen}
                chatScrollRef={chatScrollRef}
                input={input}
                setInput={setInput}
                isSending={isSending}
                canChat={canChat}
                canRegenerate={canRegenerate}
                onStop={stopStreaming}
                onRegenerate={regenerateLastResponse}
                onSubmit={handleSubmit}
              />
            </div>
          ) : null}

          {section === 'dashboard' ? (
            <div className="min-h-0 flex-1 overflow-y-auto">
              <div className="mx-auto w-full max-w-7xl p-4">
                <DashboardPage
                  status={status}
                  topologyNodes={topologyNodes}
                  selectedModel={selectedModel || status?.model_name || ''}
                  themeMode={themeMode}
                  isPublicMesh={status?.nostr_discovery ?? false}
                  inviteToken={inviteToken}
                  isLocalhost={isLocalhost}
                />
              </div>
            </div>
          ) : null}
        </main>
        <footer className={cn("shrink-0 bg-card/70", section === 'chat' ? 'hidden md:block' : '')}>
          <div className="mx-auto flex h-8 w-full max-w-7xl items-center justify-center gap-2 px-4 text-xs text-muted-foreground">
            Mesh LLM {status?.version ? `v${status.version}` : 'version loading...'}
            {status?.latest_version ? (
              <>
                <span>·</span>
                <a
                  href="https://github.com/michaelneale/mesh-llm/releases"
                  target="_blank"
                  rel="noreferrer"
                  className="underline-offset-2 hover:text-foreground hover:underline"
                  title="A newer mesh-llm version is available"
                >
                  {status?.version
                    ? `Update available: v${status.version} -> v${status.latest_version}`
                    : `Update available: v${status.latest_version}`}
                </a>
              </>
            ) : null}
            <span>·</span>
            <a
              href="https://github.com/michaelneale/mesh-llm"
              target="_blank"
              rel="noreferrer"
              className="inline-flex h-5 w-5 items-center justify-center hover:text-foreground"
              aria-label="GitHub repository"
              title="GitHub repository"
            >
              <span className="relative h-4 w-4">
                <img src={githubBlackLogo} alt="" aria-hidden="true" className="h-4 w-4 dark:hidden" />
                <img src={githubWhiteLogo} alt="" aria-hidden="true" className="hidden h-4 w-4 dark:block" />
              </span>
            </a>
          </div>
        </footer>
      </div>
    </div>
  );
}

function AppHeader({
  sections,
  section,
  setSection,
  themeMode,
  setThemeMode,
  statusError,
  inviteWithModelCommand,
  inviteWithModelName,
  inviteClientCommand,
  inviteToken,
  apiDirectUrl,
  isPublicMesh,
}: {
  sections: Array<{ key: TopSection; label: string }>;
  section: TopSection;
  setSection: (section: TopSection) => void;
  themeMode: ThemeMode;
  setThemeMode: React.Dispatch<React.SetStateAction<ThemeMode>>;
  statusError: string | null;
  inviteWithModelCommand: string;
  inviteWithModelName: string;
  inviteClientCommand: string;
  inviteToken: string;
  apiDirectUrl: string;
  isPublicMesh: boolean;
}) {
  const [inviteWithModelCopied, setInviteWithModelCopied] = useState(false);
  const [inviteClientCopied, setInviteClientCopied] = useState(false);
  const [tokenCopied, setTokenCopied] = useState(false);
  const [apiDirectCopied, setApiDirectCopied] = useState(false);
  const [isThemePopoverOpen, setIsThemePopoverOpen] = useState(false);

  async function copyInviteWithModelCommand() {
    if (!inviteWithModelCommand) return;
    try {
      await navigator.clipboard.writeText(inviteWithModelCommand);
      setInviteWithModelCopied(true);
      window.setTimeout(() => setInviteWithModelCopied(false), 1500);
    } catch {
      setInviteWithModelCopied(false);
    }
  }

  async function copyInviteClientCommand() {
    if (!inviteClientCommand) return;
    try {
      await navigator.clipboard.writeText(inviteClientCommand);
      setInviteClientCopied(true);
      window.setTimeout(() => setInviteClientCopied(false), 1500);
    } catch {
      setInviteClientCopied(false);
    }
  }

  async function copyInviteToken() {
    if (!inviteToken) return;
    try {
      await navigator.clipboard.writeText(inviteToken);
      setTokenCopied(true);
      window.setTimeout(() => setTokenCopied(false), 1500);
    } catch {
      setTokenCopied(false);
    }
  }

  async function copyApiDirectUrl() {
    if (!apiDirectUrl) return;
    try {
      await navigator.clipboard.writeText(apiDirectUrl);
      setApiDirectCopied(true);
      window.setTimeout(() => setApiDirectCopied(false), 1500);
    } catch {
      setApiDirectCopied(false);
    }
  }

  function selectThemeMode(mode: ThemeMode) {
    setThemeMode(mode);
    setIsThemePopoverOpen(false);
  }

  return (
    <header className="shrink-0 border-b bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/80">
      <div className="mx-auto flex h-14 w-full max-w-7xl items-center gap-2 px-3 md:h-16 md:gap-4 md:px-4">
        <div className="flex min-w-0 items-center gap-0">
          <div className="flex h-10 w-7 shrink-0 items-center justify-start">
            <BrandIcon className="h-6 w-6 text-primary" />
          </div>
          <div className="hidden min-w-0 sm:block">
            <div className="truncate text-base font-semibold">
              <MeshLlmWordmark />
            </div>
          </div>
        </div>
        <NavigationMenu>
          <NavigationMenuList>
            {sections.map(({ key, label }) => (
              <NavigationMenuItem key={key}>
                <NavigationMenuLink asChild>
                  <a
                    href={key === 'chat' ? '/chat' : '/dashboard'}
                    onClick={(event) => {
                      const isPlainLeftClick = event.button === 0 && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey;
                      if (!isPlainLeftClick) return;
                      event.preventDefault();
                      setSection(key);
                    }}
                    className={navigationMenuTriggerStyle()}
                    data-active={section === key ? '' : undefined}
                    aria-current={section === key ? 'page' : undefined}
                  >
                    {label}
                  </a>
                </NavigationMenuLink>
              </NavigationMenuItem>
            ))}
          </NavigationMenuList>
        </NavigationMenu>
        <TooltipProvider>
          <div className="ml-auto flex items-center gap-2">
            <Popover>
              <Tooltip>
                <TooltipTrigger asChild>
                  <PopoverTrigger asChild>
                    <Button type="button" variant="outline" size="icon" aria-label="API access">
                      <Braces className="h-4 w-4" />
                    </Button>
                  </PopoverTrigger>
                </TooltipTrigger>
                <TooltipContent>API</TooltipContent>
              </Tooltip>
              <PopoverContent className="w-[calc(100vw-2rem)] max-w-[420px] space-y-3" align="end">
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <Braces className="h-4 w-4 text-muted-foreground" />
                  <span>API Access</span>
                </div>
                <div className="text-xs text-muted-foreground">
                  OpenAI-compatible endpoint — works with any app that speaks the OpenAI API.
                </div>
              </div>
              {apiDirectUrl ? (
                <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
                  <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
                    {apiDirectUrl}
                  </code>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7 shrink-0"
                    aria-label="Copy endpoint URL"
                    onClick={() => void copyApiDirectUrl()}
                  >
                    {apiDirectCopied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                  </Button>
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="text-xs text-muted-foreground">
                    Run mesh-llm locally to get an OpenAI-compatible API on your machine:
                  </div>
                  <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
                    <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
                      {isPublicMesh ? 'mesh-llm --auto' : `mesh-llm --auto --join ${inviteToken || '(token)'}`}
                    </code>
                    <Button
                      type="button"
                      size="icon"
                      variant="ghost"
                      className="h-6 w-6 shrink-0"
                      aria-label="Copy command"
                      onClick={() => void navigator.clipboard.writeText(isPublicMesh ? 'mesh-llm --auto' : `mesh-llm --auto --join ${inviteToken || ''}`)}
                    >
                      <Copy className="h-3 w-3" />
                    </Button>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    This gives you <code className="text-[0.7rem]">http://127.0.0.1:9337/v1</code> locally — point any OpenAI-compatible app at it.
                  </div>
                </div>
              )}
              <div className="space-y-2 pt-1">
                <div className="text-xs font-medium">Use with agents</div>
                <div className="space-y-1">
                  {['claude', 'goose'].map((agent) => {
                    const cmd = isPublicMesh ? `mesh-llm ${agent}` : `mesh-llm ${agent} --join ${inviteToken || '(token)'}`;
                    return (
                      <div key={agent} className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
                        <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
                          {cmd}
                        </code>
                        <Button
                          type="button"
                          size="icon"
                          variant="ghost"
                          className="h-6 w-6 shrink-0"
                          aria-label={`Copy ${agent} command`}
                          onClick={() => void navigator.clipboard.writeText(cmd)}
                        >
                          <Copy className="h-3 w-3" />
                        </Button>
                      </div>
                    );
                  })}
                </div>
                <div className="text-xs text-muted-foreground">
                  Also works with pi and any OpenAI-compatible client.{' '}
                  <a href={`${DOCS_URL}/#agents`} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
                    Setup guide →
                  </a>
                </div>
              </div>
              <div className="text-xs text-muted-foreground pt-1">
                Don't have it yet?{' '}
                <a href={`${DOCS_URL}/#install`} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
                  Install mesh-llm →
                </a>
              </div>
              <div className="text-xs text-muted-foreground pt-1">
                Agents can gossip too — <code className="text-[0.9em]">mesh-llm blackboard install-skill</code>{' '}
                <a href={`${DOCS_URL}/#blackboard`} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
                  →
                </a>
              </div>
              </PopoverContent>
            </Popover>
            <Popover>
              <Tooltip>
                <TooltipTrigger asChild>
                  <PopoverTrigger asChild>
                    <Button
                      type="button"
                      variant="outline"
                      size="icon"
                      aria-label="Invite"
                      disabled={!inviteToken}
                    >
                      <UserPlus className="h-4 w-4" />
                    </Button>
                  </PopoverTrigger>
                </TooltipTrigger>
                <TooltipContent>Invite</TooltipContent>
              </Tooltip>
              <PopoverContent className="w-[calc(100vw-2rem)] max-w-[420px] space-y-3" align="end">
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <UserPlus className="h-4 w-4 text-muted-foreground" />
                  <span>Invite to this mesh</span>
                </div>
                <div className="text-xs text-muted-foreground">
                  Invite with a model loaded to add compute, or invite as a client for API-only access.
                </div>
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-xs font-medium">
                  <span>Contribute compute</span>
                  <Badge className="h-5 gap-1 border-emerald-500/40 bg-emerald-500/10 px-2 text-[10px] text-emerald-700 dark:text-emerald-300">
                    <Sparkles className="h-3 w-3" />
                    Recommended
                  </Badge>
                </div>
                <div className="text-xs text-muted-foreground">
                  Joins and serves the model {inviteWithModelName || 'selected model'}
                </div>
              </div>
              {inviteWithModelCommand ? (
                <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
                  <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
                    {inviteWithModelCommand}
                  </code>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7 shrink-0"
                    aria-label="Copy model command"
                    onClick={() => void copyInviteWithModelCommand()}
                  >
                    {inviteWithModelCopied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                  </Button>
                </div>
              ) : (
                <div className="text-xs text-muted-foreground">No warm model selected yet.</div>
              )}
              <div className="space-y-1 pt-1">
                <div className="text-xs font-medium">Join as client</div>
                <div className="text-xs text-muted-foreground">Connects for API access without loading a model.</div>
              </div>
              {inviteClientCommand ? (
                <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
                  <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
                    {inviteClientCommand}
                  </code>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7 shrink-0"
                    aria-label="Copy client command"
                    onClick={() => void copyInviteClientCommand()}
                  >
                    {inviteClientCopied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                  </Button>
                </div>
              ) : (
                <div className="text-xs text-muted-foreground">No invite token available yet.</div>
              )}
              <div className="space-y-1 pt-1">
                <div className="text-xs font-medium">Invite token</div>
                {inviteToken ? (
                  <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
                    <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
                      {inviteToken}
                    </code>
                    <Button
                      type="button"
                      size="icon"
                      variant="ghost"
                      className="h-7 w-7 shrink-0"
                      aria-label="Copy invite token"
                      onClick={() => void copyInviteToken()}
                    >
                      {tokenCopied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                    </Button>
                  </div>
                ) : (
                  <div className="text-xs text-muted-foreground">No invite token available yet.</div>
                )}
              </div>
              </PopoverContent>
            </Popover>
            <Popover open={isThemePopoverOpen} onOpenChange={setIsThemePopoverOpen}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <PopoverTrigger asChild>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="h-9 gap-1 px-2"
                      aria-label={`Theme: ${themeMode}`}
                    >
                      {themeMode === 'auto' ? <Laptop className="h-4 w-4" /> : null}
                      {themeMode === 'light' ? <Sun className="h-4 w-4" /> : null}
                      {themeMode === 'dark' ? <Moon className="h-4 w-4" /> : null}
                      <ChevronDown className="h-3 w-3 text-muted-foreground" />
                    </Button>
                  </PopoverTrigger>
                </TooltipTrigger>
                <TooltipContent>Theme</TooltipContent>
              </Tooltip>
              <PopoverContent className="w-40 space-y-1 p-1" align="end">
                <button
                  type="button"
                  className={cn(
                    'flex w-full items-center justify-between rounded-md px-2 py-1.5 text-xs hover:bg-muted',
                    themeMode === 'auto' ? 'bg-muted' : '',
                  )}
                  onClick={() => selectThemeMode('auto')}
                >
                  <span className="flex items-center gap-2">
                    <Laptop className="h-3.5 w-3.5" />
                    Auto
                  </span>
                  {themeMode === 'auto' ? <Check className="h-3.5 w-3.5" /> : null}
                </button>
                <button
                  type="button"
                  className={cn(
                    'flex w-full items-center justify-between rounded-md px-2 py-1.5 text-xs hover:bg-muted',
                    themeMode === 'light' ? 'bg-muted' : '',
                  )}
                  onClick={() => selectThemeMode('light')}
                >
                  <span className="flex items-center gap-2">
                    <Sun className="h-3.5 w-3.5" />
                    Light
                  </span>
                  {themeMode === 'light' ? <Check className="h-3.5 w-3.5" /> : null}
                </button>
                <button
                  type="button"
                  className={cn(
                    'flex w-full items-center justify-between rounded-md px-2 py-1.5 text-xs hover:bg-muted',
                    themeMode === 'dark' ? 'bg-muted' : '',
                  )}
                  onClick={() => selectThemeMode('dark')}
                >
                  <span className="flex items-center gap-2">
                    <Moon className="h-3.5 w-3.5" />
                    Dark
                  </span>
                  {themeMode === 'dark' ? <Check className="h-3.5 w-3.5" /> : null}
                </button>
              </PopoverContent>
            </Popover>
          </div>
        </TooltipProvider>
      </div>
      {statusError ? (
        <div className="mx-auto w-full max-w-7xl px-4 pb-3">
          <Alert variant="destructive">
            <Loader2 className="h-4 w-4 animate-spin" />
            <AlertTitle>Connection Interrupted</AlertTitle>
            <AlertDescription>{statusError}</AlertDescription>
          </Alert>
        </div>
      ) : null}
    </header>
  );
}

function ChatPage(props: {
  inviteToken: string;
  isPublicMesh: boolean;
  isFlyHosted: boolean;
  inflightRequests: number;
  warmModels: string[];
  modelStatsByName: Record<string, ModelServingStat>;
  selectedModel: string;
  setSelectedModel: (v: string) => void;
  selectedModelNodeCount: number | null;
  selectedModelVramGb: number | null;
  selectedModelVision: boolean;
  visionModels: Set<string>;
  pendingImage: string | null;
  setPendingImage: (v: string | null) => void;
  conversations: ChatConversation[];
  activeConversationId: string;
  onConversationCreate: () => void;
  onConversationSelect: (conversationId: string) => void;
  onConversationRename: (conversationId: string, title: string) => void;
  onConversationDelete: (conversationId: string) => void;
  onConversationsClear: () => void;
  messages: ChatMessage[];
  reasoningOpen: Record<string, boolean>;
  setReasoningOpen: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
  chatScrollRef: React.RefObject<HTMLDivElement>;
  input: string;
  setInput: (v: string) => void;
  isSending: boolean;
  canChat: boolean;
  canRegenerate: boolean;
  onStop: () => void;
  onRegenerate: () => void;
  onSubmit: () => void;
}) {
  const {
    inviteToken,
    warmModels,
    modelStatsByName,
    selectedModel,
    setSelectedModel,
    selectedModelNodeCount,
    selectedModelVramGb,
    selectedModelVision,
    visionModels,
    pendingImage,
    setPendingImage,
    conversations,
    activeConversationId,
    onConversationCreate,
    onConversationSelect,
    onConversationRename,
    onConversationDelete,
    onConversationsClear,
    messages,
    reasoningOpen,
    setReasoningOpen,
    chatScrollRef,
    input,
    setInput,
    isSending,
    canChat,
    canRegenerate,
    onStop,
    onRegenerate,
    onSubmit,
  } = props;

  const hasChats = conversations.length > 0;
  const selectedModelValue = selectedModel || warmModels[0] || '';
  const [editingConversationId, setEditingConversationId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const chatInputRef = useRef<HTMLTextAreaElement | null>(null);
  const imageInputRef = useRef<HTMLInputElement | null>(null);

  function handleImageSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    // Read as data URL first (handles HEIC, JPEG, PNG — whatever the browser supports)
    // then resize via canvas to max 512px (vision encoders tile at ~448px internally)
    const reader = new FileReader();
    reader.onload = () => {
      const src = reader.result as string;
      const img = new Image();
      img.onload = () => {
        const MAX = 512;
        let { width, height } = img;
        if (width > MAX || height > MAX) {
          const scale = MAX / Math.max(width, height);
          width = Math.round(width * scale);
          height = Math.round(height * scale);
        }
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (!ctx) { setPendingImage(src); return; }
        ctx.drawImage(img, 0, 0, width, height);
        setPendingImage(canvas.toDataURL('image/jpeg', 0.85));
      };
      img.onerror = () => {
        // Canvas resize failed — send original (may be large but better than nothing)
        setPendingImage(src);
      };
      img.src = src;
    };
    reader.readAsDataURL(file);
    e.target.value = '';
  }

  useEffect(() => {
    if (!canChat || isSending) return;
    chatInputRef.current?.focus();
  }, [activeConversationId, canChat, isSending]);

  function startInlineRename(conversation: ChatConversation) {
    setEditingConversationId(conversation.id);
    setEditingTitle(conversation.title);
  }

  function cancelInlineRename() {
    setEditingConversationId(null);
    setEditingTitle('');
  }

  function saveInlineRename() {
    if (!editingConversationId) return;
    onConversationRename(editingConversationId, editingTitle);
    cancelInlineRename();
  }

  function handleDelete(conversation: ChatConversation) {
    if (!window.confirm(`Delete "${conversation.title}"?`)) return;
    onConversationDelete(conversation.id);
  }

  function handleClearAll() {
    if (!window.confirm('Clear all chats?')) return;
    onConversationsClear();
  }

  // Conversation list content — reused in desktop sidebar and mobile sheet
  const conversationListContent = (
    <div className="space-y-3 p-3">
      <Button type="button" size="sm" className="w-full" onClick={() => { onConversationCreate(); setMobileSidebarOpen(false); }} disabled={isSending}>
        <MessageSquarePlus className="mr-1.5 h-4 w-4" />
        New chat
      </Button>
      <div className="flex items-center justify-between gap-2">
        <div className="text-xs text-muted-foreground">Conversations</div>
        <Button type="button" variant="ghost" size="sm" className="h-7 px-2 text-xs" onClick={handleClearAll} disabled={!hasChats || isSending}>
          <Trash2 className="mr-1.5 h-3.5 w-3.5" />
          Clear
        </Button>
      </div>
      <ScrollArea className="h-[calc(100svh_-_14rem)] md:h-[calc(100svh_-_24rem)]">
        <div className="space-y-1">
          {conversations.map((conversation) => {
            const isActive = conversation.id === activeConversationId;
            const isEditing = editingConversationId === conversation.id;
            return (
              <div key={conversation.id} className={cn('group flex items-center gap-2 rounded-md border p-2', isActive ? 'border-primary/40 bg-muted/40' : 'border-transparent')}>
                {isEditing ? (
                  <div className="min-w-0 flex-1 space-y-1">
                    <input
                      value={editingTitle}
                      onChange={(e) => setEditingTitle(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          e.preventDefault();
                          saveInlineRename();
                        } else if (e.key === 'Escape') {
                          e.preventDefault();
                          cancelInlineRename();
                        }
                      }}
                      className="h-7 w-full rounded-md border bg-background px-2 text-sm outline-none ring-offset-background focus:ring-2 focus:ring-ring"
                      autoFocus
                    />
                    <div className="text-xs text-muted-foreground">{conversation.messages.length} message{conversation.messages.length === 1 ? '' : 's'}</div>
                  </div>
                ) : (
                  <button
                    type="button"
                    className="min-w-0 flex-1 text-left"
                    onClick={() => { onConversationSelect(conversation.id); setMobileSidebarOpen(false); }}
                    disabled={isSending}
                  >
                    <div className="text-sm font-medium leading-5 [display:-webkit-box] [overflow:hidden] [-webkit-box-orient:vertical] [-webkit-line-clamp:3] [overflow-wrap:anywhere]">
                      {conversation.title}
                    </div>
                    <div className="text-xs text-muted-foreground">{conversation.messages.length} message{conversation.messages.length === 1 ? '' : 's'}</div>
                  </button>
                )}
                {isEditing ? (
                  <>
                    <Button type="button" variant="ghost" size="icon" className="h-7 w-7 shrink-0 opacity-70 hover:opacity-100" onClick={saveInlineRename} aria-label="Save conversation name">
                      <Check className="h-3.5 w-3.5" />
                    </Button>
                    <Button type="button" variant="ghost" size="icon" className="h-7 w-7 shrink-0 opacity-70 hover:opacity-100" onClick={cancelInlineRename} aria-label="Cancel rename">
                      <X className="h-3.5 w-3.5" />
                    </Button>
                  </>
                ) : (
                  <Button type="button" variant="ghost" size="icon" className="h-7 w-7 shrink-0 opacity-70 hover:opacity-100" onClick={() => startInlineRename(conversation)} disabled={isSending} aria-label="Rename conversation">
                    <Pencil className="h-3.5 w-3.5" />
                  </Button>
                )}
                <Button type="button" variant="ghost" size="icon" className="h-7 w-7 shrink-0 opacity-70 hover:opacity-100" onClick={() => handleDelete(conversation)} disabled={isSending} aria-label="Delete conversation">
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );

  return (
    <Card className="flex h-full min-h-0 flex-1 flex-col overflow-hidden">
      <CardHeader className="px-3 py-2 md:px-6 md:py-4">
        <div className="flex items-center gap-2 md:gap-3">
          {/* Mobile: chats button (opens sheet) + new chat */}
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 shrink-0 gap-1.5 md:hidden"
            onClick={() => hasChats ? setMobileSidebarOpen(true) : onConversationCreate()}
            aria-label={hasChats ? 'Chats' : 'New chat'}
          >
            {hasChats ? (
              <>
                <Hash className="h-4 w-4" />
                <span className="text-xs tabular-nums">{conversations.length}</span>
              </>
            ) : <MessageSquarePlus className="h-4 w-4" />}
          </Button>
          {hasChats ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8 shrink-0 md:hidden"
              onClick={onConversationCreate}
              disabled={isSending}
              aria-label="New chat"
            >
              <MessageSquarePlus className="h-4 w-4" />
            </Button>
          ) : null}
          <CardTitle className="hidden md:block text-base shrink-0">Chat</CardTitle>
          <div className="ml-auto flex items-center gap-2">
            {selectedModelNodeCount != null ? (
              <div className="hidden md:flex h-8 items-center gap-1.5 rounded-md border bg-muted/40 px-2">
                <Network className="h-3.5 w-3.5 text-muted-foreground" />
                <div className="text-xs leading-none">
                  <span className="font-medium">{selectedModelNodeCount}</span>
                  <span className="ml-1 text-muted-foreground">nodes</span>
                </div>
              </div>
            ) : null}
            {selectedModelVramGb != null ? (
              <div className="hidden md:flex h-8 items-center gap-1.5 rounded-md border bg-muted/40 px-2">
                <Cpu className="h-3.5 w-3.5 text-muted-foreground" />
                <div className="text-xs leading-none">
                  <span className="font-medium">{selectedModelVramGb.toFixed(1)}</span>
                  <span className="ml-1 text-muted-foreground">GB</span>
                </div>
              </div>
            ) : null}
            <span className="hidden md:inline text-xs text-muted-foreground">Model</span>
            <Select value={selectedModelValue} onValueChange={setSelectedModel} disabled={!warmModels.length}>
              <SelectTrigger className="h-8 w-full min-w-0 max-w-[180px] md:max-w-[320px] md:w-[320px]">
                <SelectValue placeholder="Select model">
                  {selectedModelValue === 'auto' ? '✨ Auto (router picks best)' : selectedModelValue ? shortName(selectedModelValue) : undefined}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {warmModels.length > 1 ? (
                  <SelectItem
                    key="auto"
                    value="auto"
                    className="group py-2 data-[state=checked]:bg-accent data-[state=checked]:text-accent-foreground"
                  >
                    <div className="flex min-w-0 flex-col gap-0.5">
                      <span className="leading-5">✨ Auto</span>
                      <span className="text-xs leading-4 text-muted-foreground group-data-[highlighted]:text-accent-foreground group-data-[state=checked]:text-accent-foreground">
                        Router picks best model for each request
                      </span>
                    </div>
                  </SelectItem>
                ) : null}
                {warmModels.map((model) => {
                  const modelStats = modelStatsByName[model];
                  return (
                    <SelectItem
                      key={model}
                      value={model}
                      className="group py-2 data-[state=checked]:bg-accent data-[state=checked]:text-accent-foreground"
                    >
                      <div className="flex min-w-0 flex-col gap-0.5">
                        <span className="truncate leading-5">
                          {shortName(model)}
                          {visionModels.has(model) && <span className="ml-1.5" title="Vision — understands images">👁</span>}
                        </span>
                        {modelStats ? (
                          <span className="grid grid-cols-[108px_132px] gap-x-3 text-xs leading-4 text-muted-foreground group-data-[highlighted]:text-accent-foreground group-data-[state=checked]:text-accent-foreground">
                            <span className="inline-flex items-center gap-1">
                              <Network className="h-3 w-3 text-muted-foreground group-data-[highlighted]:text-accent-foreground group-data-[state=checked]:text-accent-foreground" />
                              <span>Nodes</span>
                              <span className="font-medium tabular-nums text-foreground group-data-[highlighted]:text-accent-foreground group-data-[state=checked]:text-accent-foreground">{modelStats.nodes}</span>
                            </span>
                            <span className="inline-flex items-center gap-1">
                              <Cpu className="h-3 w-3 text-muted-foreground group-data-[highlighted]:text-accent-foreground group-data-[state=checked]:text-accent-foreground" />
                              <span>VRAM</span>
                              <span className="font-medium tabular-nums text-foreground group-data-[highlighted]:text-accent-foreground group-data-[state=checked]:text-accent-foreground">{modelStats.vramGb.toFixed(1)} GB</span>
                            </span>
                          </span>
                        ) : null}
                      </div>
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>
      <Separator />
      {props.isFlyHosted ? (
        <div className={cn(
          'border-b px-4 py-2 text-xs',
          props.inflightRequests > 2
            ? 'bg-orange-500/10 text-orange-700 dark:text-orange-400'
            : 'bg-muted/40 text-muted-foreground',
        )}>
          {props.inflightRequests > 2 ? (
            <>
              <span className="font-medium">⏳ Busy</span> — {props.inflightRequests} requests in flight, responses may be slow.{' '}
              For direct access run <code className="rounded bg-muted px-1 py-0.5 font-mono">mesh-llm --auto</code>{' '}
              <a href={DOCS_URL} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">Learn more →</a>
            </>
          ) : (
            <>
              <span className="font-medium">Community demo</span> — best-effort public instance.
              For direct, faster access run{' '}
              <code className="rounded bg-muted px-1 py-0.5 font-mono">mesh-llm --auto</code>{' '}
              to join the mesh or start your own.{' '}
              <a href={DOCS_URL} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">Learn more →</a>
            </>
          )}
        </div>
      ) : null}
      {/* Mobile conversation sheet */}
      <Sheet open={mobileSidebarOpen} onOpenChange={setMobileSidebarOpen}>
        <SheetContent side="left" className="w-72 p-0">
          <SheetHeader className="sr-only">
            <SheetTitle>Chats</SheetTitle>
          </SheetHeader>
          {conversationListContent}
        </SheetContent>
      </Sheet>

      <CardContent className="min-h-0 flex-1 p-0">
        <div className="flex h-full min-h-0 min-w-0 md:flex-row">
          {/* Desktop sidebar — hidden on mobile */}
          {hasChats ? (
            <aside className="hidden md:block shrink-0 md:w-72 md:border-r">
              {conversationListContent}
            </aside>
          ) : null}

          <div className="flex min-h-0 min-w-0 flex-1 flex-col">
            <div
              ref={chatScrollRef}
              className={cn(
                'min-h-0 flex-1 overflow-x-hidden overflow-y-auto px-3 py-4 md:px-6',
                messages.length === 0 ? '' : 'space-y-4',
              )}
            >
              {messages.length === 0 ? (
                <div className="flex min-h-full items-center justify-center">
                  <InviteFriendEmptyState inviteToken={inviteToken} selectedModel={selectedModel || warmModels[0] || ''} isPublicMesh={props.isPublicMesh} />
                </div>
              ) : (
                <>
                  {messages.map((message, i) => (
                    <ChatBubble
                      key={message.id}
                      message={message}
                      reasoningOpen={!!reasoningOpen[message.id]}
                      onReasoningToggle={(open) => setReasoningOpen((prev) => ({ ...prev, [message.id]: open }))}
                      streaming={isSending && i === messages.length - 1}
                    />
                  ))}

                  {isSending ? (
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Loader2 className="h-3.5 w-3.5 animate-spin" /> Streaming response...
                    </div>
                  ) : null}
                </>
              )}
            </div>
            <Separator />
            <div className="shrink-0 box-border w-full max-w-full space-y-2 overflow-hidden p-3 md:space-y-3 md:p-4">
              {pendingImage && (
                <div className="relative inline-block">
                  <img src={pendingImage} alt="Attached" className="h-20 rounded-md border object-cover" />
                  <button
                    onClick={() => setPendingImage(null)}
                    className="absolute -right-1.5 -top-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-destructive text-destructive-foreground text-xs hover:bg-destructive/80"
                    aria-label="Remove image"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              )}
              <Textarea
                ref={chatInputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    onSubmit();
                  }
                }}
                rows={2}
                placeholder={props.canChat ? 'Ask me anything...' : 'Waiting for a warm model...'}
                disabled={!props.canChat || isSending}
                className="min-h-[56px] md:min-h-[80px] resize-none text-base md:text-sm"
              />
              <div className="flex items-center justify-between gap-2">
                <div className="hidden md:block text-xs text-muted-foreground">Enter to send. Shift+Enter for newline.</div>
                <div className="flex items-center gap-2">
                  {selectedModelVision && (
                    <>
                      <input
                        ref={imageInputRef}
                        type="file"
                        accept="image/*"
                        className="hidden"
                        onChange={handleImageSelect}
                      />
                      <Button
                        type="button"
                        variant="outline"
                        size="icon"
                        onClick={() => imageInputRef.current?.click()}
                        disabled={!props.canChat || isSending}
                        title="Attach image"
                        aria-label="Attach image"
                      >
                        <ImagePlus className="h-4 w-4" />
                      </Button>
                    </>
                  )}
                  {isSending ? (
                    <Button type="button" variant="outline" size="icon" onClick={onStop} aria-label="Stop">
                      <Square className="h-3.5 w-3.5" />
                    </Button>
                  ) : (
                    <Button
                      type="button"
                      variant="outline"
                      size="icon"
                      onClick={onRegenerate}
                      disabled={!canRegenerate}
                      aria-label="Regenerate"
                    >
                      <RotateCcw className="h-4 w-4" />
                    </Button>
                  )}
                  <Button
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={onSubmit}
                    disabled={!props.canChat || (!input.trim() && !pendingImage) || isSending}
                  >
                    {isSending ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Send className="mr-2 h-4 w-4" />}
                    Send
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function InviteFriendEmptyState({ inviteToken, selectedModel, isPublicMesh }: { inviteToken: string; selectedModel: string; isPublicMesh: boolean }) {
  const [open, setOpen] = useState(false);
  const [inviteWithModelCopied, setInviteWithModelCopied] = useState(false);
  const inviteWithModelCommand = inviteToken && selectedModel ? `mesh-llm --join ${inviteToken} --model ${selectedModel}` : '';

  async function copyInviteWithModelCommand() {
    if (!inviteWithModelCommand) return;
    try {
      await navigator.clipboard.writeText(inviteWithModelCommand);
      setInviteWithModelCopied(true);
      window.setTimeout(() => setInviteWithModelCopied(false), 1500);
    } catch {
      setInviteWithModelCopied(false);
    }
  }

  if (isPublicMesh) {
    return (
      <div className="mx-auto w-full max-w-md space-y-4 px-2 text-center">
        <div className="flex justify-center">
          <BrandIcon className="h-12 w-12 text-primary/50 animate-wiggle" />
        </div>
        <p className="text-sm text-muted-foreground">
          Mesh LLM is a project to let people contribute spare compute, build private personal AI, using open models.
        </p>
        <button
          type="button"
          onClick={() => setOpen(!open)}
          className="mx-auto flex items-center gap-1.5 text-xs text-muted-foreground/70 hover:text-foreground transition-colors"
        >
          <ChevronDown className={cn('h-3 w-3 transition-transform', open ? '' : '-rotate-90')} />
          <span>Learn more…</span>
        </button>
        {open ? (
          <div className="space-y-4 rounded-md border border-dashed p-3 text-left">
            <div className="text-xs text-muted-foreground">
              <a href={DOCS_URL} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
                Learn about this project →
              </a>
            </div>
            <Separator />
            <div className="space-y-2">
              <div className="text-xs font-medium">Contribute to the pool</div>
              <div className="text-xs text-muted-foreground">
                Have a spare machine? Add it to this mesh and share compute with others.
              </div>
              <code className="block rounded-md border bg-muted/40 px-2 py-1.5 text-xs">
                mesh-llm --auto
              </code>
            </div>
            <Separator />
            <div className="space-y-2">
              <div className="text-xs font-medium">Run your own private mesh</div>
              <div className="text-xs text-muted-foreground">
                Pool machines across your home, office, or friends — fully private, no cloud needed.{' '}
                <a href={DOCS_URL} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
                  Getting started →
                </a>
              </div>
            </div>
            <Separator />
            <div className="space-y-2">
              <div className="text-xs font-medium">Use with coding agents</div>
              <div className="text-xs text-muted-foreground">
                Works with Claude Code, Goose, pi, and any OpenAI-compatible client.{' '}
                <a href={`${DOCS_URL}/#agents`} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
                  Agent setup →
                </a>
              </div>
            </div>
            <Separator />
            <div className="space-y-2">
              <div className="text-xs font-medium">Agent gossip</div>
              <div className="text-xs text-muted-foreground">
                Let your agents coordinate across machines — share status, findings, and questions. Works with any LLM setup.{' '}
                <a href={`${DOCS_URL}/#blackboard`} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
                  Blackboard docs →
                </a>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    );
  }

  // Private mesh — invite a friend to join
  return (
    <div className="mx-auto w-full max-w-md space-y-3 px-2 text-center">
      <div className="flex justify-center">
        <BrandIcon className="h-12 w-12 text-primary/50 animate-wiggle" />
      </div>
      <p className="text-sm text-muted-foreground">
        Mesh LLM lets you build private personal AI, using open models.{' '}
        <a href={DOCS_URL} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
          Learn more →
        </a>
      </p>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="mx-auto flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
      >
        <ChevronDown className={cn('h-3.5 w-3.5 transition-transform', open ? '' : '-rotate-90')} />
        <Network className="h-3.5 w-3.5" />
        <span>Invite someone to the mesh</span>
      </button>
      {open && inviteWithModelCommand ? (
        <div className="space-y-2 rounded-md border border-dashed p-3 text-left">
          <div className="text-xs text-muted-foreground">
            Share this command — they'll join and contribute compute:
          </div>
          <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
            <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
              {inviteWithModelCommand}
            </code>
            <Button
              type="button"
              size="icon"
              variant="ghost"
              className="h-7 w-7 shrink-0"
              aria-label="Copy command"
              onClick={() => void copyInviteWithModelCommand()}
            >
              {inviteWithModelCopied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
            </Button>
          </div>
          <div className="text-xs text-muted-foreground pt-1">
            Don't have it yet?{' '}
            <a href={`${DOCS_URL}/#install`} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
              Install mesh-llm →
            </a>
          </div>
          <div className="text-xs text-muted-foreground pt-1">
            Agents can gossip too —{' '}
            <a href={`${DOCS_URL}/#blackboard`} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
              Blackboard docs →
            </a>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function DashboardPage({
  status,
  topologyNodes,
  selectedModel,
  themeMode,
  isPublicMesh,
  inviteToken,
  isLocalhost,
}: {
  status: StatusPayload | null;
  topologyNodes: TopologyNode[];
  selectedModel: string;
  themeMode: ThemeMode;
  isPublicMesh: boolean;
  inviteToken: string;
  isLocalhost: boolean;
}) {
  const [modelFilter, setModelFilter] = useState<'all' | 'warm' | 'cold'>('all');
  const [isMeshOverviewFullscreen, setIsMeshOverviewFullscreen] = useState(false);
  const filteredModels = useMemo(() => {
    const models = status?.mesh_models ?? [];
    return [...models]
      .filter((m) => (modelFilter === 'all' ? true : m.status === modelFilter))
      .sort((a, b) => (b.node_count - a.node_count) || a.name.localeCompare(b.name));
  }, [status?.mesh_models, modelFilter]);
  const totalMeshVramGb = useMemo(() => meshGpuVram(status), [status]);
  const sortedPeers = useMemo(() => {
    return [...(status?.peers ?? [])].sort((a, b) => (b.vram_gb - a.vram_gb) || a.id.localeCompare(b.id));
  }, [status?.peers]);
  const peerRows = useMemo(() => {
    return sortedPeers.map((peer) => {
      const statusLabel = peer.role === 'Client'
        ? 'Client'
        : peer.serving && peer.serving !== '(idle)'
          ? 'Serving'
          : peer.role === 'Host'
            ? 'Host'
            : 'Idle';
      const modelLabel = peer.serving && peer.serving !== '(idle)' ? shortName(peer.serving) : 'idle';
      const latencyLabel = formatLatency(peer.rtt_ms);
      const sharePct = peer.role !== 'Client' && totalMeshVramGb > 0
        ? Math.round((Math.max(0, peer.vram_gb) / totalMeshVramGb) * 100)
        : null;
      return {
        ...peer,
        statusLabel,
        modelLabel,
        latencyLabel,
        shareLabel: sharePct == null ? 'n/a' : `${sharePct}%`,
      };
    });
  }, [sortedPeers, totalMeshVramGb]);

  useEffect(() => {
    const prevOverflow = document.body.style.overflow;
    if (isMeshOverviewFullscreen) document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = prevOverflow;
    };
  }, [isMeshOverviewFullscreen]);

  useEffect(() => {
    if (!isMeshOverviewFullscreen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      setIsMeshOverviewFullscreen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isMeshOverviewFullscreen]);

  function toggleMeshOverviewFullscreen() {
    setIsMeshOverviewFullscreen((prev) => !prev);
  }

  return (
    <div className="space-y-4">
      <Alert className="border-primary/20 bg-primary/5">
        <Network className="h-4 w-4" />
        <AlertTitle className="text-sm font-medium">
          {isPublicMesh ? 'Welcome to the public mesh' : 'Your private mesh'}
        </AlertTitle>
        <AlertDescription className="text-xs text-muted-foreground">
          {isPublicMesh
            ? 'Mesh LLM is a project to let people contribute spare compute, build private personal AI, using open models.'
            : 'Mesh LLM lets you build private personal AI, using open models. Pool machines across your home, office, or friends, no cloud needed.'}
          {' '}
          <a href={DOCS_URL} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
            Learn more →
          </a>
          {' · '}
          <a href="https://github.com/michaelneale/mesh-llm" target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 underline hover:text-foreground">
            GitHub
          </a>
        </AlertDescription>
      </Alert>
      <TooltipProvider>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <StatCard
          title="Node ID"
          value={status?.node_id ?? 'n/a'}
          valueSuffix={(
            <Badge className={cn('h-6 px-2 text-[10px] font-semibold tracking-wide', topologyStatusClass(status?.node_status ?? 'n/a'))}>
              {status?.node_status ?? 'n/a'}
            </Badge>
          )}
          icon={<Hash className="h-4 w-4" />}
          tooltip="Current node identifier in this mesh."
        />
        <StatCard
          title="Active Models"
          value={`${(status?.mesh_models ?? []).filter((m) => m.status === 'warm').length}`}
          icon={<Sparkles className="h-4 w-4" />}
          tooltip="Models currently loaded and serving across the mesh."
        />
        <StatCard
          title="Mesh VRAM"
          value={`${meshGpuVram(status).toFixed(1)} GB`}
          icon={<Cpu className="h-4 w-4" />}
          tooltip="Total GPU VRAM across non-client nodes in the mesh."
        />
        <StatCard
          title="Nodes"
          value={`${topologyNodes.length}`}
          icon={<Network className="h-4 w-4" />}
          tooltip="Total nodes currently visible in topology."
        />
        <StatCard
          title="Inflight"
          value={`${status?.inflight_requests ?? 0}`}
          icon={<Gauge className="h-4 w-4" />}
          tooltip="Current in-flight request count."
        />
        </div>
      </TooltipProvider>

      <div className="grid items-start gap-4 lg:grid-cols-7">
        <div className="lg:col-span-5">
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between gap-2">
                <CardTitle className="text-sm">Mesh Overview</CardTitle>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-8 gap-1.5"
                  onClick={() => void toggleMeshOverviewFullscreen()}
                >
                  <Maximize2 className="h-3.5 w-3.5" />
                  Fullscreen
                </Button>
              </div>
            </CardHeader>
            <CardContent className="pt-0">
              {isMeshOverviewFullscreen ? (
                <div className="flex h-[360px] md:h-[420px] lg:h-[460px] xl:h-[520px] items-center justify-center rounded-lg border border-dashed bg-muted/20 text-sm text-muted-foreground">
                  Mesh Overview is open in fullscreen.
                </div>
              ) : (
                <MeshTopologyDiagram
                  status={status}
                  nodes={topologyNodes}
                  selectedModel={selectedModel}
                  themeMode={themeMode}
                  fullscreen={false}
                  heightClass="h-[360px] md:h-[420px] lg:h-[460px] xl:h-[520px]"
                />
              )}
            </CardContent>
          </Card>
        </div>

        <Card className="lg:col-span-2">
          <CardHeader className="pb-2">
            <div className="flex flex-wrap items-center gap-2">
              <CardTitle className="text-sm">Model Catalog</CardTitle>
              <div className="ml-auto flex shrink-0 items-center gap-2">
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
          <CardContent className="pt-0">
            {filteredModels.length > 0 ? (
              <div className="h-[360px] overflow-y-auto pr-2 md:h-[420px] lg:h-[460px] xl:h-[520px]">
                <div className="space-y-2">
                  {filteredModels.map((model) => (
                    <div key={model.name} className="rounded-md border p-3">
                      <div className="flex flex-col items-start gap-2 sm:flex-row sm:items-start">
                        <div className="flex h-7 w-7 items-center justify-center rounded-md border bg-muted/40 text-muted-foreground">
                          <Sparkles className="h-3.5 w-3.5" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="text-sm font-medium leading-5 [overflow-wrap:anywhere]">{shortName(model.name)}</div>
                          <div className="text-xs leading-4 text-muted-foreground [overflow-wrap:anywhere]">{model.name}</div>
                        </div>
                        <Badge
                          className={cn(
                            'shrink-0 gap-1 self-start',
                            model.status === 'warm' ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300' : '',
                            model.status === 'cold' ? 'border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300' : '',
                          )}
                        >
                          <span className="h-1.5 w-1.5 rounded-full bg-current" />
                          {model.status === 'warm' ? 'Warm' : model.status === 'cold' ? 'Cold' : model.status}
                        </Badge>
                      </div>
                      <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
                        <span>{model.node_count} node{model.node_count === 1 ? '' : 's'}</span>
                        <span className="flex items-center gap-2">
                          {model.vision && <span title="Vision — understands images">👁</span>}
                          {model.size_gb.toFixed(1)} GB
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="h-[360px] md:h-[420px] lg:h-[460px] xl:h-[520px]">
                <DashboardPanelEmpty
                  icon={<Sparkles className="h-4 w-4" />}
                  title={(status?.mesh_models.length ?? 0) > 0 ? `No ${modelFilter} models` : 'No model catalog data'}
                  description={(status?.mesh_models.length ?? 0) > 0 ? 'Try changing the model filter.' : 'Model metadata will appear once the mesh reports available models.'}
                />
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Connected Peers</CardTitle>
        </CardHeader>
        <CardContent className="min-h-0 pt-0">
          {peerRows.length > 0 ? (
            <ScrollArea horizontal className="max-h-[18rem] md:max-h-[20rem]">
              <div className="pr-3">
                <Table className="min-w-[920px]">
                  <TableHeader>
                    <TableRow>
                      <TableHead>ID</TableHead>
                      <TableHead>Role</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Model</TableHead>
                      <TableHead className="text-right">Latency</TableHead>
                      <TableHead className="text-right">VRAM</TableHead>
                      <TableHead className="text-right whitespace-nowrap">Share</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {peerRows.map((peer) => (
                      <TableRow key={peer.id}>
                        <TableCell className="font-mono text-xs">{peer.id}</TableCell>
                        <TableCell>{peer.role}</TableCell>
                        <TableCell>{peer.statusLabel}</TableCell>
                        <TableCell className="max-w-[180px] truncate">{peer.modelLabel}</TableCell>
                        <TableCell className="text-right">{peer.latencyLabel}</TableCell>
                        <TableCell className="text-right">{peer.vram_gb.toFixed(1)} GB</TableCell>
                        <TableCell className="text-right whitespace-nowrap">{peer.shareLabel}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </ScrollArea>
          ) : (
            <DashboardPanelEmpty
              icon={<Network className="h-4 w-4" />}
              title="No peers connected"
              description="Invite another node to join this mesh to see connected peers."
            />
          )}
        </CardContent>
      </Card>

      {isMeshOverviewFullscreen && typeof document !== 'undefined'
        ? createPortal(
          <div className="fixed inset-0 z-[120] bg-black/55 backdrop-blur-sm">
            <div className="h-full w-full p-3 md:p-4">
              <Card className="flex h-full min-h-0 w-full flex-col shadow-2xl shadow-black/65">
                <CardHeader className="shrink-0 pb-2">
                  <div className="flex items-center justify-between gap-2">
                    <CardTitle className="text-sm">Mesh Overview</CardTitle>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="h-8 gap-1.5"
                      onClick={() => void toggleMeshOverviewFullscreen()}
                    >
                      <Minimize2 className="h-3.5 w-3.5" />
                      Exit Fullscreen
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="flex min-h-0 flex-1 p-0">
                <MeshTopologyDiagram
                  status={status}
                  nodes={topologyNodes}
                  selectedModel={selectedModel}
                  themeMode={themeMode}
                  fullscreen
                  heightClass="min-h-[420px]"
                  containerStyle={{ width: '100%', height: 'calc(100dvh - 8rem)', minHeight: 420 }}
                />
              </CardContent>
            </Card>
          </div>
          </div>,
          document.body,
        )
        : null}

      {/* Connect panel */}
      <Card className="mt-4">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Connect</CardTitle>
          <div className="text-xs text-muted-foreground">
            Run mesh-llm on your machine to get a local OpenAI-compatible API and contribute compute to the mesh.
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1.5">
            <div className="text-xs font-medium">1. Install</div>
            <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
              <a href="https://docs.anarchai.org/#install" target="_blank" rel="noopener noreferrer" className="min-w-0 flex-1 text-xs text-primary underline hover:text-foreground">
                docs.anarchai.org/#install
              </a>
            </div>
          </div>
          <div className="space-y-1.5">
            <div className="text-xs font-medium">2. Run</div>
            {(() => {
              const cmd = isPublicMesh ? 'mesh-llm --auto' : `mesh-llm --auto --join ${inviteToken || '(token)'}`;
              return (
                <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
                  <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">{cmd}</code>
                  <Button type="button" size="icon" variant="ghost" className="h-6 w-6 shrink-0" aria-label="Copy" onClick={() => void navigator.clipboard.writeText(cmd)}>
                    <Copy className="h-3 w-3" />
                  </Button>
                </div>
              );
            })()}
            <div className="text-xs text-muted-foreground">
              This auto-selects a model for your hardware, joins the mesh, and serves a local API at <code className="text-[0.7rem]">http://127.0.0.1:9337/v1</code>
            </div>
          </div>
          {isLocalhost ? null : (
            <div className="text-xs text-muted-foreground">
              <a href={DOCS_URL} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
                Full docs →
              </a>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="flex items-center justify-center gap-3 py-2 text-xs text-muted-foreground">
        <a href={DOCS_URL} target="_blank" rel="noopener noreferrer" className="underline-offset-2 hover:text-foreground hover:underline">
          Docs
        </a>
        <span>·</span>
        <a href={`${DOCS_URL}/#agents`} target="_blank" rel="noopener noreferrer" className="underline-offset-2 hover:text-foreground hover:underline">
          Agents
        </a>
        <span>·</span>
        <a href={`${DOCS_URL}/#models`} target="_blank" rel="noopener noreferrer" className="underline-offset-2 hover:text-foreground hover:underline">
          Models
        </a>
        <span>·</span>
        <a href={`${DOCS_URL}/#running`} target="_blank" rel="noopener noreferrer" className="underline-offset-2 hover:text-foreground hover:underline">
          Common patterns
        </a>
      </div>
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
  statusLabel: string;
  latencyMs?: number | null;
  loadedModel: string;
  loadedModels: string[];
  vramGb: number;
  vramSharePct: number;
  hostname?: string;
  isSoc?: boolean;
  gpus?: { name: string; vram_bytes: number }[];
};

type TopologyFlowNodeData = {
  node: PositionedTopologyNode;
  info: TopologyNodeInfo;
  selected: boolean;
  sameModelAsCurrent: boolean;
};

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
                {data.info.loadedModels.map((m) => (
                  <div key={m} className="inline-flex items-center gap-1 text-[11px] font-medium leading-4">
                    <Sparkles className="h-3 w-3 shrink-0 text-muted-foreground" />
                    <span className="min-w-0 truncate" title={m}>{shortName(m)}</span>
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
          {data.info.hostname && (
            <div className="inline-flex items-center gap-1 rounded-full border bg-muted/30 px-2 py-1">
              <Server className="h-3 w-3 text-muted-foreground" />
              <span className="text-muted-foreground">Host</span>
              <span className="font-medium">{data.info.hostname}</span>
            </div>
          )}
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
            <MemoryStick className="h-3 w-3 text-muted-foreground" />
            <span className="text-muted-foreground">VRAM</span>
            <span className="font-medium">{data.info.vramGb.toFixed(1)} GB</span>
          </div>
          <div className="inline-flex items-center gap-1 rounded-full border bg-muted/30 px-2 py-1">
            <Gauge className="h-3 w-3 text-muted-foreground" />
            <span className="text-muted-foreground">Share</span>
            <span className="whitespace-nowrap font-medium">{data.info.vramSharePct}%</span>
          </div>
          {data.info.gpus?.map((gpu, i) => {
              const lower = gpu.name.toLowerCase();
              const isNvidia = lower.includes("nvidia") || lower.includes("jetson");
              const isAmd = lower.includes("amd");
              const isIntel = lower.includes("intel");
              const iconColor = isNvidia
                ? "#76b900"
                : isAmd
                  ? "#ED1C24"
                  : isIntel
                    ? "#0071C5"
                    : undefined;
              const model = gpu.name
                .replace(/^NVIDIA GeForce\s+/i, "")
                .replace(/^NVIDIA Quadro\s+/i, "")
                .replace(/^NVIDIA\s+/i, "")
                .replace(/^AMD Radeon\s+/i, "")
                .replace(/^AMD\s+/i, "")
                .replace(/^Intel Arc\s+/i, "")
                .replace(/^Intel\s+/i, "")
                .replace(/^Apple\s+/i, "")
                .trim();
              const vramGb = gpu.vram_bytes / (1024 * 1024 * 1024);
              const GpuIcon = data.info.isSoc ? Cpu : Gpu;
              return (
                <div
                  key={`${gpu.name}-${i}`}
                  className="group/gpu inline-flex items-center gap-1 rounded-full border bg-muted/30 px-2 py-1"
                >
                  <GpuIcon
                    className="h-3 w-3"
                    style={iconColor ? { color: iconColor } : undefined}
                  />
                  <span className="text-muted-foreground">{data.info.isSoc ? "SoC" : "GPU"}</span>
                  <span className="relative inline-flex font-medium">
                    <span className="group-hover/gpu:invisible">
                      {model}
                    </span>
                    <span className="absolute left-0 top-0 whitespace-nowrap opacity-0 transition-opacity duration-200 group-hover/gpu:opacity-100">
                      {`${Math.round(vramGb)} GB`}
                    </span>
                  </span>
                </div>
              );
            })}
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
    return <EmptyPanel text="No topology data yet." />;
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
  const clientEdgeStride = total > 320 ? 6 : total > 220 ? 4 : total > 120 ? 2 : 1;
  const meshVramGb = nodes.filter((n) => !n.client).reduce((sum, n) => sum + Math.max(0, n.vram), 0);
  const currentNodeServingModel = useMemo(() => {
    const current = nodes.find((n) => n.self);
    if (!current || current.client || !current.serving || current.serving === '(idle)') return '';
    return current.serving;
  }, [nodes]);

  const [selectedNodeId, setSelectedNodeId] = useState(center.id);

  useEffect(() => {
    setSelectedNodeId((prev) => (nodes.some((n) => n.id === prev) ? prev : center.id));
  }, [nodes, center.id]);

  const nodeInfoById = useMemo(() => {
    const out = new Map<string, TopologyNodeInfo>();
    for (const node of nodes) {
      const servingModel = !node.client && node.serving && node.serving !== '(idle)' ? node.serving : '';
      const servingModels = !node.client ? node.servingModels.filter((m) => m && m !== '(idle)') : [];
      const role = node.client ? 'Client' : node.host ? 'Host' : servingModel ? 'Worker' : 'Idle';
      const vramSharePct = !node.client && meshVramGb > 0 ? Math.round((Math.max(0, node.vram) / meshVramGb) * 100) : 0;
      out.set(node.id, {
        role,
        statusLabel: node.statusLabel,
        latencyMs: node.latencyMs ?? null,
        loadedModel: node.client ? 'n/a' : servingModels.length > 0 ? servingModels.map(shortName).join(', ') : servingModel ? shortName(servingModel) : 'idle',
        loadedModels: node.client ? [] : servingModels.length > 0 ? servingModels : (servingModel ? [servingModel] : []),
        vramGb: Math.max(0, node.vram),
        vramSharePct,
        hostname: node.hostname,
        isSoc: node.isSoc,
        gpus: node.gpus,
      });
    }
    return out;
  }, [nodes, meshVramGb]);
  const flowColorMode = themeMode === 'auto'
    ? (typeof document !== 'undefined' && document.documentElement.classList.contains('dark') ? 'dark' : 'light')
    : themeMode;
  const flowLayoutKey = useMemo(
    () => `${fullscreen ? 'fs' : 'std'}:${positioned.map((p) => p.id).sort().join(',')}`,
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
    return positioned.map((p) => ({
      id: p.id,
      type: 'topologyNode',
      position: { x: p.x, y: p.y },
      origin: [0.5, 0],
      data: {
        node: p,
        info: nodeInfoById.get(p.id) ?? {
          role: 'Node',
          statusLabel: 'n/a',
          latencyMs: null,
          loadedModel: 'idle',
          loadedModels: [],
          vramGb: 0,
          vramSharePct: 0,
        },
        selected: p.id === selectedNodeId,
        sameModelAsCurrent: !!currentNodeServingModel && !p.client && p.serving === currentNodeServingModel,
      },
      draggable: false,
      selectable: false,
      connectable: false,
      zIndex: p.id === center.id ? 10 : 1,
    }));
  }, [positioned, nodeInfoById, selectedNodeId, center.id, currentNodeServingModel]);

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
          className: `mesh-edge mesh-edge--${p.bucket}`,
          animated: false,
          style: {
            stroke,
            strokeWidth: p.bucket === 'client' ? 1.8 : 2.4,
            strokeDasharray: p.bucket === 'client' ? '2 8' : '2 6',
          },
        };
      });
  }, [positioned, center.id, clientEdgeStride]);

  return (
    <div className={cn(
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
  const placeRow = (
    row: TopologyNode[],
    bucket: PositionedTopologyNode['bucket'],
    y: number,
    horizontalSpacing: number,
    positioned: PositionedTopologyNode[],
  ) => {
    const startX = -((row.length - 1) * horizontalSpacing) / 2;
    row.forEach((node, index) => {
      positioned.push({
        ...node,
        bucket,
        x: startX + (index * horizontalSpacing),
        y,
      });
    });
  };

  const all: Array<PositionedTopologyNode> = [
    ...serving.map((n) => ({ ...n, bucket: 'serving' as const, x: 0, y: 0 })),
    ...workers.map((n) => ({ ...n, bucket: 'worker' as const, x: 0, y: 0 })),
    ...clients.map((n) => ({ ...n, bucket: 'client' as const, x: 0, y: 0 })),
  ];

  const positioned: PositionedTopologyNode[] = [{ ...center, x: 0, y: 0, bucket: 'center' }];
  const peerCount = all.length;
  if (peerCount === 0) return positioned;

  if (peerCount <= 6) {
    const horizontalSpacing = 270;
    const bandOffset = 215;
    const rowStep = 118;
    const topRows: Array<{ nodes: TopologyNode[]; bucket: PositionedTopologyNode['bucket'] }> = [];
    const bottomRows: Array<{ nodes: TopologyNode[]; bucket: PositionedTopologyNode['bucket'] }> = [];

    if (serving.length) {
      topRows.push({ nodes: serving, bucket: 'serving' });
    }

    if (workers.length) {
      if (serving.length === 0) {
        topRows.push({ nodes: workers, bucket: 'worker' });
      } else {
        bottomRows.push({ nodes: workers, bucket: 'worker' });
      }
    }

    if (clients.length) {
      bottomRows.push({ nodes: clients, bucket: 'client' });
    }

    topRows.forEach((row, index) => {
      const distanceFromCenter = bandOffset + ((topRows.length - index - 1) * rowStep);
      placeRow(row.nodes, row.bucket, -distanceFromCenter, horizontalSpacing, positioned);
    });

    bottomRows.forEach((row, index) => {
      const distanceFromCenter = bandOffset + (index * rowStep);
      placeRow(row.nodes, row.bucket, distanceFromCenter, horizontalSpacing, positioned);
    });

    return positioned;
  }

  // Small meshes get a larger first ring so the graph uses the available canvas.
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
    for (let i = 0; i < peerCount; i += 1) {
      const angle = -Math.PI / 2 + ((2 * Math.PI * i) / peerCount);
      const node = all[i];
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
    for (let i = 0; i < take; i += 1) {
      const angle = -Math.PI / 2 + phase + ((2 * Math.PI * i) / take);
      const node = all[offset + i];
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

// KaTeX math renderer — loads from CDN on first use
let katexCssLoaded = false;
const katexPromise = import('https://cdn.jsdelivr.net/npm/katex@0.16/dist/katex.mjs' as string).then(m => {
  if (!katexCssLoaded) {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://cdn.jsdelivr.net/npm/katex@0.16/dist/katex.min.css';
    document.head.appendChild(link);
    katexCssLoaded = true;
  }
  return m.default;
}).catch(() => null);

function KaTeXBlock({ math, display }: { math: string; display: boolean }) {
  const [html, setHtml] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    katexPromise.then((katex) => {
      if (cancelled || !katex) return;
      try {
        const rendered = katex.renderToString(math, { displayMode: display, throwOnError: false });
        if (!cancelled) setHtml(rendered);
      } catch {
        if (!cancelled) setHtml(null);
      }
    });
    return () => { cancelled = true; };
  }, [math, display]);

  if (html === null) return display ? <div className="my-2 overflow-x-auto text-sm"><code>{math}</code></div> : <code>{math}</code>;
  return display
    ? <div className="my-2 overflow-x-auto" dangerouslySetInnerHTML={{ __html: html }} />
    : <span dangerouslySetInnerHTML={{ __html: html }} />;
}

// Mermaid diagram renderer — loads mermaid from CDN on first use
const mermaidPromise = import('https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs' as string).then(m => {
  m.default.initialize({ startOnLoad: false, theme: 'dark', securityLevel: 'loose' });
  return m.default;
}).catch(() => null);

function MermaidBlock({ code }: { code: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    mermaidPromise.then(async (mermaid) => {
      if (cancelled || !mermaid) { setError('Mermaid failed to load'); return; }
      try {
        const id = `mermaid-${Date.now()}-${Math.random().toString(36).slice(2)}`;
        const { svg: rendered } = await mermaid.render(id, code);
        if (!cancelled) setSvg(rendered);
      } catch (e: unknown) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Render failed');
      }
    });
    return () => { cancelled = true; };
  }, [code]);

  if (error) return <pre className="my-2 rounded-lg border border-border/70 bg-background/80 p-3 text-xs text-muted-foreground"><code>{code}</code></pre>;
  if (!svg) return <div className="my-2 flex items-center gap-2 rounded-lg border border-border/70 bg-background/80 p-3 text-xs text-muted-foreground"><Loader2 className="h-3 w-3 animate-spin" />Rendering diagram…</div>;
  return <div ref={containerRef} className="my-2 overflow-x-auto rounded-lg border border-border/70 bg-background/80 p-3 [&_svg]:max-w-full" dangerouslySetInnerHTML={{ __html: svg }} />;
}

function MarkdownMessage({ content, streaming }: { content: string; streaming?: boolean }) {
  return (
    <div
      className={cn(
        'break-words text-sm leading-6',
        '[&_a]:underline [&_a]:underline-offset-2',
        '[&_blockquote]:my-2 [&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-3 [&_blockquote]:italic',
        '[&_code]:rounded [&_code]:bg-background/70 [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[0.9em]',
        '[&_h1]:mb-2 [&_h1]:mt-3 [&_h1]:text-base [&_h1]:font-semibold [&_h1:first-child]:mt-0',
        '[&_h2]:mb-2 [&_h2]:mt-3 [&_h2]:text-sm [&_h2]:font-semibold [&_h2:first-child]:mt-0',
        '[&_hr]:my-3 [&_hr]:border-border',
        '[&_li]:my-0.5',
        '[&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5',
        '[&_p]:my-2 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0',
        '[&_pre]:my-2 [&_pre]:max-w-full [&_pre]:overflow-x-auto [&_pre]:whitespace-pre [&_pre]:rounded-lg [&_pre]:border [&_pre]:border-border/70 [&_pre]:bg-background/80 [&_pre]:p-3',
        '[&_pre_code]:bg-transparent [&_pre_code]:p-0',
        '[&_table]:my-2 [&_table]:w-full [&_table]:border-collapse [&_table]:text-xs [&_table]:block [&_table]:overflow-x-auto',
        '[&_td]:border [&_td]:border-border/60 [&_td]:px-2 [&_td]:py-1',
        '[&_th]:border [&_th]:border-border/60 [&_th]:bg-muted/40 [&_th]:px-2 [&_th]:py-1 [&_th]:text-left',
        '[&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5',
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          code({ className, children, ...props }) {
            const text = String(children).replace(/\n$/, '');
            if (!streaming) {
              if (/language-mermaid/.test(className || '')) return <MermaidBlock code={text} />;
              if (/language-math/.test(className || '')) return <KaTeXBlock math={text} display={/math-display/.test(className || '')} />;
            }
            return <code className={className} {...props}>{children}</code>;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function ChatBubble({
  message,
  reasoningOpen,
  onReasoningToggle,
  streaming,
}: {
  message: ChatMessage;
  reasoningOpen: boolean;
  onReasoningToggle: (open: boolean) => void;
  streaming?: boolean;
}) {
  const isUser = message.role === 'user';
  const isThinking = !isUser && message.reasoning && !message.content;
  const hasFinishedThinking = !isUser && message.reasoning && !!message.content;

  return (
    <div className={cn('flex', isUser ? 'justify-end' : 'justify-start')}>
      <div className="w-full min-w-0 max-w-[92%] md:max-w-[82%]">
        <div className="mb-1 flex items-center gap-2 px-1 text-xs text-muted-foreground">
          {isUser ? <User className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5" />}
          <span>{isUser ? 'You' : 'Assistant'}</span>
          {message.model ? <span>· {shortName(message.model)}</span> : null}
        </div>

        {/* Thinking indicator — click to expand and watch reasoning stream live */}
        {isThinking ? (
          <div className="mb-2 rounded-lg border border-dashed">
            <button
              type="button"
              className="flex w-full items-center gap-2 px-3 py-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => onReasoningToggle(!reasoningOpen)}
            >
              <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0" />
              <span>Thinking…</span>
              <ChevronDown className={cn('ml-auto h-3 w-3 transition-transform', reasoningOpen ? '' : '-rotate-90')} />
            </button>
            {reasoningOpen && message.reasoning ? (
              <div className="border-t border-dashed px-3 pb-2 pt-1">
                <ScrollArea className="max-h-60">
                  <div className="whitespace-pre-wrap text-xs leading-5 text-muted-foreground">{message.reasoning}</div>
                </ScrollArea>
              </div>
            ) : null}
          </div>
        ) : null}

        {/* Collapsed reasoning accordion — shown after thinking is done */}
        {hasFinishedThinking ? (
          <Accordion
            type="single"
            collapsible
            value={reasoningOpen ? 'reasoning' : ''}
            onValueChange={(v) => onReasoningToggle(v === 'reasoning')}
            className="mb-2"
          >
            <AccordionItem value="reasoning" className="rounded-lg border border-dashed px-3">
              <AccordionTrigger className="py-2 text-xs text-muted-foreground hover:no-underline">
                <span className="flex items-center gap-1.5">
                  <Sparkles className="h-3 w-3" />
                  Thought for a moment
                </span>
              </AccordionTrigger>
              <AccordionContent>
                <div className="whitespace-pre-wrap text-xs leading-5 text-muted-foreground">{message.reasoning}</div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        ) : null}

        {/* Attached image */}
        {isUser && message.image ? (
          <div className="mb-2">
            <img src={message.image} alt="Attached" className="max-h-48 rounded-lg border object-contain" />
          </div>
        ) : null}

        {/* Main content */}
        {isUser || message.content ? (
          <div
            className={cn(
              'rounded-lg border px-4 py-3 text-sm leading-6 break-words',
              isUser
                ? 'bg-muted whitespace-pre-wrap'
                : message.error
                  ? 'border-destructive/50 text-destructive'
                  : 'bg-background',
            )}
          >
            {message.content ? <MarkdownMessage content={message.content} streaming={streaming} /> : !isUser ? '...' : ''}
          </div>
        ) : null}

        {message.stats ? <div className="mt-1 px-1 text-xs text-muted-foreground">{message.stats}</div> : null}
      </div>
    </div>
  );
}

function StatCard({
  title,
  value,
  valueSuffix,
  icon,
  tooltip,
}: {
  title: string;
  value: string;
  valueSuffix?: ReactNode;
  icon: ReactNode;
  tooltip?: string;
}) {
  const card = (
    <Card>
      <CardContent className="p-3">
        <div className="mb-2 flex items-center gap-2 text-muted-foreground">{icon}<span className="text-xs">{title}</span></div>
        <div className="flex min-w-0 items-center gap-2 text-sm font-semibold text-foreground">
          <span className="truncate">{value}</span>
          {valueSuffix}
        </div>
      </CardContent>
    </Card>
  );
  if (!tooltip) return card;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div>{card}</div>
      </TooltipTrigger>
      <TooltipContent side="bottom" align="center" sideOffset={8}>
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}

function EmptyPanel({ text }: { text: string }) {
  return (
    <Card>
      <CardContent className="p-4 text-sm text-muted-foreground">{text}</CardContent>
    </Card>
  );
}

function DashboardPanelEmpty({
  icon,
  title,
  description,
}: {
  icon: ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="flex h-full min-h-[18rem] flex-col items-center justify-center rounded-md border border-dashed bg-muted/20 px-4 text-center md:min-h-[20rem]">
      <div className="mb-2 flex h-8 w-8 items-center justify-center rounded-full border bg-background text-muted-foreground">
        {icon}
      </div>
      <div className="text-sm font-medium">{title}</div>
      <div className="mt-1 max-w-md text-xs text-muted-foreground">{description}</div>
    </div>
  );
}

function meshGpuVram(status: StatusPayload | null) {
  if (!status) return 0;
  return (status.is_client ? 0 : status.my_vram_gb || 0) + (status.peers || []).filter((p) => p.role !== 'Client').reduce((s, p) => s + p.vram_gb, 0);
}

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
