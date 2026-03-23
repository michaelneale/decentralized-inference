import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { Loader2 } from 'lucide-react';

import { cn } from '../lib/utils';

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

export default function ChatMarkdownMessage({ content, streaming }: { content: string; streaming?: boolean }) {
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
