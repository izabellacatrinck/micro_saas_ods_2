'use client'

import { useState, useRef, useEffect, Fragment } from 'react'
import { Send, Loader2, Sparkles, BookOpen, ChevronDown, ChevronUp, Copy, Check } from 'lucide-react'

// ─── Dog mascot SVG ───────────────────────────────────────────────────────────
function DogMascot({ size = 22, className = '' }: { size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      {/* floppy ears */}
      <ellipse cx="8.5"  cy="11" rx="4.5" ry="6" fill="#c87820" transform="rotate(-12 8.5 11)" />
      <ellipse cx="23.5" cy="11" rx="4.5" ry="6" fill="#c87820" transform="rotate(12 23.5 11)" />
      {/* head */}
      <circle cx="16" cy="16" r="9" fill="#f0a030" />
      {/* snout */}
      <ellipse cx="16" cy="19.5" rx="4.5" ry="3" fill="#e8932a" />
      {/* nose */}
      <ellipse cx="16" cy="17.5" rx="2.2" ry="1.5" fill="#6b3800" />
      {/* nostrils */}
      <circle cx="14.8" cy="17.6" r="0.55" fill="#3a1e00" />
      <circle cx="17.2" cy="17.6" r="0.55" fill="#3a1e00" />
      {/* mouth */}
      <path d="M13.5 20.5 Q16 22.2 18.5 20.5" stroke="#6b3800" strokeWidth="1.1" fill="none" strokeLinecap="round" />
      {/* eyes */}
      <circle cx="11.5" cy="14" r="2.2" fill="#1a0f00" />
      <circle cx="20.5" cy="14" r="2.2" fill="#1a0f00" />
      {/* eye shine */}
      <circle cx="12.2" cy="13.2" r="0.75" fill="white" />
      <circle cx="21.2" cy="13.2" r="0.75" fill="white" />
      {/* eyebrow arcs (expressive) */}
      <path d="M9.5 11.5 Q11.5 10.2 13.5 11.5" stroke="#8a5c18" strokeWidth="1" fill="none" strokeLinecap="round" />
      <path d="M18.5 11.5 Q20.5 10.2 22.5 11.5" stroke="#8a5c18" strokeWidth="1" fill="none" strokeLinecap="round" />
    </svg>
  )
}

// ─── Config ───────────────────────────────────────────────────────────────────
const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, '') ?? 'http://127.0.0.1:7860'

const LIB_COLORS: Record<string, { accent: string; text: string; bg: string }> = {
  pandas:     { accent: '#f0a030', text: '#f5b84a', bg: 'rgba(240,160,48,0.1)' },
  numpy:      { accent: '#2dd4bf', text: '#4de8d4', bg: 'rgba(45,212,191,0.08)' },
  matplotlib: { accent: '#f87171', text: '#fca5a5', bg: 'rgba(248,113,113,0.08)' },
  seaborn:    { accent: '#a78bfa', text: '#c4b5fd', bg: 'rgba(167,139,250,0.08)' },
}

const QUICK_PROMPTS = [
  { tag: 'merge & join',   text: 'Explique a diferença entre merge, join e concat no pandas.' },
  { tag: 'broadcasting',  text: 'Como funciona broadcasting no NumPy? Mostre um exemplo prático.' },
  { tag: 'groupby',        text: 'Como usar groupby com múltiplas funções de agregação no pandas?' },
  { tag: 'matplotlib',     text: 'Como personalizar eixos, título e legenda em gráficos Matplotlib?' },
]

// ─── Types ────────────────────────────────────────────────────────────────────
type Citation = {
  library?: string
  section?: string
  source?: string
}

type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  outIndex?: number
}

// ─── Content renderer (bold + code blocks) ───────────────────────────────────
function MsgContent({ text }: { text: string }) {
  const parts = text.split(/(```[\w]*\n[\s\S]*?```)/g)

  return (
    <div className="msg-ai-body">
      {parts.map((part, i) => {
        if (part.startsWith('```')) {
          const lines = part.split('\n')
          const lang = lines[0].replace(/^```/, '').trim() || 'python'
          const code = lines.slice(1).join('\n').replace(/```$/, '')
          return <CodeBlock key={i} lang={lang} code={code} />
        }

        // Handle **bold** and `inline code`
        const segments = part.split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
        return (
          <Fragment key={i}>
            {segments.map((seg, j) => {
              if (seg.startsWith('**') && seg.endsWith('**'))
                return <strong key={j}>{seg.slice(2, -2)}</strong>
              if (seg.startsWith('`') && seg.endsWith('`'))
                return <code key={j}>{seg.slice(1, -1)}</code>
              return <span key={j}>{seg}</span>
            })}
          </Fragment>
        )
      })}
    </div>
  )
}

function CodeBlock({ lang, code }: { lang: string; code: string }) {
  const [copied, setCopied] = useState(false)

  function copy() {
    navigator.clipboard.writeText(code.trim())
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }

  return (
    <div className="code-block">
      <div className="code-block-header">
        <span className="code-lang">{lang}</span>
        <button className="copy-btn" onClick={copy}>
          {copied ? <Check size={10} style={{ marginRight: 3, display: 'inline' }} /> : <Copy size={10} style={{ marginRight: 3, display: 'inline' }} />}
          {copied ? 'copiado' : 'copiar'}
        </button>
      </div>
      <pre><code>{code.trim()}</code></pre>
    </div>
  )
}

// ─── Assistant message ────────────────────────────────────────────────────────
function AssistantMsg({ msg }: { msg: Message }) {
  const [open, setOpen] = useState(false)
  const libs = [...new Set(msg.citations?.map(c => c.library).filter(Boolean))] as string[]
  const primary = libs[0]
  const lc = primary ? LIB_COLORS[primary] : null

  return (
    <div className="msg-ai-row">
      <div className="msg-ai-label">
        <DogMascot size={18} className="msg-dog" />
        <span
          className="out-label"
          style={lc ? { '--lib-text': lc.text } as React.CSSProperties : {}}
        >
          Out[{msg.outIndex}]
        </span>
      </div>
      <div
        className="msg-ai"
        style={lc ? { '--lib-border': lc.accent } as React.CSSProperties : {}}
      >
        <MsgContent text={msg.content} />

        {(msg.citations?.length ?? 0) > 0 && (
          <div className="citations">
            <button className="cit-toggle" onClick={() => setOpen(o => !o)}>
              <BookOpen size={11} />
              <span>{msg.citations!.length} fonte{msg.citations!.length !== 1 ? 's' : ''}</span>
              {open ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
            </button>

            {open && (
              <div className="cit-list">
                {msg.citations!.map((c, i) => {
                  const cl = c.library ? LIB_COLORS[c.library] : null
                  return (
                    <span
                      key={i}
                      className="cit-chip"
                      style={cl
                        ? { '--lib-accent': cl.accent, '--lib-text': cl.text } as React.CSSProperties
                        : {}}
                    >
                      {c.library && <span className="cit-lib">{c.library}</span>}
                      {c.section && <span className="cit-sec">{c.section}</span>}
                    </span>
                  )
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Typing indicator ─────────────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div className="typing-wrap">
      <div className="msg-ai-label">
        <DogMascot size={18} className="msg-dog" />
        <span className="out-label" style={{ color: 'var(--text-dim)' }}>…</span>
      </div>
      <div className="typing-dots">
        <span /><span /><span />
      </div>
    </div>
  )
}

// ─── Cold-start banner ────────────────────────────────────────────────────────
function WakingBanner() {
  return (
    <div className="waking-banner">
      <Loader2 size={12} className="spin" />
      <span>O servidor está acordando (HF Space em modo gratuito). Pode levar ~30s…</span>
    </div>
  )
}

// ─── Main page ────────────────────────────────────────────────────────────────
export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [outIdx, setOutIdx] = useState(1)
  const [serverStatus, setServerStatus] = useState<'unknown' | 'waking' | 'ready'>('unknown')

  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  // Warm-up: hit /health on mount so the HF Space starts loading models immediately
  useEffect(() => {
    let cancelled = false
    async function warmUp() {
      try {
        const r = await fetch(`${BACKEND_URL}/health`, { method: 'GET' })
        if (cancelled) return
        if (r.ok) {
          const d = await r.json()
          setServerStatus(d.models_loaded ? 'ready' : 'waking')
        } else {
          setServerStatus('waking')
        }
      } catch {
        if (!cancelled) setServerStatus('waking')
      }
    }
    warmUp()
    return () => { cancelled = true }
  }, [])

  // auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 140) + 'px'
  }, [input])

  async function send(question: string) {
    const q = question.trim()
    if (!q || loading) return

    setMessages(prev => [...prev, { id: Date.now().toString(), role: 'user', content: q }])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch(`${BACKEND_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`)
      const data = await res.json()
      setServerStatus('ready')

      const idx = outIdx
      setOutIdx(n => n + 1)
      setMessages(prev => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: data.answer ?? 'Sem resposta.',
          citations: data.citations ?? [],
          outIndex: idx,
        },
      ])
    } catch (err: any) {
      const idx = outIdx
      setOutIdx(n => n + 1)
      setMessages(prev => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `Erro ao conectar com o backend.\n\n\`\`\`\n${err?.message ?? err}\n\`\`\`\n\nVerifique se o HF Space está no ar: ${BACKEND_URL}`,
          outIndex: idx,
        },
      ])
    } finally {
      setLoading(false)
      textareaRef.current?.focus()
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    send(input)
  }

  function handleKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send(input)
    }
  }

  const isEmpty = messages.length === 0 && !loading

  return (
    <div className="app-shell">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-brand">
          <DogMascot size={28} className="header-dog" />
          <span className="logo">data<span className="logo-dot">.</span></span>
          <span className="logo-tag">assistente PT‑BR</span>
        </div>

        <div className="header-libs">
          {Object.entries(LIB_COLORS).map(([lib, { accent, text, bg }]) => (
            <span
              key={lib}
              className="lib-pill"
              style={{ '--lib-accent': accent, '--lib-text': text, '--lib-bg': bg } as React.CSSProperties}
            >
              {lib}
            </span>
          ))}
        </div>
      </header>

      {/* ── Chat ── */}
      <main className="chat-main">
        {serverStatus === 'waking' && <WakingBanner />}
        <div className="chat-inner">
          {isEmpty ? (
            <div className="empty-state">
              <div className="empty-orb">
                <Sparkles size={24} />
              </div>
              <h2 className="empty-title">O que você quer aprender hoje?</h2>
              <p className="empty-sub">
                Pergunte sobre pandas, NumPy, Matplotlib ou Seaborn em português.
                As respostas são geradas com base na documentação oficial.
              </p>
              <div className="quick-grid">
                {QUICK_PROMPTS.map(q => (
                  <button key={q.tag} className="quick-btn" onClick={() => send(q.text)}>
                    <span className="quick-tag">{q.tag}</span>
                    <span className="quick-text">{q.text}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages">
              {messages.map(msg =>
                msg.role === 'user' ? (
                  <div key={msg.id} className="msg-user-row">
                    <div className="msg-user">{msg.content}</div>
                  </div>
                ) : (
                  <AssistantMsg key={msg.id} msg={msg} />
                )
              )}
              {loading && <TypingIndicator />}
              <div ref={bottomRef} />
            </div>
          )}
        </div>
      </main>

      {/* ── Input ── */}
      <footer className="input-bar">
        <form className="input-inner" onSubmit={handleSubmit}>
          <div className="input-box">
            <textarea
              ref={textareaRef}
              className="input-ta"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Faça uma pergunta sobre análise de dados em Python…"
              rows={1}
              disabled={loading}
            />
            <button
              type="submit"
              className="send-btn"
              disabled={loading || !input.trim()}
              aria-label="Enviar"
            >
              {loading
                ? <Loader2 size={15} className="spin" />
                : <Send size={15} />}
            </button>
          </div>
          <p className="input-hint">Enter para enviar · Shift+Enter para nova linha</p>
        </form>
      </footer>
    </div>
  )
}
