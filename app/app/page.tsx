'use client'

import { useState, useRef, useEffect, Fragment } from 'react'
import { Send, Loader2, Sparkles, BookOpen, ChevronDown, ChevronUp, Copy, Check, Plus, Trash2 } from 'lucide-react'
import { useChatHistory, type Chat, type StoredMessage } from '../hooks/useChatHistory'

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

// Message is the same shape as StoredMessage from the hook
type Message = StoredMessage

// ─── Relative time helper ─────────────────────────────────────────────────────
function relativeTime(ts: number): string {
  const diff = Date.now() - ts
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'agora'
  if (mins < 60) return `há ${mins} min`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `há ${hours}h`
  const days = Math.floor(hours / 24)
  if (days === 1) return 'ontem'
  if (days < 7) return `há ${days} dias`
  return new Date(ts).toLocaleDateString('pt-BR', { day: '2-digit', month: '2-digit' })
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

  async function copy() {
    try {
      await navigator.clipboard.writeText(code.trim())
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    } catch {
      // clipboard write failed silently — don't show "copiado"
    }
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

// ─── Chat item (sidebar) ──────────────────────────────────────────────────────
function ChatItem({
  chat, isActive, onSelect, onRename, onDelete,
}: {
  chat: Chat
  isActive: boolean
  onSelect: () => void
  onRename: (title: string) => void
  onDelete: () => void
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(chat.title)
  const inputRef = useRef<HTMLInputElement>(null)

  function startEdit(e: React.MouseEvent) {
    e.stopPropagation()
    setDraft(chat.title)
    setEditing(true)
    setTimeout(() => inputRef.current?.focus(), 0)
  }

  function commitEdit() {
    const trimmed = draft.trim()
    if (trimmed) onRename(trimmed)
    setEditing(false)
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter') commitEdit()
    if (e.key === 'Escape') { setEditing(false); setDraft(chat.title) }
  }

  return (
    <div
      className={`chat-item${isActive ? ' active' : ''}`}
      onClick={onSelect}
    >
      <div className="chat-item-body">
        {editing ? (
          <input
            ref={inputRef}
            className="chat-item-input"
            value={draft}
            onChange={e => setDraft(e.target.value)}
            onBlur={commitEdit}
            onKeyDown={handleKeyDown}
            onClick={e => e.stopPropagation()}
          />
        ) : (
          <span className="chat-item-title" onDoubleClick={startEdit}>
            {chat.title}
          </span>
        )}
        <span className="chat-item-meta">{relativeTime(chat.updatedAt)}</span>
      </div>
      <button
        className="chat-item-delete"
        onClick={e => { e.stopPropagation(); onDelete() }}
        aria-label="Deletar conversa"
      >
        <Trash2 size={13} />
      </button>
    </div>
  )
}

// ─── Sidebar ──────────────────────────────────────────────────────────────────
function Sidebar({
  chats, activeChatId, onNew, onSelect, onRename, onDelete,
}: {
  chats: Chat[]
  activeChatId: string | null
  onNew: () => void
  onSelect: (id: string) => void
  onRename: (id: string, title: string) => void
  onDelete: (id: string) => void
}) {
  const sorted = [...chats].sort((a, b) => b.updatedAt - a.updatedAt)
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <button className="new-chat-btn" onClick={onNew}>
          <Plus size={14} />
          Novo chat
        </button>
      </div>
      <div className="chat-list">
        {sorted.map(chat => (
          <ChatItem
            key={chat.id}
            chat={chat}
            isActive={chat.id === activeChatId}
            onSelect={() => onSelect(chat.id)}
            onRename={title => onRename(chat.id, title)}
            onDelete={() => onDelete(chat.id)}
          />
        ))}
      </div>
    </aside>
  )
}

// ─── Assistant message ────────────────────────────────────────────────────────
function AssistantMsg({ msg }: { msg: Message }) {
  const [open, setOpen] = useState(false)
  const libs = Array.from(new Set(msg.citations?.map(c => (c as Citation).library).filter(Boolean) ?? [])) as string[]
  const primary = libs[0]
  const lc = primary ? LIB_COLORS[primary] : null

  return (
    <div className="msg-ai-row">
      <div className="msg-ai-label">
        <DogMascot size={18} className="msg-dog" />
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
                  const citation = c as Citation
                  const cl = citation.library ? LIB_COLORS[citation.library] : null
                  return (
                    <span
                      key={i}
                      className="cit-chip"
                      style={cl
                        ? { '--lib-accent': cl.accent, '--lib-text': cl.text } as React.CSSProperties
                        : {}}
                    >
                      {citation.library && <span className="cit-lib">{citation.library}</span>}
                      {citation.section && <span className="cit-sec">{citation.section}</span>}
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
  const {
    chats, activeChat, activeChatId,
    newChat, selectChat, saveMessage, setTitle, deleteChat,
  } = useChatHistory()

  const messages: Message[] = activeChat?.messages ?? []

  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
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

    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', content: q }
    saveMessage(userMsg)
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

      saveMessage({
        id: crypto.randomUUID(),
        role: 'assistant',
        content: data.answer ?? 'Sem resposta.',
        citations: data.citations ?? [],
      })
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      saveMessage({
        id: crypto.randomUUID(),
        role: 'assistant',
        content: `Erro ao conectar com o backend.\n\n\`\`\`\n${msg}\n\`\`\`\n\nVerifique se o HF Space está no ar: ${BACKEND_URL}`,
      })
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
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        onNew={newChat}
        onSelect={selectChat}
        onRename={setTitle}
        onDelete={deleteChat}
      />

      <div className="main-area">
        {/* ── Header ── */}
        <header className="app-header">
          <div className="header-brand">
            <DogMascot size={28} className="header-dog" />
            <span className="logo">data<span className="logo-dot">.</span></span>
            <span className="logo-tag">assistente rag PT‑BR</span>
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
    </div>
  )
}
