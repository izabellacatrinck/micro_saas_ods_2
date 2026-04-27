# Chat Persistence & Sidebar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 240px fixed sidebar with chat history to the RAG frontend, persisted in `localStorage`, using a `useChatHistory` hook.

**Architecture:** A `useChatHistory` hook owns all chat state (load/save `localStorage`, CRUD actions). `page.tsx` calls the hook, derives `messages` from the active chat, and renders a `<Sidebar>` component alongside the existing chat area. `app-shell` becomes a flex-row; the existing header/chat/input is wrapped in `.main-area`.

**Tech Stack:** Next.js 13, TypeScript, CSS custom properties (no new dependencies)

---

## File Map

| File | Change |
|---|---|
| `app/hooks/useChatHistory.ts` | **Create** — hook + types |
| `app/app/page.tsx` | **Modify** — import hook, add Sidebar+ChatItem components, wire send(), wrap layout |
| `app/app/globals.css` | **Modify** — app-shell → flex-row, add sidebar/main-area styles |

---

### Task 1: `useChatHistory` hook

**Files:**
- Create: `app/hooks/useChatHistory.ts`

- [ ] **Step 1: Create the hooks directory and file**

```bash
mkdir -p app/hooks
```

- [ ] **Step 2: Write the hook**

Create `app/hooks/useChatHistory.ts` with this exact content:

```typescript
'use client'

import { useState, useEffect, useRef } from 'react'

// ─── Types ────────────────────────────────────────────────────────────────────
export type StoredMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Array<{ library?: string; section?: string; source?: string }>
}

export type Chat = {
  id: string
  title: string
  messages: StoredMessage[]
  createdAt: number
  updatedAt: number
}

const STORAGE_KEY = 'rag_chats'

// ─── Hook ─────────────────────────────────────────────────────────────────────
export function useChatHistory() {
  const [chats, setChats] = useState<Chat[]>([])
  const [activeChatId, setActiveChatId] = useState<string | null>(null)
  // refs so async callbacks (fetch responses) always read current values
  const activeChatIdRef = useRef<string | null>(null)
  const persistReady = useRef(false)

  // Load from localStorage on mount
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (raw) setChats(JSON.parse(raw) as Chat[])
    } catch {
      // corrupted data — start fresh
    }
    persistReady.current = true
  }, [])

  // Persist on every chats change, but only after initial load
  useEffect(() => {
    if (!persistReady.current) return
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(chats))
    } catch {
      // quota exceeded — in-session state still works
    }
  }, [chats])

  // Derived: the active Chat object (null when in "new chat" mode)
  const activeChat = chats.find(c => c.id === activeChatId) ?? null

  // ─── Actions ────────────────────────────────────────────────────────────────
  function _setActive(id: string | null) {
    activeChatIdRef.current = id
    setActiveChatId(id)
  }

  /** Clear the active chat — next message creates a new one */
  function newChat() {
    _setActive(null)
  }

  /** Load a previous chat into view */
  function selectChat(id: string) {
    _setActive(id)
  }

  /**
   * Append a message to the active chat.
   * If no active chat and msg is from the user, creates a new Chat automatically.
   * Uses activeChatIdRef so it is safe to call from async callbacks.
   */
  function saveMessage(msg: StoredMessage) {
    const cid = activeChatIdRef.current
    if (cid) {
      setChats(prev =>
        prev.map(c =>
          c.id === cid
            ? { ...c, messages: [...c.messages, msg], updatedAt: Date.now() }
            : c
        )
      )
    } else {
      // Only user messages can start a new chat
      if (msg.role !== 'user') return
      const newId = crypto.randomUUID()
      _setActive(newId)
      const newChatObj: Chat = {
        id: newId,
        title: msg.content.slice(0, 52),
        messages: [msg],
        createdAt: Date.now(),
        updatedAt: Date.now(),
      }
      setChats(prev => [newChatObj, ...prev])
    }
  }

  /** Rename a chat. Ignores empty strings (ChatItem reverts the input). */
  function setTitle(id: string, title: string) {
    const trimmed = title.trim()
    if (!trimmed) return
    setChats(prev => prev.map(c => c.id === id ? { ...c, title: trimmed } : c))
  }

  /** Delete a chat. If it was active, resets to new-chat mode. */
  function deleteChat(id: string) {
    setChats(prev => prev.filter(c => c.id !== id))
    if (activeChatIdRef.current === id) _setActive(null)
  }

  return {
    chats,
    activeChat,
    activeChatId,
    newChat,
    selectChat,
    saveMessage,
    setTitle,
    deleteChat,
  }
}
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd app && npx tsc --noEmit
```

Expected: no errors (or only pre-existing errors unrelated to this file).

- [ ] **Step 4: Commit**

```bash
git add app/hooks/useChatHistory.ts
git commit -m "feat(chat): add useChatHistory hook with localStorage persistence"
```

---

### Task 2: Wire hook + add Sidebar/ChatItem to page.tsx

**Files:**
- Modify: `app/app/page.tsx`

- [ ] **Step 1: Update imports at the top of `app/app/page.tsx`**

Replace the existing import lines (lines 1–4) with:

```typescript
'use client'

import { useState, useRef, useEffect, Fragment } from 'react'
import { Send, Loader2, Sparkles, BookOpen, ChevronDown, ChevronUp, Copy, Check, Plus, Trash2 } from 'lucide-react'
import { useChatHistory, type Chat, type StoredMessage } from '../hooks/useChatHistory'
```

- [ ] **Step 2: Remove the `Message` type and add a `relativeTime` helper**

Delete this block (around line 70–76):
```typescript
type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
}
```

Add right after the `Citation` type:
```typescript
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
```

- [ ] **Step 3: Add `ChatItem` component before `AssistantMsg`**

Insert this block before the `// ─── Assistant message ───` comment:

```typescript
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
```

- [ ] **Step 4: Update the `Home` component**

Replace the entire `export default function Home()` function with:

```typescript
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

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: q }
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
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer ?? 'Sem resposta.',
        citations: data.citations ?? [],
      })
    } catch (err: any) {
      saveMessage({
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Erro ao conectar com o backend.\n\n\`\`\`\n${err?.message ?? err}\n\`\`\`\n\nVerifique se o HF Space está no ar: ${BACKEND_URL}`,
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
```

- [ ] **Step 5: Verify TypeScript compiles**

```bash
cd app && npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add app/app/page.tsx
git commit -m "feat(chat): add Sidebar and ChatItem components, wire useChatHistory"
```

---

### Task 3: CSS — sidebar layout

**Files:**
- Modify: `app/app/globals.css`

- [ ] **Step 1: Update `.app-shell` (lines 62–70)**

Replace the existing `.app-shell` block:

```css
/* ─── App shell ─── */
.app-shell {
  display: flex;
  flex-direction: row;
  height: 100vh;
  max-width: 1200px;
  margin: 0 auto;
  border-left: 1px solid var(--border-dim);
  border-right: 1px solid var(--border-dim);
  overflow: hidden;
}
```

- [ ] **Step 2: Add `.main-area` right after `.app-shell`**

```css
/* ─── Main area (header + chat + input) ─── */
.main-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  overflow: hidden;
}
```

- [ ] **Step 3: Add sidebar styles at the end of `globals.css`**

Append to the end of the file (after the `.spin` rule):

```css
/* ─── Sidebar ─── */
.sidebar {
  width: 240px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  background: var(--bg);
  border-right: 1px solid var(--border);
  overflow: hidden;
}

.sidebar-header {
  padding: 10px;
  border-bottom: 1px solid var(--border-dim);
  flex-shrink: 0;
}

.new-chat-btn {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 8px 12px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  color: var(--text-soft);
  font-family: var(--font-ui);
  font-size: 12.5px;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s, color 0.15s;
}
.new-chat-btn:hover {
  background: color-mix(in srgb, var(--amber) 8%, var(--panel));
  border-color: var(--amber-dim);
  color: var(--text);
}

.chat-list {
  flex: 1;
  overflow-y: auto;
  padding: 6px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.chat-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 10px;
  border-radius: var(--radius-sm);
  border: 1px solid transparent;
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s;
  min-width: 0;
}
.chat-item:hover {
  background: var(--panel);
  border-color: var(--border-dim);
}
.chat-item.active {
  background: var(--panel-soft);
  border-color: var(--border-dim);
  border-left: 2px solid var(--amber);
  padding-left: 8px;
}

.chat-item-body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
  overflow: hidden;
}

.chat-item-title {
  font-size: 12px;
  color: var(--text-soft);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.4;
}
.chat-item.active .chat-item-title { color: var(--text); }

.chat-item-input {
  font-size: 12px;
  color: var(--text);
  background: var(--panel-soft);
  border: 1px solid var(--amber-dim);
  border-radius: 3px;
  padding: 1px 4px;
  width: 100%;
  font-family: var(--font-ui);
  outline: none;
}

.chat-item-meta {
  font-size: 10px;
  color: var(--text-dim);
  line-height: 1;
}

.chat-item-delete {
  flex-shrink: 0;
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  padding: 2px;
  border-radius: 3px;
  opacity: 0;
  transition: opacity 0.12s, color 0.12s;
  display: flex;
  align-items: center;
}
.chat-item:hover .chat-item-delete { opacity: 1; }
.chat-item-delete:hover { color: #f87171; }
```

- [ ] **Step 4: Verify the build compiles locally**

```bash
cd app && npm run build
```

Expected: `✓ Compiled successfully` with no TypeScript errors.

- [ ] **Step 5: Manual smoke test**

Open `http://localhost:3000` (run `npm run dev` in `app/`):

1. Page loads with sidebar on the left showing empty state (no chats yet)
2. Ask a question → chat appears in sidebar with title = first 52 chars of question
3. Click "Novo chat" → empty state appears, sidebar still shows previous chat
4. Ask another question → new chat appears at top of sidebar
5. Click a previous chat → its messages load in the main area
6. Double-click a chat title → input appears; type new name + Enter → title updates
7. Hover a chat item → red trash icon appears; click it → chat removed
8. Reload the page → all chats are still there (persisted in localStorage)
9. Delete all chats → empty sidebar, empty state in main area

- [ ] **Step 6: Commit and push**

```bash
git add app/app/globals.css
git commit -m "feat(chat): sidebar layout and styles"
git push origin main
```
