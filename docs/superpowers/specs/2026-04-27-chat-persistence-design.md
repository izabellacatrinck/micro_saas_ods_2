# Chat Persistence & Sidebar Design

## Goal

Add a persistent chat history sidebar to the RAG PT-BR frontend, letting users revisit, rename, and delete past conversations — all stored in `localStorage`, no backend changes required.

## Architecture

**Storage:** `localStorage` key `"rag_chats"` holds a JSON array of `Chat` objects. Reads on mount, writes on every state change via a `useEffect` in the hook.

**State management:** A custom hook `useChatHistory` owns all chat state. `page.tsx` calls the hook and passes actions down as props. No React Context needed.

**Layout:** `app-shell` becomes a flex-row with a 240px fixed sidebar and a flex-1 main area. The existing header/chat/input structure is unchanged inside the main area.

## Data Model

```ts
type StoredMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
}

type Chat = {
  id: string        // crypto.randomUUID()
  title: string     // first user message, sliced to 52 chars
  messages: StoredMessage[]
  createdAt: number // Date.now()
  updatedAt: number // updated on every new message
}
```

Chats are sorted by `updatedAt` descending (most recent at top). No cap on number of chats.

## Hook: `useChatHistory` (`app/hooks/useChatHistory.ts`)

```ts
export function useChatHistory() {
  const [chats, setChats]           // persisted array
  const [activeChatId, setActive]   // null = "new chat" mode

  // derived
  const activeChat: Chat | null     // chats.find(c => c.id === activeChatId)

  // actions
  newChat()                         // sets activeChatId to null, clears visible messages
  selectChat(id: string)            // loads a previous chat
  saveMessage(msg: StoredMessage)   // appends to active chat; creates new Chat on first message
  setTitle(id: string, t: string)   // rename
  deleteChat(id: string)            // removes; if active, calls newChat()

  return { chats, activeChat, activeChatId, newChat, selectChat, saveMessage, setTitle, deleteChat }
}
```

**Auto-save flow:**
1. User sends first message in "new chat" mode → `saveMessage` creates a `Chat` with `title = question.slice(0, 52)` and `id = crypto.randomUUID()`, sets it as active.
2. Every subsequent message (user or assistant) → `saveMessage` appends to the active chat and bumps `updatedAt`.
3. `page.tsx` calls `saveMessage` immediately when the user sends (for the user message) and again when the assistant response arrives.

**Persistence:** a single `useEffect([chats])` writes `JSON.stringify(chats)` to `localStorage` on every change. On mount, reads and parses `localStorage.getItem("rag_chats") ?? "[]"`.

## Components

### `<Sidebar />` (defined in `page.tsx`)

Props: `{ chats, activeChatId, onNew, onSelect, onRename, onDelete }`

Structure:
- **Header:** `+ Novo chat` button (full width, amber accent on hover)
- **List:** scrollable, one `<ChatItem />` per chat, ordered by `updatedAt` desc
- **Footer:** empty (reserved for future use)

### `<ChatItem />` (defined in `page.tsx`)

Props: `{ chat, isActive, onSelect, onRename, onDelete }`

States:
- **Default:** title (truncated, single line) + relative timestamp ("há 2 min", "ontem", date if older)
- **Active:** left border in `--amber`, slightly brighter background (`--panel-soft`)
- **Hover:** delete icon (trash, 13px) appears on the right
- **Editing:** clicking the title replaces it with an `<input>` pre-filled with current title; confirms on `Enter` or `blur`, cancels on `Escape`

## Layout Changes (`globals.css`)

```
.app-shell        → flex-row (was flex-column)
.sidebar          → 240px wide, flex-column, border-right, background: --surface
.sidebar-header   → padding, border-bottom
.new-chat-btn     → full-width button, amber hover
.chat-list        → flex-1, overflow-y: auto, flex-column gap
.chat-item        → padding, border-radius, cursor pointer, hover bg --panel
.chat-item.active → border-left: 2px solid --amber, bg --panel-soft
.chat-item-title  → truncated single line (text-overflow: ellipsis)
.chat-item-meta   → font-size 10px, color --text-muted
.chat-item-delete → hidden by default, shown on .chat-item:hover
.main-area        → flex-1, flex-column (holds header + chat-main + input-bar)
```

## Files

| File | Action |
|---|---|
| `app/hooks/useChatHistory.ts` | Create |
| `app/app/page.tsx` | Modify — add Sidebar component, wire hook |
| `app/app/globals.css` | Modify — add sidebar styles, adjust app-shell |

## Error Handling

- `localStorage` parse error (corrupted data): catch and reset to `[]`
- `localStorage` quota exceeded: silently ignore write failure (chat still works in-session)
- Empty title after rename: revert to previous title

## Out of Scope

- Search across chats
- Export / import chats
- Sync across devices
- Chat folders / grouping
