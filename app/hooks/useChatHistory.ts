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
  function saveMessage(msg: StoredMessage, targetChatId?: string | null): string | null {
    const cid = targetChatId !== undefined ? targetChatId : activeChatIdRef.current
    if (cid) {
      setChats(prev =>
        prev.map(c =>
          c.id === cid
            ? { ...c, messages: [...c.messages, msg], updatedAt: Date.now() }
            : c
        )
      )
      return cid
    } else {
      // Only user messages can start a new chat
      if (msg.role !== 'user') return null
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
      return newId
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
