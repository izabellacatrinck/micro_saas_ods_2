'use client'

import { FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import {
  ArrowUpRight,
  BookOpenText,
  Bot,
  Braces,
  ChevronDown,
  ChevronUp,
  Command,
  Database,
  Loader2,
  Send,
  Sparkles,
  User,
} from 'lucide-react'

type Citation = {
  library?: string
  section?: string
  source_type?: string
}

type RetrievedChunk = {
  text?: string
  score?: number
  metadata?: {
    library?: string
    section?: string
    section_heading?: string
    source_type?: string
  }
}

type AssistantMeta = {
  model?: string
  backend?: string
  retrievalCount?: number
}

type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  retrievedChunks?: RetrievedChunk[]
  meta?: AssistantMeta
  error?: boolean
}

const quickPrompts = [
  'Explique quando usar merge, join e concat no pandas.',
  'Como transformar uma Series em DataFrame sem perder o índice?',
  'Mostre um exemplo de broadcasting no NumPy em linguagem simples.',
  'Como personalizar legenda e cores em um gráfico do Matplotlib?',
]

const libraryHighlights = [
  { label: 'pandas', icon: Database, tone: 'Frames, joins, limpeza e reshape' },
  { label: 'NumPy', icon: Braces, tone: 'Arrays, broadcasting e vetorização' },
  { label: 'Matplotlib', icon: Sparkles, tone: 'Composição e controle fino de plots' },
  { label: 'Seaborn', icon: BookOpenText, tone: 'Estatística visual com APIs de alto nível' },
]

function formatScore(score?: number) {
  if (typeof score !== 'number' || Number.isNaN(score)) return 'n/d'
  return score.toFixed(2)
}

function normalizeText(text: string) {
  return text.replace(/\r\n/g, '\n').trim()
}

function renderRichText(text: string) {
  const blocks = normalizeText(text).split(/```/)
  return blocks.map((block, index) => {
    if (index % 2 === 1) {
      const [firstLine, ...rest] = block.split('\n')
      const language = firstLine.trim().match(/^[a-zA-Z0-9_+#.-]+$/) ? firstLine.trim() : ''
      const code = language ? rest.join('\n') : block
      return (
        <pre key={index} className="overflow-x-auto rounded-2xl border border-[color:var(--line)] bg-[color:var(--panel-strong)] p-4 text-sm leading-6 text-[color:var(--text-primary)]">
          <code>{code.trim()}</code>
        </pre>
      )
    }

    return block
      .split(/\n{2,}/)
      .map((paragraph, paragraphIndex) => paragraph.trim())
      .filter(Boolean)
      .map((paragraph, paragraphIndex) => {
        const lines = paragraph.split('\n').map((line) => line.trim()).filter(Boolean)
        const isList = lines.every((line) => /^([-*]|\d+\.)\s+/.test(line))

        if (isList) {
          return (
            <ul key={`${index}-${paragraphIndex}`} className="space-y-2 pl-5 text-[color:var(--text-primary)]">
              {lines.map((line, lineIndex) => (
                <li key={lineIndex} className="list-disc marker:text-[color:var(--accent)]">
                  {line.replace(/^([-*]|\d+\.)\s+/, '')}
                </li>
              ))}
            </ul>
          )
        }

        return (
          <p key={`${index}-${paragraphIndex}`} className="text-[15px] leading-7 text-[color:var(--text-primary)]">
            {paragraph}
          </p>
        )
      })
  })
}

function AssistantBubble({ message }: { message: Message }) {
  const [showSources, setShowSources] = useState(false)
  const [showContext, setShowContext] = useState(false)

  return (
    <article className="grid gap-4 rounded-[28px] border border-[color:var(--line)] bg-[color:var(--panel)] p-5 shadow-[0_16px_60px_rgba(15,23,42,0.08)]">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[color:var(--accent)]/12 text-[color:var(--accent)]">
            <Bot className="h-5 w-5" />
          </div>
          <div>
            <p className="text-sm font-semibold text-[color:var(--text-primary)]">Assistente</p>
            <p className="text-xs text-[color:var(--text-muted)]">
              {message.meta?.model ?? 'Resposta fundamentada na documentação indexada'}
            </p>
          </div>
        </div>
        {message.meta?.retrievalCount ? (
          <Badge variant="outline" className="rounded-full border-[color:var(--line-strong)] bg-[color:var(--panel-strong)] px-3 py-1 text-[11px] uppercase tracking-[0.12em] text-[color:var(--text-muted)]">
            {message.meta.retrievalCount} fontes
          </Badge>
        ) : null}
      </div>

      <div className="grid gap-4">{renderRichText(message.content)}</div>

      <div className="flex flex-wrap gap-3">
        {message.citations?.length ? (
          <Collapsible open={showSources} onOpenChange={setShowSources}>
            <CollapsibleTrigger asChild>
              <Button variant="outline" className="h-10 rounded-full border-[color:var(--line-strong)] bg-transparent px-4 text-[color:var(--text-primary)]">
                <BookOpenText className="mr-2 h-4 w-4" />
                Fontes
                {showSources ? <ChevronUp className="ml-2 h-4 w-4" /> : <ChevronDown className="ml-2 h-4 w-4" />}
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3 grid gap-3">
              {message.citations.map((citation, index) => (
                <div
                  key={`${message.id}-citation-${index}`}
                  className="grid gap-1 rounded-2xl border border-[color:var(--line)] bg-[color:var(--panel-strong)] p-4"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge className="rounded-full bg-[color:var(--accent)] px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.12em] text-[color:var(--accent-foreground)]">
                      {citation.library ?? 'desconhecida'}
                    </Badge>
                    <span className="text-sm font-medium text-[color:var(--text-primary)]">
                      {citation.section ?? 'Seção não identificada'}
                    </span>
                  </div>
                  <p className="text-sm text-[color:var(--text-muted)]">
                    Origem: {citation.source_type ?? 'documentação indexada'}
                  </p>
                </div>
              ))}
            </CollapsibleContent>
          </Collapsible>
        ) : null}

        {message.retrievedChunks?.length ? (
          <Collapsible open={showContext} onOpenChange={setShowContext}>
            <CollapsibleTrigger asChild>
              <Button variant="outline" className="h-10 rounded-full border-[color:var(--line-strong)] bg-transparent px-4 text-[color:var(--text-primary)]">
                <Command className="mr-2 h-4 w-4" />
                Contexto
                {showContext ? <ChevronUp className="ml-2 h-4 w-4" /> : <ChevronDown className="ml-2 h-4 w-4" />}
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3 grid gap-3">
              {message.retrievedChunks.map((chunk, index) => (
                <div
                  key={`${message.id}-chunk-${index}`}
                  className="grid gap-3 rounded-2xl border border-[color:var(--line)] bg-[color:var(--panel-strong)] p-4"
                >
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge variant="outline" className="rounded-full border-[color:var(--line-strong)] px-3 py-1 text-[11px] uppercase tracking-[0.12em] text-[color:var(--text-muted)]">
                        Trecho {index + 1}
                      </Badge>
                      <span className="text-sm font-medium text-[color:var(--text-primary)]">
                        {chunk.metadata?.library ?? 'biblioteca desconhecida'}
                      </span>
                      <span className="text-sm text-[color:var(--text-muted)]">
                        {chunk.metadata?.section_heading ?? chunk.metadata?.section ?? 'Seção sem título'}
                      </span>
                    </div>
                    <span className="text-xs uppercase tracking-[0.12em] text-[color:var(--text-muted)]">
                      Score {formatScore(chunk.score)}
                    </span>
                  </div>
                  <p className="text-sm leading-6 text-[color:var(--text-secondary)]">
                    {normalizeText(chunk.text ?? 'Trecho indisponível.')}
                  </p>
                </div>
              ))}
            </CollapsibleContent>
          </Collapsible>
        ) : null}
      </div>
    </article>
  )
}

function UserBubble({ message }: { message: Message }) {
  return (
    <article className="ml-auto grid max-w-[48rem] gap-3 rounded-[28px] border border-[color:var(--line-strong)] bg-[color:var(--panel-strong)] p-5">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[color:var(--text-primary)] text-[color:var(--surface)]">
          <User className="h-5 w-5" />
        </div>
        <div>
          <p className="text-sm font-semibold text-[color:var(--text-primary)]">Você</p>
          <p className="text-xs text-[color:var(--text-muted)]">Pergunta enviada para a API do backend</p>
        </div>
      </div>
      <p className="text-[15px] leading-7 text-[color:var(--text-primary)]">{message.content}</p>
    </article>
  )
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [activePrompt, setActivePrompt] = useState<string | null>(null)
  const viewportRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    viewportRef.current?.scrollTo({ top: viewportRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, isLoading])

  const lastAssistant = useMemo(
    () => [...messages].reverse().find((message) => message.role === 'assistant' && !message.error),
    [messages]
  )

  async function submitQuestion(question: string) {
    const trimmed = question.trim()
    if (!trimmed || isLoading) return

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: trimmed,
    }

    setMessages((current) => [...current, userMessage])
    setInput('')
    setIsLoading(true)
    setActivePrompt(trimmed)

    try {
      const response = await fetch('/backend/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed }),
      })

      const payload = await response.json()

      if (!response.ok) {
        throw new Error(payload?.detail ?? payload?.error ?? 'Falha ao consultar o assistente.')
      }

      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: payload.answer,
        citations: payload.citations ?? [],
      }

      setMessages((current) => [...current, assistantMessage])
    } catch (error) {
      const assistantMessage: Message = {
        id: `assistant-error-${Date.now()}`,
        role: 'assistant',
        content:
          error instanceof Error
            ? error.message
            : 'Não consegui concluir a resposta agora. Verifique a conexão com o backend e tente de novo.',
        error: true,
      }

      setMessages((current) => [...current, assistantMessage])
    } finally {
      setIsLoading(false)
      setActivePrompt(null)
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    void submitQuestion(input)
  }

  return (
    <main className="min-h-screen bg-[color:var(--surface)] text-[color:var(--text-primary)]">
      <section className="hero-grid border-b border-[color:var(--line)]">
        <div className="mx-auto grid w-full max-w-[1400px] gap-10 px-5 py-8 lg:grid-cols-[1.15fr_0.85fr] lg:px-8 lg:py-10">
          <div className="grid gap-8">
            <div className="inline-flex w-fit items-center gap-2 rounded-full border border-[color:var(--line-strong)] bg-[color:var(--panel)] px-4 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
              <Sparkles className="h-4 w-4 text-[color:var(--accent)]" />
              Micro SaaS ODS 2
            </div>

            <div className="grid gap-5">
              <h1 className="max-w-[12ch] text-4xl font-semibold leading-none text-[color:var(--text-primary)] sm:text-5xl lg:text-6xl">
                Assistente de código para análise de dados em Python.
              </h1>
              <p className="max-w-2xl text-base leading-7 text-[color:var(--text-secondary)] sm:text-lg">
                Uma interface de trabalho para conversar com a documentação indexada de `pandas`, `NumPy`, `Matplotlib` e `Seaborn` em português, com respostas fundamentadas, citações claras e contexto recuperado.
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              {quickPrompts.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  onClick={() => {
                    setInput(prompt)
                    void submitQuestion(prompt)
                  }}
                  disabled={isLoading}
                  className="group rounded-[24px] border border-[color:var(--line)] bg-[color:var(--panel)] px-4 py-4 text-left transition hover:-translate-y-0.5 hover:border-[color:var(--line-strong)] hover:bg-[color:var(--panel-strong)] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <div className="flex items-start justify-between gap-3">
                    <span className="text-sm leading-6 text-[color:var(--text-primary)]">{prompt}</span>
                    <ArrowUpRight className="mt-1 h-4 w-4 shrink-0 text-[color:var(--accent)] transition group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="grid gap-4">
            <div className="grid gap-3 rounded-[30px] border border-[color:var(--line)] bg-[color:var(--panel)] p-5 shadow-[0_18px_80px_rgba(15,23,42,0.08)]">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-[color:var(--text-primary)]">Cobertura do assistente</p>
                  <p className="text-sm text-[color:var(--text-muted)]">Especializado nas bibliotecas que já estão no corpus traduzido.</p>
                </div>
                <div className="rounded-2xl bg-[color:var(--accent)]/12 p-3 text-[color:var(--accent)]">
                  <Bot className="h-5 w-5" />
                </div>
              </div>

              <div className="grid gap-3">
                {libraryHighlights.map(({ label, icon: Icon, tone }) => (
                  <div key={label} className="flex items-center gap-3 rounded-[22px] border border-[color:var(--line)] bg-[color:var(--panel-strong)] px-4 py-3">
                    <div className="rounded-2xl bg-[color:var(--surface)] p-2 text-[color:var(--accent)]">
                      <Icon className="h-4 w-4" />
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-[color:var(--text-primary)]">{label}</p>
                      <p className="text-sm text-[color:var(--text-muted)]">{tone}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid gap-3 rounded-[30px] border border-[color:var(--line)] bg-[color:var(--panel)] p-5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-[color:var(--text-primary)]">Última resposta útil</p>
                  <p className="text-sm text-[color:var(--text-muted)]">Leitura rápida do que o sistema acabou de produzir.</p>
                </div>
                <Badge className="rounded-full bg-[color:var(--accent)] px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.12em] text-[color:var(--accent-foreground)]">
                  RAG
                </Badge>
              </div>

              <div className="rounded-[22px] border border-[color:var(--line)] bg-[color:var(--panel-strong)] p-4">
                <p className="line-clamp-5 text-sm leading-7 text-[color:var(--text-secondary)]">
                  {lastAssistant?.content ??
                    'As respostas vão aparecer aqui com o mesmo foco do chat principal: documentação, exemplos práticos e contexto recuperado.'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto grid w-full max-w-[1400px] gap-6 px-5 py-6 lg:grid-cols-[0.75fr_1.25fr] lg:px-8 lg:py-8">
        <aside className="grid h-fit gap-6 lg:sticky lg:top-6">
          <div className="grid gap-4 rounded-[30px] border border-[color:var(--line)] bg-[color:var(--panel)] p-5">
            <div>
              <p className="text-sm font-semibold text-[color:var(--text-primary)]">Modo de trabalho</p>
              <p className="mt-1 text-sm leading-6 text-[color:var(--text-muted)]">
                Faça perguntas como faria para um colega técnico: peça explicação, comparação, exemplo, diagnóstico ou tradução de conceitos para código.
              </p>
            </div>
            <div className="grid gap-3">
              <div className="rounded-[22px] border border-[color:var(--line)] bg-[color:var(--panel-strong)] p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[color:var(--text-muted)]">Melhor quando</p>
                <p className="mt-2 text-sm leading-6 text-[color:var(--text-secondary)]">
                  Você quer entender APIs, escolher abordagens e localizar a parte certa da documentação sem abrir dezenas de abas.
                </p>
              </div>
              <div className="rounded-[22px] border border-[color:var(--line)] bg-[color:var(--panel-strong)] p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[color:var(--text-muted)]">Restrições</p>
                <p className="mt-2 text-sm leading-6 text-[color:var(--text-secondary)]">
                  O sistema responde com base no corpus indexado. Quando a informação não está nele, a interface deve sinalizar isso com honestidade.
                </p>
              </div>
            </div>
          </div>

          <div className="grid gap-3 rounded-[30px] border border-[color:var(--line)] bg-[color:var(--panel)] p-5">
            <p className="text-sm font-semibold text-[color:var(--text-primary)]">Atalhos de conversa</p>
            {quickPrompts.map((prompt) => (
              <button
                key={`aside-${prompt}`}
                type="button"
                onClick={() => setInput(prompt)}
                className="rounded-[20px] border border-[color:var(--line)] bg-[color:var(--panel-strong)] px-4 py-3 text-left text-sm leading-6 text-[color:var(--text-secondary)] transition hover:border-[color:var(--line-strong)] hover:text-[color:var(--text-primary)]"
              >
                {prompt}
              </button>
            ))}
          </div>
        </aside>

        <section className="grid min-h-[720px] grid-rows-[1fr_auto] rounded-[32px] border border-[color:var(--line)] bg-[color:var(--panel)] shadow-[0_25px_90px_rgba(15,23,42,0.08)]">
          <div ref={viewportRef} className="grid gap-5 overflow-y-auto px-5 py-5 sm:px-6 sm:py-6">
            {messages.length === 0 ? (
              <div className="grid place-items-center py-10">
                <div className="grid max-w-xl gap-5 text-center">
                  <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-[24px] bg-[color:var(--accent)]/12 text-[color:var(--accent)]">
                    <Bot className="h-8 w-8" />
                  </div>
                  <div className="grid gap-3">
                    <h2 className="text-2xl font-semibold text-[color:var(--text-primary)]">Pronto para revisar código com contexto.</h2>
                    <p className="text-base leading-7 text-[color:var(--text-secondary)]">
                      Pergunte como resolver uma operação, quando usar certa API ou por que um comportamento de `pandas` ou `NumPy` acontece.
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              messages.map((message) =>
                message.role === 'assistant' ? (
                  <AssistantBubble key={message.id} message={message} />
                ) : (
                  <UserBubble key={message.id} message={message} />
                )
              )
            )}

            {isLoading ? (
              <div className="grid gap-4 rounded-[28px] border border-dashed border-[color:var(--line-strong)] bg-[color:var(--panel-strong)] p-5">
                <div className="flex items-center gap-3">
                  <Loader2 className="h-5 w-5 animate-spin text-[color:var(--accent)]" />
                  <p className="text-sm font-medium text-[color:var(--text-primary)]">
                    Buscando contexto e preparando a resposta
                  </p>
                </div>
                <p className="text-sm leading-6 text-[color:var(--text-muted)]">
                  {activePrompt ? `Pergunta ativa: "${activePrompt}"` : 'Consultando o backend RAG.'}
                </p>
              </div>
            ) : null}
          </div>

          <div className="border-t border-[color:var(--line)] bg-[color:var(--panel)] px-5 py-5 sm:px-6">
            <form onSubmit={handleSubmit} className="grid gap-4">
              <div className="rounded-[28px] border border-[color:var(--line-strong)] bg-[color:var(--surface)] p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.6)]">
                <textarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  placeholder="Descreva sua dúvida, cole um trecho de código ou peça um exemplo com pandas, NumPy, Matplotlib ou Seaborn."
                  rows={4}
                  disabled={isLoading}
                  className="min-h-[112px] w-full resize-none bg-transparent px-2 py-2 text-[15px] leading-7 text-[color:var(--text-primary)] outline-none placeholder:text-[color:var(--text-muted)] disabled:cursor-not-allowed"
                />
                <div className="mt-3 flex flex-col gap-3 border-t border-[color:var(--line)] pt-3 sm:flex-row sm:items-center sm:justify-between">
                  <p className="text-xs uppercase tracking-[0.14em] text-[color:var(--text-muted)]">
                    Respostas ancoradas no corpus indexado
                  </p>
                  <Button
                    type="submit"
                    disabled={isLoading || !input.trim()}
                    className="h-11 rounded-full bg-[color:var(--accent)] px-5 text-sm font-semibold text-[color:var(--accent-foreground)] hover:bg-[color:var(--accent-strong)]"
                  >
                    {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Send className="mr-2 h-4 w-4" />}
                    Enviar pergunta
                  </Button>
                </div>
              </div>
            </form>
          </div>
        </section>
      </section>
    </main>
  )
}
