import type { Metadata } from 'next'
import { Sora, Lora, Fira_Code } from 'next/font/google'
import './globals.css'

const sora = Sora({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-ui',
})

const lora = Lora({
  subsets: ['latin'],
  weight: ['400', '500'],
  style: ['normal', 'italic'],
  variable: '--font-prose',
})

const firaCode = Fira_Code({
  subsets: ['latin'],
  weight: ['400', '500'],
  variable: '--font-code',
})

export const metadata: Metadata = {
  title: 'data. — assistente RAG PT-BR',
  description:
    'Assistente em português para análise de dados com Python — pandas, NumPy, Matplotlib e Seaborn.',
}

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="pt-BR">
      <body className={`${sora.variable} ${lora.variable} ${firaCode.variable}`}>
        {children}
      </body>
    </html>
  )
}
