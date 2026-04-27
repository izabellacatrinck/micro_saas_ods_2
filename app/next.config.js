/** @type {import('next').NextConfig} */

// ARCHITECTURE:
//   Frontend  → Vercel          (this Next.js app)
//   Backend   → HF Space        (FastAPI + RAG pipeline)
//
// The frontend calls the backend DIRECTLY from the browser.
// CORS is already set to "*" on the FastAPI side, so this works fine.
//
// LOCAL:       set NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:7860 in .env.local
// PRODUCTION:  set NEXT_PUBLIC_BACKEND_URL=https://tadashiroo-rag-pt-backend.hf.space
//              in Vercel Project Settings → Environment Variables

const nextConfig = {}

module.exports = nextConfig
