/** @type {import('next').NextConfig} */
const backendUrl = process.env.HF_SPACE_URL || 'http://127.0.0.1:7860'

const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/backend/:path*',
        destination: `${backendUrl}/:path*`,
      },
    ]
  },
}

module.exports = nextConfig
