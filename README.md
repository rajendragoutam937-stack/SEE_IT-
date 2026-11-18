# AI Vision — Live Object Detection & AI Chatbot

## Quick Start (Self-Hosting)

1. Install dependencies:
   ```bash
   npm install
   ```

2. Create a `.env` file in the project root with your Groq API key (free at https://console.groq.com/keys) and/or Gemini API key:
   ```env
   GROQ_API_KEY="YOUR_GROQ_API_KEY"
   GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
   PORT=3000
   ```

3. Run in development or production:
   ```bash
   npm run dev
   # or for production:
   npm run build && npm start
   ```

Note: Object detection (TensorFlow.js COCO-SSD) runs directly in the browser.
The AI Chatbot calls `/api/chat` on `server.mjs`, so host with Node.js (e.g., Render, Railway, Cloud Run, Fly.io, or a VPS) and set `GROQ_API_KEY` (or `GEMINI_API_KEY`) in environment variables.
