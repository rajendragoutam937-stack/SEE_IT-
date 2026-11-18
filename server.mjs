import 'dotenv/config';
import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORT = Number(process.env.PORT || 3000);
const IS_PROD = process.env.NODE_ENV === 'production';

function createGeminiClient() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function parseDataUrl(dataUrl) {
  if (typeof dataUrl !== 'string') return null;
  const match = dataUrl.match(/^data:(image\/(?:jpeg|jpg|png|webp));base64,([A-Za-z0-9+/=]+)$/i);
  if (!match) return null;
  const mimeType = match[1].toLowerCase() === 'image/jpg' ? 'image/jpeg' : match[1].toLowerCase();
  return {
    mimeType,
    data: match[2],
  };
}

function extractReadableErrorMessage(raw) {
  if (!raw) return 'Unknown server error.';
  try {
    const outer = JSON.parse(raw);
    const innerMsg = outer?.error?.message || outer?.message;
    if (typeof innerMsg === 'string') {
      try {
        const inner = JSON.parse(innerMsg);
        if (inner?.error?.message) return inner.error.message;
      } catch {
        return innerMsg;
      }
    }
  } catch {
    // Not raw JSON
  }
  return raw;
}

function formatChatError(error) {
  const raw = error instanceof Error ? error.message : String(error || 'Unknown error');
  const readable = extractReadableErrorMessage(raw);
  const isGroq = raw.startsWith('Groq API error');
  const providerName = isGroq ? 'Groq' : 'Gemini';
  const envVarName = isGroq ? 'GROQ_API_KEY' : 'GEMINI_API_KEY';

  if (raw.includes('API_KEY_INVALID') || raw.includes('PERMISSION_DENIED') || raw.includes('403') || raw.includes('401') || raw.includes('invalid_api_key')) {
    return {
      status: 401,
      message: `${providerName} API authentication failed. Please verify your ${envVarName}.`,
    };
  }
  if (raw.includes('RESOURCE_EXHAUSTED') || raw.includes('429') || raw.includes('rate_limit_exceeded')) {
    return {
      status: 429,
      message: `${providerName} free-tier rate limit reached. Wait a moment and try again.`,
    };
  }
  if (raw.includes('NOT_FOUND') || raw.includes('404') || raw.includes('model_not_found')) {
    return {
      status: 404,
      message: `This API key does not have access to the configured ${providerName} model.`,
    };
  }
  if (raw.includes('503') || raw.includes('UNAVAILABLE')) {
    return {
      status: 503,
      message: 'The AI service is temporarily experiencing high demand. Please try again in a moment.',
    };
  }
  return {
    status: 500,
    message: readable || `Failed to reach ${providerName} API.`,
  };
}

function isTransientError(error) {
  const raw = error instanceof Error ? error.message : String(error || '');
  return raw.includes('503') || raw.includes('UNAVAILABLE') || raw.includes('500') || raw.includes('ECONNRESET');
}

async function withRetry(fn, retries = 1) {
  let lastError;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (attempt < retries && isTransientError(error)) {
        await new Promise((resolve) => setTimeout(resolve, 450));
        continue;
      }
      throw error;
    }
  }
  throw lastError;
}

async function streamChatModel(ai, model, thinkingLevel, contents, systemInstruction, onDelta, abortSignal) {
  const stream = await ai.models.generateContentStream({
    model,
    contents,
    config: {
      systemInstruction,
      thinkingConfig: { thinkingLevel },
      abortSignal,
    },
  });

  let emittedAny = false;
  for await (const chunk of stream) {
    const delta = chunk?.text || '';
    if (delta) {
      emittedAny = true;
      onDelta(delta);
    }
  }
  return emittedAny;
}

async function sendGroqRequest(groqApiKey, model, groqMessages, abortSignal) {
  const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${groqApiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      messages: groqMessages,
      stream: true,
      max_tokens: 700,
    }),
    signal: abortSignal,
  });

  if (!response.ok) {
    const errText = await response.text().catch(() => '');
    const err = new Error(`Groq API error (${response.status}): ${errText || response.statusText}`);
    err.status = response.status;
    throw err;
  }

  return response;
}

async function streamGroqChat(groqApiKey, sanitizedHistory, imageDataUrl, systemInstruction, onDelta, abortSignal) {
  const hasImage = Boolean(imageDataUrl && parseDataUrl(imageDataUrl));
  const buildMessages = (includeImage) => {
    const msgs = [{ role: 'system', content: systemInstruction }];
    sanitizedHistory.forEach((msg, idx) => {
      const role = msg.role === 'model' ? 'assistant' : 'user';
      const isLastUser = idx === sanitizedHistory.length - 1 && role === 'user';
      if (isLastUser && includeImage) {
        msgs.push({
          role: 'user',
          content: [
            { type: 'text', text: msg.text },
            { type: 'image_url', image_url: { url: imageDataUrl } },
          ],
        });
      } else {
        msgs.push({ role, content: msg.text });
      }
    });
    return msgs;
  };

  let response;
  if (hasImage) {
    try {
      response = await sendGroqRequest(
        groqApiKey,
        'meta-llama/llama-4-scout-17b-16e-instruct',
        buildMessages(true),
        abortSignal
      );
    } catch (err) {
      if (err?.status === 400 || err?.status === 404) {
        response = await sendGroqRequest(
          groqApiKey,
          'llama-3.3-70b-versatile',
          buildMessages(false),
          abortSignal
        );
      } else {
        throw err;
      }
    }
  } else {
    response = await sendGroqRequest(
      groqApiKey,
      'llama-3.3-70b-versatile',
      buildMessages(false),
      abortSignal
    );
  }

  if (!response.body) {
    throw new Error('Groq streaming response unavailable.');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let emittedAny = false;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const dataStr = trimmed.slice(5).trim();
      if (dataStr === '[DONE]') continue;
      try {
        const json = JSON.parse(dataStr);
        const delta = json?.choices?.[0]?.delta?.content || '';
        if (delta) {
          emittedAny = true;
          onDelta(delta);
        }
      } catch {
        // Skip partial SSE frame
      }
    }
  }

  return emittedAny;
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '12mb' }));

  app.get('/api/health', (_req, res) => {
    const geminiConfigured = Boolean(process.env.GEMINI_API_KEY);
    const groqConfigured = Boolean(process.env.GROQ_API_KEY);
    res.json({
      ok: true,
      geminiConfigured,
      groqConfigured,
      aiConfigured: geminiConfigured || groqConfigured,
    });
  });

  app.post('/api/chat', async (req, res) => {
    const groqApiKey = process.env.GROQ_API_KEY;
    const ai = createGeminiClient();
    if (!ai && !groqApiKey) {
      res.status(503).json({
        error: 'No AI API key configured. Set GROQ_API_KEY or GEMINI_API_KEY in environment variables.',
      });
      return;
    }

    const { messages, detectedObjects, imageDataUrl, mediaMode } = req.body || {};
    if (!Array.isArray(messages) || messages.length === 0) {
      res.status(400).json({ error: 'Conversation messages are required.' });
      return;
    }

    const sanitizedHistory = messages
      .filter((m) => m && (m.role === 'user' || m.role === 'model') && typeof m.text === 'string' && m.text.trim().length > 0)
      .slice(-10);

    if (sanitizedHistory.length === 0) {
      res.status(400).json({ error: 'At least one non-empty message is required.' });
      return;
    }

    const detectionSummary = Array.isArray(detectedObjects) && detectedObjects.length > 0
      ? detectedObjects
          .map((obj) => `${obj.label} (${Math.round(Number(obj.confidence) || 0)}%)`)
          .join(', ')
      : 'None currently detected';

    const modeLabel = mediaMode === 'camera'
      ? 'Live camera feed'
      : mediaMode === 'image'
        ? 'Uploaded image'
        : 'No active visual input';

    const systemInstruction = [
      'You are a fast, helpful AI Vision and Knowledge Assistant inside an object detection workspace.',
      'Give clear, direct, concise answers in natural language.',
      `Current visual input mode: ${modeLabel}.`,
      `Objects detected right now: ${detectionSummary}.`,
      'If an image frame is attached, inspect it directly to answer visual questions accurately. For general knowledge questions, answer immediately and clearly.',
    ].join('\n');

    const contents = sanitizedHistory.map((msg, idx) => {
      const isLastUser = idx === sanitizedHistory.length - 1 && msg.role === 'user';
      const parts = [];
      if (isLastUser && imageDataUrl) {
        const parsedImage = parseDataUrl(imageDataUrl);
        if (parsedImage) {
          parts.push({
            inlineData: {
              mimeType: parsedImage.mimeType,
              data: parsedImage.data,
            },
          });
        }
      }
      parts.push({ text: msg.text });
      return {
        role: msg.role,
        parts,
      };
    });

    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    res.flushHeaders?.();
    res.write(': connected\n\n');

    const writeEvent = (event, data) => {
      res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    };

    const clientAbort = new AbortController();
    res.on('close', () => {
      if (!res.writableEnded) {
        clientAbort.abort();
      }
    });

    const upstreamSignal = AbortSignal.any([clientAbort.signal, AbortSignal.timeout(20000)]);

    try {
      let emitted = false;

      if (groqApiKey) {
        try {
          emitted = await streamGroqChat(
            groqApiKey,
            sanitizedHistory,
            imageDataUrl,
            systemInstruction,
            (delta) => writeEvent('delta', { text: delta }),
            upstreamSignal
          );
        } catch (groqErr) {
          if (clientAbort.signal.aborted || !ai) {
            throw groqErr;
          }
        }
      }

      if (!emitted && ai) {
        try {
          emitted = await withRetry(() =>
            streamChatModel(
              ai,
              'gemini-3.1-flash-lite',
              ThinkingLevel.MINIMAL,
              contents,
              systemInstruction,
              (delta) => {
                writeEvent('delta', { text: delta });
              },
              upstreamSignal
            )
          );
        } catch (primaryErr) {
          if (clientAbort.signal.aborted) {
            return;
          }
          if (!emitted) {
            await withRetry(() =>
              streamChatModel(
                ai,
                'gemini-4-hydrogen',
                ThinkingLevel.LOW,
                contents,
                systemInstruction,
                (delta) => {
                  writeEvent('delta', { text: delta });
                },
                upstreamSignal
              )
            );
          } else {
            throw primaryErr;
          }
        }
      }

      writeEvent('done', { ok: true });
      res.end();
    } catch (error) {
      if (clientAbort.signal.aborted) {
        res.end();
        return;
      }
      if (error?.name === 'TimeoutError' || error?.name === 'AbortError') {
        writeEvent('error', { message: 'AI provider took too long to respond. Please try again.' });
        res.end();
        return;
      }
      const formatted = formatChatError(error);
      writeEvent('error', { message: formatted.message });
      res.end();
    }
  });

  app.post('/api/vision/detect', async (req, res) => {
    const ai = createGeminiClient();
    if (!ai) {
      res.status(503).json({
        error: 'GEMINI_API_KEY is not configured on the server.',
      });
      return;
    }

    const { imageDataUrl } = req.body || {};
    const parsedImage = parseDataUrl(imageDataUrl);
    if (!parsedImage) {
      res.status(400).json({ error: 'Valid base64 image data URL is required.' });
      return;
    }

    try {
      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: {
          parts: [
            {
              inlineData: {
                mimeType: parsedImage.mimeType,
                data: parsedImage.data,
              },
            },
            {
              text: 'Detect all clearly visible physical objects, people, animals, plants, furniture, and everyday items in this image. Return tight bounding boxes in normalized 0..1000 coordinates [ymin, xmin, ymax, xmax], concise object labels, and calibrated confidence scores between 0 and 1.',
            },
          ],
        },
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                label: { type: Type.STRING },
                confidence: { type: Type.NUMBER },
                box_2d: {
                  type: Type.ARRAY,
                  items: { type: Type.NUMBER },
                },
              },
              required: ['label', 'confidence', 'box_2d'],
            },
          },
        },
      });

      const rawText = (response.text || '[]').trim();
      const parsed = JSON.parse(rawText);
      const detections = Array.isArray(parsed)
        ? parsed
            .filter(
              (item) =>
                item &&
                typeof item.label === 'string' &&
                Array.isArray(item.box_2d) &&
                item.box_2d.length === 4
            )
            .map((item) => ({
              label: item.label.trim(),
              score: Math.max(0.01, Math.min(0.99, Number(item.confidence) || 0.85)),
              box_2d: item.box_2d.map((v) => Math.max(0, Math.min(1000, Number(v) || 0))),
            }))
        : [];

      res.json({ detections });
    } catch (error) {
      const formatted = formatChatError(error);
      res.status(formatted.status).json({ error: formatted.message });
    }
  });

  const PROJECT_EXPORT_FILES = [
    { src: '.env.example', dest: '.env.example' },
    { src: '.gitignore', dest: '.gitignore' },
    { src: 'index.html', dest: 'index.html' },
    { src: 'metadata.json', dest: 'metadata.json' },
    { src: 'package.json', dest: 'package.json' },
    { src: 'server.mjs', dest: 'server.mjs' },
    { src: 'tsconfig.json', dest: 'tsconfig.json' },
    { src: 'vite.config.ts', dest: 'vite.config.ts' },
    { src: 'src/app.js', dest: 'app.js' },
    { src: 'src/chat.js', dest: 'chat.js' },
    { src: 'src/detector.js', dest: 'detector.js' },
    { src: 'src/index.css', dest: 'index.css' },
    { src: 'src/theme.js', dest: 'theme.js' },
  ];

  const EXPORT_README = [
    '# AI Vision — Live Object Detection & AI Chatbot',
    '',
    '## Quick Start (Self-Hosting)',
    '',
    '1. Install dependencies:',
    '   ```bash',
    '   npm install',
    '   ```',
    '',
    '2. Create a `.env` file in the project root with your Groq API key (free at https://console.groq.com/keys) and/or Gemini API key:',
    '   ```env',
    '   GROQ_API_KEY="YOUR_GROQ_API_KEY"',
    '   GEMINI_API_KEY="YOUR_GEMINI_API_KEY"',
    '   PORT=3000',
    '   ```',
    '',
    '3. Run in development or production:',
    '   ```bash',
    '   npm run dev',
    '   # or for production:',
    '   npm run build && npm start',
    '   ```',
    '',
    'Note: Object detection (TensorFlow.js COCO-SSD) runs directly in the browser.',
    'The AI Chatbot calls `/api/chat` on `server.mjs`, so host with Node.js (e.g., Render, Railway, Cloud Run, Fly.io, or a VPS) and set `GROQ_API_KEY` (or `GEMINI_API_KEY`) in environment variables.',
    '',
  ].join('\n');

  function crc32(buf) {
    let crc = 0xffffffff;
    for (let i = 0; i < buf.length; i++) {
      crc ^= buf[i];
      for (let j = 0; j < 8; j++) {
        crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
      }
    }
    return (crc ^ 0xffffffff) >>> 0;
  }

  function buildZipBuffer(entries) {
    const localParts = [];
    const centralParts = [];
    let offset = 0;

    // Valid MS-DOS timestamp (2026-11-17 10:00:00) so Windows File Explorer never rejects 0x0000 dates
    const dosTime = 0x5000;
    const dosDate = 0x5d71;

    for (const entry of entries) {
      const nameBuf = Buffer.from(entry.name, 'utf8');
      const rawData = Buffer.isBuffer(entry.data) ? entry.data : Buffer.from(entry.data, 'utf8');
      const crc = crc32(rawData);

      const localHeader = Buffer.alloc(30);
      localHeader.writeUInt32LE(0x04034b50, 0);
      localHeader.writeUInt16LE(10, 4); // version needed: 1.0 (STORE)
      localHeader.writeUInt16LE(0, 6);  // flags: 0 (pure ASCII paths)
      localHeader.writeUInt16LE(0, 8);  // method: 0 (STORE - universal compatibility)
      localHeader.writeUInt16LE(dosTime, 10);
      localHeader.writeUInt16LE(dosDate, 12);
      localHeader.writeUInt32LE(crc, 14);
      localHeader.writeUInt32LE(rawData.length, 18);
      localHeader.writeUInt32LE(rawData.length, 22);
      localHeader.writeUInt16LE(nameBuf.length, 26);
      localHeader.writeUInt16LE(0, 28);

      localParts.push(localHeader, nameBuf, rawData);

      const centralHeader = Buffer.alloc(46);
      centralHeader.writeUInt32LE(0x02014b50, 0);
      centralHeader.writeUInt16LE(20, 4); // version made by: 2.0 (MS-DOS/Windows)
      centralHeader.writeUInt16LE(10, 6); // version needed: 1.0
      centralHeader.writeUInt16LE(0, 8);
      centralHeader.writeUInt16LE(0, 10); // method: 0 (STORE)
      centralHeader.writeUInt16LE(dosTime, 12);
      centralHeader.writeUInt16LE(dosDate, 14);
      centralHeader.writeUInt32LE(crc, 16);
      centralHeader.writeUInt32LE(rawData.length, 20);
      centralHeader.writeUInt32LE(rawData.length, 24);
      centralHeader.writeUInt16LE(nameBuf.length, 28);
      centralHeader.writeUInt16LE(0, 30);
      centralHeader.writeUInt16LE(0, 32);
      centralHeader.writeUInt16LE(0, 34);
      centralHeader.writeUInt16LE(0, 36);
      centralHeader.writeUInt32LE(0x20, 38); // Windows archive file attribute
      centralHeader.writeUInt32LE(offset, 42);

      centralParts.push(centralHeader, nameBuf);
      offset += localHeader.length + nameBuf.length + rawData.length;
    }

    const centralDir = Buffer.concat(centralParts);
    const eocd = Buffer.alloc(22);
    eocd.writeUInt32LE(0x06054b50, 0);
    eocd.writeUInt16LE(0, 4);
    eocd.writeUInt16LE(0, 6);
    eocd.writeUInt16LE(entries.length, 8);
    eocd.writeUInt16LE(entries.length, 10);
    eocd.writeUInt32LE(centralDir.length, 12);
    eocd.writeUInt32LE(offset, 16);
    eocd.writeUInt16LE(0, 20);

    return Buffer.concat([...localParts, centralDir, eocd]);
  }

  app.get('/api/export-project', (req, res) => {
    const code = typeof req.query.code === 'string' ? req.query.code.trim() : '';
    if (code !== 'BRUHHD2DCODEFILES') {
      res.status(403).json({ error: 'Invalid export code.' });
      return;
    }

    const entries = [{ name: 'ai-vision-project/README.md', data: EXPORT_README }];
    for (const item of PROJECT_EXPORT_FILES) {
      const absPath = path.resolve(__dirname, item.src);
      if (fs.existsSync(absPath)) {
        let data = fs.readFileSync(absPath);
        if (item.dest === 'index.html') {
          const rewritten = data
            .toString('utf8')
            .replace('href="/src/index.css"', 'href="/index.css"')
            .replace('src="/src/app.js"', 'src="/app.js"');
          data = Buffer.from(rewritten, 'utf8');
        }
        entries.push({
          name: `ai-vision-project/${item.dest}`,
          data,
        });
      }
    }

    const zipBuf = buildZipBuffer(entries);
    res.setHeader('Content-Type', 'application/zip');
    res.setHeader('Content-Disposition', 'attachment; filename="ai-vision-project.zip"');
    res.setHeader('Content-Length', String(zipBuf.length));
    res.send(zipBuf);
  });

  if (!IS_PROD) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Aperture Vision server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
