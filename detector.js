import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

const MIN_CONFIDENCE = 0.45;
const MAX_DETECTIONS = 20;

const CLASS_LABEL_MAP = {
  'potted plant': 'Plant',
  'cell phone': 'Phone',
  'dining table': 'Table',
};

const CLASS_COLOR_MAP = {
  person: '#0066FF',
  'potted plant': '#00B81D',
  chair: '#9333EA',
  couch: '#9333EA',
  bed: '#9333EA',
  'dining table': '#9333EA',
  laptop: '#0284C7',
  tv: '#0284C7',
  keyboard: '#0284C7',
  mouse: '#0284C7',
  'cell phone': '#0284C7',
  cup: '#EA580C',
  bottle: '#EA580C',
  bowl: '#EA580C',
  book: '#D97706',
  backpack: '#D97706',
  dog: '#0D9488',
  cat: '#0D9488',
  bird: '#0D9488',
};

const FALLBACK_PALETTE = ['#0066FF', '#00B81D', '#9333EA', '#0284C7', '#EA580C', '#0D9488'];

export function getDetectionColor(rawClass, index = 0) {
  const key = (rawClass || '').toLowerCase().trim();
  if (CLASS_COLOR_MAP[key]) return CLASS_COLOR_MAP[key];
  let hash = index;
  for (let i = 0; i < key.length; i++) {
    hash = (hash * 31 + key.charCodeAt(i)) | 0;
  }
  return FALLBACK_PALETTE[Math.abs(hash) % FALLBACK_PALETTE.length];
}

export function formatObjectLabel(rawLabel) {
  if (!rawLabel || typeof rawLabel !== 'string') return 'Object';
  const normalized = rawLabel.trim().toLowerCase();
  if (CLASS_LABEL_MAP[normalized]) return CLASS_LABEL_MAP[normalized];
  return normalized
    .split(/\s+/)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function computeIoU(boxA, boxB) {
  const xA = Math.max(boxA[0], boxB[0]);
  const yA = Math.max(boxA[1], boxB[1]);
  const xB = Math.min(boxA[0] + boxA[2], boxB[0] + boxB[2]);
  const yB = Math.min(boxA[1] + boxA[3], boxB[1] + boxB[3]);

  const interW = Math.max(0, xB - xA);
  const interH = Math.max(0, yB - yA);
  const interArea = interW * interH;
  if (interArea <= 0) return 0;

  const areaA = boxA[2] * boxA[3];
  const areaB = boxB[2] * boxB[3];
  return interArea / (areaA + areaB - interArea);
}

export function getContainedMediaRect(containerWidth, containerHeight, mediaWidth, mediaHeight) {
  if (!containerWidth || !containerHeight || !mediaWidth || !mediaHeight) {
    return { x: 0, y: 0, width: containerWidth, height: containerHeight };
  }

  const containerRatio = containerWidth / containerHeight;
  const mediaRatio = mediaWidth / mediaHeight;

  let drawWidth = containerWidth;
  let drawHeight = containerHeight;
  let offsetX = 0;
  let offsetY = 0;

  if (mediaRatio > containerRatio) {
    drawWidth = containerWidth;
    drawHeight = containerWidth / mediaRatio;
    offsetY = (containerHeight - drawHeight) / 2;
  } else {
    drawHeight = containerHeight;
    drawWidth = containerHeight * mediaRatio;
    offsetX = (containerWidth - drawWidth) / 2;
  }

  return { x: offsetX, y: offsetY, width: drawWidth, height: drawHeight };
}

export class VisionDetector {
  constructor() {
    this.model = null;
    this.modelPromise = null;
    this.useServerFallback = false;
    this.previousCameraDetections = [];
  }

  async ensureModelLoaded() {
    if (this.model) return { engine: 'coco-ssd' };
    if (this.useServerFallback) return { engine: 'gemini-vision' };

    if (!this.modelPromise) {
      this.modelPromise = (async () => {
        try {
          try {
            await tf.setBackend('webgl');
          } catch {
            // Fall back to default CPU/WASM backend if WebGL is unavailable
          }
          await tf.ready();
          this.model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
          return { engine: 'coco-ssd' };
        } catch (err) {
          console.warn('Local COCO-SSD model failed to load, switching to server Gemini Vision fallback:', err);
          this.useServerFallback = true;
          return { engine: 'gemini-vision' };
        }
      })();
    }

    return this.modelPromise;
  }

  resetTracking() {
    this.previousCameraDetections = [];
  }

  captureSnapshotDataUrl(mediaEl, maxDimension = 960, quality = 0.82) {
    const mediaW = mediaEl.videoWidth || mediaEl.naturalWidth || 0;
    const mediaH = mediaEl.videoHeight || mediaEl.naturalHeight || 0;
    if (!mediaW || !mediaH) return null;

    const scale = Math.min(1, maxDimension / Math.max(mediaW, mediaH));
    const targetW = Math.max(1, Math.round(mediaW * scale));
    const targetH = Math.max(1, Math.round(mediaH * scale));

    const offscreen = document.createElement('canvas');
    offscreen.width = targetW;
    offscreen.height = targetH;
    const ctx = offscreen.getContext('2d');
    if (!ctx) return null;

    ctx.drawImage(mediaEl, 0, 0, targetW, targetH);
    return offscreen.toDataURL('image/jpeg', quality);
  }

  async detectViaServer(mediaEl) {
    const mediaW = mediaEl.videoWidth || mediaEl.naturalWidth || 0;
    const mediaH = mediaEl.videoHeight || mediaEl.naturalHeight || 0;
    const imageDataUrl = this.captureSnapshotDataUrl(mediaEl, 1024, 0.85);
    if (!imageDataUrl) return [];

    const response = await fetch('/api/vision/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageDataUrl }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || 'Vision detection request failed.');
    }

    const items = Array.isArray(payload.detections) ? payload.detections : [];
    return items.map((item, index) => {
      const [ymin, xmin, ymax, xmax] = item.box_2d;
      const x = (xmin / 1000) * mediaW;
      const y = (ymin / 1000) * mediaH;
      const w = ((xmax - xmin) / 1000) * mediaW;
      const h = ((ymax - ymin) / 1000) * mediaH;
      return {
        id: `det-${index}-${item.label}`,
        label: formatObjectLabel(item.label),
        rawClass: item.label,
        confidence: Math.round(item.score * 100),
        score: item.score,
        bbox: [x, y, w, h],
      };
    });
  }

  async detectStaticImage(imgEl) {
    await this.ensureModelLoaded();
    if (this.useServerFallback || !this.model) {
      return this.detectViaServer(imgEl);
    }

    const predictions = await this.model.detect(imgEl, MAX_DETECTIONS, MIN_CONFIDENCE);
    return predictions
      .filter((p) => p.score >= MIN_CONFIDENCE)
      .sort((a, b) => b.score - a.score)
      .map((p, index) => ({
        id: `det-${index}-${p.class}`,
        label: formatObjectLabel(p.class),
        rawClass: p.class,
        confidence: Math.round(p.score * 100),
        score: p.score,
        bbox: [...p.bbox],
      }));
  }

  async detectCameraFrame(videoEl) {
    await this.ensureModelLoaded();
    if (this.useServerFallback || !this.model) {
      return this.detectViaServer(videoEl);
    }

    const predictions = await this.model.detect(videoEl, MAX_DETECTIONS, MIN_CONFIDENCE);
    const filtered = predictions
      .filter((p) => p.score >= MIN_CONFIDENCE)
      .sort((a, b) => b.score - a.score);

    const smoothed = filtered.map((pred, index) => {
      let bestMatch = null;
      let bestIoU = 0;

      for (const prev of this.previousCameraDetections) {
        if (prev.rawClass !== pred.class) continue;
        const iou = computeIoU(prev.bbox, pred.bbox);
        if (iou > 0.5 && iou > bestIoU) {
          bestIoU = iou;
          bestMatch = prev;
        }
      }

      const bbox = bestMatch
        ? pred.bbox.map((val, i) => bestMatch.bbox[i] * 0.28 + val * 0.72)
        : [...pred.bbox];

      return {
        id: `det-${index}-${pred.class}`,
        label: formatObjectLabel(pred.class),
        rawClass: pred.class,
        confidence: Math.round(pred.score * 100),
        score: pred.score,
        bbox,
      };
    });

    this.previousCameraDetections = smoothed;
    return smoothed;
  }
}

function drawTopRoundedRect(ctx, x, y, w, h, radius) {
  const r = Math.min(radius, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

export function renderDetections({ canvas, stageContainer, mediaEl, detections }) {
  if (!canvas || !stageContainer) return;

  const rect = stageContainer.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const targetWidth = Math.max(1, Math.round(rect.width * dpr));
  const targetHeight = Math.max(1, Math.round(rect.height * dpr));

  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, rect.height);

  if (!mediaEl || !detections || detections.length === 0) {
    ctx.restore();
    return;
  }

  const mediaW = mediaEl.videoWidth || mediaEl.naturalWidth || 0;
  const mediaH = mediaEl.videoHeight || mediaEl.naturalHeight || 0;
  if (!mediaW || !mediaH) {
    ctx.restore();
    return;
  }

  const mediaRect = getContainedMediaRect(rect.width, rect.height, mediaW, mediaH);
  const scaleX = mediaRect.width / mediaW;
  const scaleY = mediaRect.height / mediaH;

  const labelFont = '700 13px "Plus Jakarta Sans", -apple-system, sans-serif';
  const labelHeight = 26;
  const labelPadX = 10;

  detections.forEach((det, index) => {
    const [bx, by, bw, bh] = det.bbox;
    const x = mediaRect.x + bx * scaleX;
    const y = mediaRect.y + by * scaleY;
    const w = Math.max(6, bw * scaleX);
    const h = Math.max(6, bh * scaleY);

    const color = getDetectionColor(det.rawClass || det.label, index);

    ctx.save();

    // Crisp colored bounding rectangle
    ctx.lineWidth = 2.25;
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);

    // Connected rounded-top label tab directly above the box
    const labelText = `${det.label} ${det.confidence}%`;
    ctx.font = labelFont;
    const textWidth = ctx.measureText(labelText).width;
    const labelWidth = Math.ceil(textWidth + labelPadX * 2);

    const clampedX = Math.max(2, Math.min(x - 1, rect.width - labelWidth - 2));
    const placeInside = y - labelHeight < 4;
    const labelY = placeInside ? Math.max(2, y) : y - labelHeight;

    ctx.fillStyle = color;
    drawTopRoundedRect(ctx, clampedX, labelY, labelWidth, labelHeight, 6);
    ctx.fill();

    // Label text
    ctx.fillStyle = '#ffffff';
    ctx.textBaseline = 'middle';
    ctx.fillText(labelText, clampedX + labelPadX, labelY + labelHeight / 2 + 0.5);

    ctx.restore();
  });

  ctx.restore();
}
