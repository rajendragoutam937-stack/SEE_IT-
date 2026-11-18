import { initThemeToggle } from './theme.js';
import { VisionDetector, renderDetections } from './detector.js';
import { ChatController } from './chat.js';

const SUPPORTED_IMAGE_TYPES = new Set(['image/jpeg', 'image/jpg', 'image/png', 'image/webp']);
const SUPPORTED_EXTENSIONS = /\.(jpe?g|png|webp)$/i;

const dom = {
  themeLightBtn: document.getElementById('theme-light-btn'),
  themeDarkBtn: document.getElementById('theme-dark-btn'),
  apiStatusDot: document.getElementById('api-status-dot'),
  apiStatusText: document.getElementById('api-status-text'),

  stageContainer: document.getElementById('stage-container'),
  emptyState: document.getElementById('empty-state'),
  emptyCameraBtn: document.getElementById('empty-camera-btn'),
  emptyUploadBtn: document.getElementById('empty-upload-btn'),

  previewImage: document.getElementById('preview-image'),
  cameraVideo: document.getElementById('camera-video'),
  detectionCanvas: document.getElementById('detection-canvas'),

  stageStatusPill: document.getElementById('stage-status-pill'),
  stageStatusDot: document.getElementById('stage-status-dot'),
  stageStatusText: document.getElementById('stage-status-text'),
  stageTopActions: document.getElementById('stage-top-actions'),
  overlayStopBtn: document.getElementById('overlay-stop-btn'),
  switchCameraBtn: document.getElementById('switch-camera-btn'),

  stageLoading: document.getElementById('stage-loading'),
  stageLoadingText: document.getElementById('stage-loading-text'),
  stageError: document.getElementById('stage-error'),
  stageErrorTitle: document.getElementById('stage-error-title'),
  stageErrorMessage: document.getElementById('stage-error-message'),
  stageErrorDismiss: document.getElementById('stage-error-dismiss'),

  cameraBtn: document.getElementById('camera-btn'),
  cameraBtnLabel: document.getElementById('camera-btn-label'),
  uploadBtn: document.getElementById('upload-btn'),
  fileInput: document.getElementById('file-input'),

  chatMessages: document.getElementById('chat-messages'),
  chatForm: document.getElementById('chat-form'),
  chatInput: document.getElementById('chat-input'),
  sendBtn: document.getElementById('send-btn'),
  clearChatBtn: document.getElementById('clear-chat-btn'),
};

const state = {
  mode: 'idle', // 'idle' | 'image' | 'camera'
  detections: [],
  mediaStream: null,
  facingMode: 'environment',
  cameraLoopActive: false,
  rafId: null,
  isInferencing: false,
  lastServerFallbackTime: 0,
  objectUrl: null,
};

const detector = new VisionDetector();

function getActiveMediaElement() {
  if (state.mode === 'camera') return dom.cameraVideo;
  if (state.mode === 'image') return dom.previewImage;
  return null;
}

function redrawCanvas() {
  renderDetections({
    canvas: dom.detectionCanvas,
    stageContainer: dom.stageContainer,
    mediaEl: getActiveMediaElement(),
    detections: state.detections,
  });
}

function showLoading(message) {
  dom.stageLoadingText.textContent = message;
  dom.stageLoading.hidden = false;
}

function hideLoading() {
  dom.stageLoading.hidden = true;
}

function showError(title, message) {
  dom.stageErrorTitle.textContent = title;
  dom.stageErrorMessage.textContent = message;
  dom.stageError.hidden = false;
}

function hideError() {
  dom.stageError.hidden = true;
}

function updateStageOverlays() {
  if (state.mode === 'camera') {
    dom.stageStatusPill.hidden = false;
    dom.stageStatusDot.className = 'status-dot status-dot-green';
    dom.stageStatusText.textContent = 'Camera Active';
    dom.stageTopActions.hidden = false;
    dom.overlayStopBtn.hidden = false;
  } else if (state.mode === 'image') {
    const count = state.detections.length;
    dom.stageStatusPill.hidden = false;
    dom.stageStatusDot.className = 'status-dot status-dot-blue';
    dom.stageStatusText.textContent =
      count === 0 ? 'Image Loaded' : count === 1 ? '1 Object Detected' : `${count} Objects Detected`;
    dom.stageTopActions.hidden = true;
    dom.overlayStopBtn.hidden = true;
    dom.switchCameraBtn.hidden = true;
  } else {
    dom.stageStatusPill.hidden = true;
    dom.stageTopActions.hidden = true;
    dom.overlayStopBtn.hidden = true;
    dom.switchCameraBtn.hidden = true;
  }
}

function updateCameraButtonUI(active) {
  if (!dom.cameraBtn || !dom.cameraBtnLabel) return;
  if (active) {
    dom.cameraBtn.classList.remove('btn-blue');
    dom.cameraBtn.classList.add('btn-stop-action');
    dom.cameraBtnLabel.textContent = 'Stop Camera';
  } else {
    dom.cameraBtn.classList.remove('btn-stop-action');
    dom.cameraBtn.classList.add('btn-blue');
    dom.cameraBtnLabel.textContent = 'Open Camera';
  }
}

function stopCameraStream({ keepModeIfImage = false } = {}) {
  state.cameraLoopActive = false;
  if (state.rafId !== null) {
    cancelAnimationFrame(state.rafId);
    state.rafId = null;
  }
  state.isInferencing = false;
  detector.resetTracking();

  if (state.mediaStream) {
    for (const track of state.mediaStream.getTracks()) {
      track.stop();
    }
    state.mediaStream = null;
  }

  dom.cameraVideo.srcObject = null;
  dom.cameraVideo.hidden = true;
  dom.switchCameraBtn.hidden = true;
  updateCameraButtonUI(false);

  if (!keepModeIfImage && state.mode === 'camera') {
    state.mode = 'idle';
    state.detections = [];
    dom.emptyState.hidden = false;
    redrawCanvas();
    updateStageOverlays();
  }
}

async function checkMultipleCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter((d) => d.kind === 'videoinput');
    dom.switchCameraBtn.hidden = !(state.mode === 'camera' && videoInputs.length > 1);
  } catch {
    dom.switchCameraBtn.hidden = true;
  }
}

function runCameraDetectionLoop() {
  if (!state.cameraLoopActive || state.mode !== 'camera') return;

  const loop = async () => {
    if (!state.cameraLoopActive || state.mode !== 'camera') return;

    const video = dom.cameraVideo;
    if (video.readyState >= 2 && video.videoWidth > 0 && !state.isInferencing) {
      const now = performance.now();
      const canRunThisFrame = !detector.useServerFallback || now - state.lastServerFallbackTime > 1200;

      if (canRunThisFrame) {
        state.isInferencing = true;
        if (detector.useServerFallback) {
          state.lastServerFallbackTime = now;
        }

        try {
          const detections = await detector.detectCameraFrame(video);
          if (state.cameraLoopActive && state.mode === 'camera') {
            state.detections = detections;
            redrawCanvas();
          }
        } catch (err) {
          console.error('Camera frame detection error:', err);
        } finally {
          state.isInferencing = false;
        }
      }
    }

    redrawCanvas();
    if (state.cameraLoopActive && state.mode === 'camera') {
      state.rafId = requestAnimationFrame(loop);
    }
  };

  state.rafId = requestAnimationFrame(loop);
}

async function startCamera(facing = state.facingMode) {
  hideError();

  if (!navigator.mediaDevices?.getUserMedia) {
    showError(
      'Camera Unavailable',
      'Your browser does not support live camera capture. Try uploading a picture instead.'
    );
    return;
  }

  stopCameraStream({ keepModeIfImage: false });

  dom.previewImage.hidden = true;
  if (state.objectUrl) {
    URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;
  }
  dom.previewImage.removeAttribute('src');

  showLoading('Requesting camera access and loading vision model...');

  try {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: facing },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
    } catch {
      stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
    }

    state.mediaStream = stream;
    state.facingMode = facing;
    state.mode = 'camera';
    state.detections = [];

    dom.emptyState.hidden = true;
    dom.cameraVideo.srcObject = stream;
    dom.cameraVideo.hidden = false;
    updateCameraButtonUI(true);
    updateStageOverlays();

    await new Promise((resolve) => {
      if (dom.cameraVideo.readyState >= 1) {
        resolve();
      } else {
        dom.cameraVideo.onloadedmetadata = () => resolve();
      }
    });

    await dom.cameraVideo.play();
    await detector.ensureModelLoaded();
    hideLoading();

    await checkMultipleCameras();
    state.cameraLoopActive = true;
    runCameraDetectionLoop();
  } catch (error) {
    hideLoading();
    stopCameraStream();

    const name = error?.name || '';
    if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
      showError(
        'Camera Permission Denied',
        'Allow camera access in your browser settings to run live object detection, or upload an image.'
      );
    } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
      showError(
        'No Camera Found',
        'No working camera was detected on this device. You can upload a picture to analyze objects instead.'
      );
    } else if (name === 'NotReadableError' || name === 'TrackStartError') {
      showError(
        'Camera In Use',
        'Your camera appears to be in use by another application. Close the other app and try again.'
      );
    } else {
      showError(
        'Camera Error',
        error instanceof Error ? error.message : 'Unable to start the camera feed.'
      );
    }
  }
}

async function handleImageFile(file) {
  if (!file) return;
  hideError();

  const validType = SUPPORTED_IMAGE_TYPES.has(file.type.toLowerCase()) || SUPPORTED_EXTENSIONS.test(file.name || '');
  if (!validType) {
    showError('Unsupported Format', 'Please upload a JPG, JPEG, PNG, or WEBP image.');
    return;
  }

  stopCameraStream({ keepModeIfImage: true });

  if (state.objectUrl) {
    URL.revokeObjectURL(state.objectUrl);
  }
  const objectUrl = URL.createObjectURL(file);
  state.objectUrl = objectUrl;

  state.mode = 'image';
  state.detections = [];

  dom.emptyState.hidden = true;
  dom.cameraVideo.hidden = true;
  dom.previewImage.removeAttribute('src');
  dom.previewImage.hidden = false;
  redrawCanvas();
  updateStageOverlays();

  showLoading('Analyzing image...');

  try {
    await new Promise((resolve, reject) => {
      dom.previewImage.onload = () => resolve();
      dom.previewImage.onerror = () => reject(new Error('Failed to decode the uploaded image file.'));
      dom.previewImage.src = objectUrl;
    });

    const detections = await detector.detectStaticImage(dom.previewImage);
    state.detections = detections;
    hideLoading();
    redrawCanvas();
    updateStageOverlays();
  } catch (error) {
    hideLoading();
    showError(
      'Detection Failed',
      error instanceof Error ? error.message : 'Unable to run object detection on this image.'
    );
  }
}

function triggerFilePicker() {
  if (dom.fileInput) {
    dom.fileInput.value = '';
    dom.fileInput.click();
  }
}

async function checkApiHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    if (data?.aiConfigured || data?.geminiConfigured) {
      dom.apiStatusDot.className = 'status-dot status-dot-green';
      dom.apiStatusText.textContent = 'Connected';
      dom.apiStatusText.classList.remove('is-offline');
    } else {
      dom.apiStatusDot.className = 'status-dot status-dot-red';
      dom.apiStatusText.textContent = 'Key Missing';
      dom.apiStatusText.classList.add('is-offline');
    }
  } catch {
    dom.apiStatusDot.className = 'status-dot status-dot-red';
    dom.apiStatusText.textContent = 'Offline';
    dom.apiStatusText.classList.add('is-offline');
  }
}

new ChatController({
  messagesEl: dom.chatMessages,
  formEl: dom.chatForm,
  inputEl: dom.chatInput,
  sendBtn: dom.sendBtn,
  clearBtn: dom.clearChatBtn,
  getVisualContext: () => {
    const activeMedia = getActiveMediaElement();
    const imageDataUrl = activeMedia ? detector.captureSnapshotDataUrl(activeMedia, 384, 0.65) : null;
    return {
      mediaMode: state.mode,
      detectedObjects: state.detections.map((d) => ({
        label: d.label,
        confidence: d.confidence,
      })),
      imageDataUrl,
    };
  },
});

initThemeToggle({
  lightBtn: dom.themeLightBtn,
  darkBtn: dom.themeDarkBtn,
  onThemeChange: () => {
    redrawCanvas();
  },
});

dom.uploadBtn?.addEventListener('click', triggerFilePicker);
dom.emptyUploadBtn?.addEventListener('click', triggerFilePicker);

dom.fileInput?.addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (file) handleImageFile(file);
});

dom.cameraBtn?.addEventListener('click', () => {
  if (state.mode === 'camera') {
    stopCameraStream();
  } else {
    startCamera(state.facingMode);
  }
});

dom.emptyCameraBtn?.addEventListener('click', () => {
  startCamera(state.facingMode);
});

dom.overlayStopBtn?.addEventListener('click', () => {
  stopCameraStream();
});

dom.switchCameraBtn?.addEventListener('click', () => {
  if (state.mode !== 'camera') return;
  const nextFacing = state.facingMode === 'environment' ? 'user' : 'environment';
  startCamera(nextFacing);
});

dom.stageErrorDismiss?.addEventListener('click', hideError);

dom.stageContainer?.addEventListener('dragover', (e) => {
  e.preventDefault();
  dom.stageContainer.classList.add('drag-over');
});

dom.stageContainer?.addEventListener('dragleave', () => {
  dom.stageContainer.classList.remove('drag-over');
});

dom.stageContainer?.addEventListener('drop', (e) => {
  e.preventDefault();
  dom.stageContainer.classList.remove('drag-over');
  const file = e.dataTransfer?.files?.[0];
  if (file) handleImageFile(file);
});

if (typeof ResizeObserver !== 'undefined' && dom.stageContainer) {
  const observer = new ResizeObserver(() => {
    redrawCanvas();
  });
  observer.observe(dom.stageContainer);
}
window.addEventListener('resize', redrawCanvas);

window.addEventListener('load', () => {
  checkApiHealth();
  detector.ensureModelLoaded().catch(() => {});
});
