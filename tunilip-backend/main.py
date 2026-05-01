"""
TUNILip+ Backend — FastAPI
Reproduit EXACTEMENT extract_videomae_features() du notebook Kaggle
Pipeline: vidéo → 16 frames 224×224 → VideoMAE frozen → mean-pool spatial → (8, 768)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import torch
import tempfile
import os
import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tunilip")

# ── Globals ──────────────────────────────────────────────────
vmae_processor = None
vmae_model     = None
DEVICE         = None
VMAE_MODEL_ID  = "MCG-NJU/videomae-base"
NUM_FRAMES     = 16   # même valeur que le notebook


# ══════════════════════════════════════════════════════════════
# STARTUP — charge VideoMAE une seule fois
# ══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vmae_processor, vmae_model, DEVICE

    logger.info(f"⏳ Chargement {VMAE_MODEL_ID} …")
    try:
        from transformers import VideoMAEModel, VideoMAEImageProcessor

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"   Device : {DEVICE}")

        vmae_processor = VideoMAEImageProcessor.from_pretrained(VMAE_MODEL_ID)
        vmae_model     = VideoMAEModel.from_pretrained(VMAE_MODEL_ID)
        vmae_model.eval()
        vmae_model = vmae_model.to(DEVICE)

        # Freeze tous les poids (extraction pure)
        for p in vmae_model.parameters():
            p.requires_grad = False

        n_params = sum(p.numel() for p in vmae_model.parameters())
        logger.info(f"✅ VideoMAE chargé — {n_params:,} params (TOUS GELÉS) sur {DEVICE}")

    except Exception as e:
        logger.error(f"❌ Impossible de charger VideoMAE : {e}")
        # On laisse le serveur démarrer quand même
        # /health retournera model_ready=False

    yield  # application en cours d'exécution

    logger.info("Shutdown — libération modèle")
    if vmae_model is not None:
        del vmae_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════
app = FastAPI(
    title="TUNILip+ Feature Extractor",
    description="Extrait des features VideoMAE (8, 768) depuis une vidéo de lecture labiale tunisienne",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restreindre en production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
# HELPER — extraction identique au notebook
# ══════════════════════════════════════════════════════════════
def extract_frames_224(video_path: str, num_frames: int = NUM_FRAMES):
    """
    Identique à extract_frames_224() du notebook.
    Retourne une liste de num_frames arrays uint8 (224, 224, 3).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo : {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise ValueError("Vidéo vide (0 frames)")

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    # padding si nécessaire
    while len(frames) < num_frames:
        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

    return frames[:num_frames]


@torch.no_grad()
def extract_videomae_features(video_path: str) -> np.ndarray:
    """
    Reproduit EXACTEMENT extract_videomae_features() du notebook.

    Pipeline :
      1. Extraire 16 frames 224×224
      2. vmae_processor → pixel_values
      3. vmae_model(**inputs) → last_hidden_state (1, 1568, 768)
      4. Reshape → (8 temporal × 196 spatial, 768)
      5. Mean-pool spatial → (8, 768)

    Retourne : np.ndarray float32 (8, 768)
    """
    if vmae_model is None or vmae_processor is None:
        raise RuntimeError("VideoMAE non chargé — serveur en cours d'initialisation")

    frames = extract_frames_224(video_path, NUM_FRAMES)

    # vmae_processor attend une liste de arrays uint8 (H, W, 3)
    inputs = vmae_processor(frames, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    out    = vmae_model(**inputs)
    # last_hidden_state : (1, 1568, 768)
    hidden = out.last_hidden_state.squeeze(0).cpu().numpy()  # (1568, 768)

    # VideoMAE-base : 8 temporal × 14×14 spatial = 8 × 196 = 1568
    T_temp = 8
    T_spat = 196
    hidden = hidden[: T_temp * T_spat].reshape(T_temp, T_spat, 768)
    hidden = hidden.mean(axis=1)   # mean-pool spatial → (8, 768)

    return hidden.astype(np.float32)


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": vmae_model is not None,
        "device": str(DEVICE) if DEVICE else "unknown",
        "model_id": VMAE_MODEL_ID,
    }


@app.post("/extract-features")
async def extract_features(video: UploadFile = File(...)):
    """
    Reçoit une vidéo (mp4 / webm / avi …),
    retourne les features VideoMAE shape (8, 768).

    Response JSON :
    {
        "features": [[f0_0, f0_1, …, f0_767], …, [f7_0, …, f7_767]],
        "shape": [8, 768],
        "model_id": "MCG-NJU/videomae-base"
    }
    """
    # Vérification type MIME
    allowed = {"video/mp4", "video/webm", "video/avi",
               "video/quicktime", "application/octet-stream"}
    if video.content_type and video.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Type de fichier non supporté : {video.content_type}"
        )

    # Sauvegarde temporaire
    suffix = os.path.splitext(video.filename or "video.mp4")[-1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        logger.info(f"Extraction features pour : {video.filename} ({len(content)/1024:.1f} KB)")
        features = extract_videomae_features(tmp_path)
        logger.info(f"✅ Features extraites — shape {features.shape}")

        return JSONResponse({
            "features": features.tolist(),   # liste Python → JSON
            "shape": list(features.shape),   # [8, 768]
            "model_id": VMAE_MODEL_ID,
        })

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur extraction : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.get("/")
def root():
    return {
        "service": "TUNILip+ VideoMAE Feature Extractor",
        "endpoints": {
            "POST /extract-features": "Envoie une vidéo, reçoit (8, 768) features",
            "GET  /health":           "Statut du serveur et du modèle",
        }
    }
