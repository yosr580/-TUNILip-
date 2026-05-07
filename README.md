<div align="center">

<img src="https://img.shields.io/badge/TUNILip%2B-Tunisian%20Lip%20Reading-0cf2c8?style=for-the-badge&labelColor=080b12" alt="TUNILip+"/>

# TUNILip+ — Tunisian Dialect Lip Reading

**The first automated lip reading system dedicated to Tunisian Arabic**

[![Live Demo](https://img.shields.io/badge/▶%20Live%20Demo-Vercel-0cf2c8?style=flat-square&logo=vercel&logoColor=white)](https://tuni-lip-e4li.vercel.app/)
[![HuggingFace Backend](https://img.shields.io/badge/🤗%20Backend-HuggingFace%20Spaces-fbbf24?style=flat-square)](https://huggingface.co/spaces/yossss2/tunilip-backend/tree/main)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)]()
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![TensorFlow.js](https://img.shields.io/badge/TF.js-4.15.0-FF6F00?style=flat-square&logo=tensorflow)](https://www.tensorflow.org/js)

> ♥ Dedicated to Al Razi Hospital & the Deaf and Hard-of-Hearing Community

</div>

---

## 🧠 What is TUNILip+?

TUNILip+ is a real-time visual speech recognition system that reads lip movements and predicts spoken words in the **Tunisian Arabic dialect**. It combines:

- **CNN spatial encoding** to extract mouth region features frame-by-frame
- **BiLSTM temporal modeling** with a custom **Temporal Attention** layer to capture movement patterns over time
- **VideoMAE** (Video Masked Autoencoder) as a powerful feature extractor on the backend
- **MediaPipe Face Mesh** for accurate, real-time mouth landmark detection in the browser

The entire inference pipeline runs **in the browser** via TensorFlow.js — no server needed after the initial model load.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     FRONTEND (Vercel)                    │
│                                                          │
│  Webcam / Video File                                     │
│       │                                                  │
│       ▼                                                  │
│  MediaPipe Face Mesh (468 landmarks)                     │
│       │  → mouth ROI crop (40 landmarks)                 │
│       ▼                                                  │
│  Frame Sequence (16 frames @ 224×224)                    │
│       │                                                  │
│       ├──── Local TF.js Model (CNN + BiLSTM) ────────────┤
│       │         └── TemporalAttention layer              │
│       │                  └── softmax → 13 classes        │
│       │                                                  │
│       └──── (Optional) Backend API ─────────────────────┤
│                                                          │
└──────────────────────────────────────────────────────────┘
                          │
              ┌───────────▼───────────┐
              │   BACKEND (HF Space)  │
              │                       │
              │  FastAPI + VideoMAE   │
              │  MCG-NJU/videomae-base│
              │  16 frames → (8, 768) │
              │  feature vectors      │
              └───────────────────────┘
```

---

## 🗣️ Vocabulary — 13 Tunisian Words

| # | Tunisian | Arabic | English |
|---|----------|--------|---------|
| 001 | **3asslema** | عسلامة | Hello / I greet you |
| 002 | **Aatchan** | عطشان | I am thirsty |
| 003 | **Ghattini** | غطيني | Cover me |
| 004 | **Hamdoullah** | الحمد لله | Thank God |
| 005 | **Inshallah** | إن شاء الله | God willing |
| 006 | **Jiian** | جيعان | I am hungry |
| 007 | **Mahsour** | محصور | I need the toilet |
| 008 | **Mawjoue** | موجوع | I am in pain |
| 009 | **Metkallak** | متقلق | I am worried |
| 010 | **Nadhafli** | نظفلي | Clean this for me |
| 011 | **Skhont** | سخنت | I am hot |
| 012 | **Yaychek** | يعيشك | Thank you |
| 013 | **Yezzini** | يزيني | That is enough / Stop |

> 📊 Dataset: 50+ volunteer speakers · 650+ video recordings

---

## 📁 Repository Structure

```
TUNILip+/
├── index.html                  # Single-page frontend app (TF.js inference)
├── vercel.json                 # Vercel deployment config (CORS headers for model files)
│
├── model/
│   ├── model.json              # TF.js model architecture
│   └── group1-shard1of1.bin    # Model weights (~2.5 MB)
│
└── tunilip-backend/            # FastAPI backend (HuggingFace Spaces)
    ├── main.py                 # VideoMAE feature extraction API
    ├── requirements.txt        # Python dependencies
    ├── Procfile                # Process entry point
    ├── render.yaml             # Render.com deployment config
    └── .python-version         # Python 3.11.0
```

---

## 🚀 Getting Started

### Frontend (local)

The frontend is a single `index.html` — no build step required.

```bash
git clone https://github.com/your-username/TUNILip-
cd TUNILip-

# Serve locally (the model files require HTTP — can't open index.html directly)
python -m http.server 8080
# or
npx serve .
```

Open `http://localhost:8080` in your browser.

> ⚠️ The `model/` directory contains Git LFS files. Make sure Git LFS is installed (`git lfs install`) before cloning, or the `.bin` weights file will be a pointer stub.

### Backend (local)

```bash
cd tunilip-backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

API docs available at `http://localhost:7860/docs`

> ⏳ First startup downloads **VideoMAE** (~350 MB) from HuggingFace Hub. Subsequent starts take ~30s.

---

## 🔌 Backend API

Hosted on HuggingFace Spaces: `https://yossss2-tunilip-backend.hf.space`

### Endpoints

#### `GET /health`
Returns model readiness status.
```json
{
  "status": "ok",
  "model_ready": true,
  "device": "cpu",
  "model_id": "MCG-NJU/videomae-base"
}
```

#### `POST /extract-features`
Upload a video file, receive VideoMAE feature vectors.

```bash
curl -X POST https://yossss2-tunilip-backend.hf.space/extract-features \
  -F "video=@your_video.mp4"
```

**Response:**
```json
{
  "features": [[...], [...], ...],   // shape (8, 768)
  "shape": [8, 768],
  "model_id": "MCG-NJU/videomae-base"
}
```

**Supported formats:** `mp4`, `webm`, `avi`, `mov`

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla HTML/CSS/JS, TensorFlow.js 4.15.0 |
| Face Detection | MediaPipe Face Mesh (468 landmarks) |
| ML Model (browser) | CNN + BiLSTM + TemporalAttention (TF.js) |
| ML Model (backend) | VideoMAE (`MCG-NJU/videomae-base`, 86M params) |
| Backend Framework | FastAPI + Uvicorn |
| Backend Hosting | HuggingFace Spaces |
| Frontend Hosting | Vercel |
| Video Processing | OpenCV (backend), Canvas API (frontend) |

---

## 🧪 How the Inference Pipeline Works

1. **Mouth Detection** — MediaPipe Face Mesh extracts 40 mouth landmark points from each video frame
2. **ROI Crop** — The mouth region is cropped and resized to 224×224
3. **Frame Sampling** — 16 evenly-spaced frames are sampled from the input video
4. **Feature Extraction** — Either local CNN processes the frames, or VideoMAE on the backend extracts `(8, 768)` temporal features
5. **Temporal Modeling** — BiLSTM processes the frame sequence, capturing motion dynamics
6. **Attention** — A custom `TemporalAttention` layer weights the most informative frames
7. **Classification** — Softmax output over 13 Tunisian word classes

---

## 📦 Backend Dependencies

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-multipart==0.0.9
transformers>=4.44.2
torch>=2.9.0
torchvision>=0.19.0
opencv-python-headless>=4.10.0
numpy>=1.26.0
```

---

## 🤝 Contributing

This project was built to support the **Tunisian deaf and hard-of-hearing community**. Contributions are welcome:

- 🎥 **Donate video recordings** of yourself saying the vocabulary words
- 🐛 **Report bugs** via GitHub Issues
- 🌍 **Expand vocabulary** — more words, more impact
- 🧪 **Improve model accuracy** — PRs for architecture improvements welcome

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ in Tunisia · Dedicated to Al Razi Hospital & the Deaf and Hard-of-Hearing Community

</div>
