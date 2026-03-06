# Vultus: Advanced Face Detection System

Vultus (Latin for *face*) is a comprehensive, full-stack facial detection and recognition platform designed for educational and high-security environments. The system integrates a modern React dashboard, a robust Node.js backend, and a high-performance Python-based machine learning inference engine.

---

## Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [The Three Models](#-the-three-models)
  - [1. Face Recognition Model (Custom Trained)](#1-face-recognition-model-custom-trained)
  - [2. Face Detection Model (YOLOv8-Face)](#2-face-detection-model-yolov8-face)
  - [3. Emotion Classification Model (FER+)](#3-emotion-classification-model-fer)
- [Production Pipeline](#-production-pipeline)
- [Getting Started](#-getting-started)
- [Model Management](#-model-management)
- [API Reference](#-api-reference)
- [Development & Testing](#-development--testing)

---

## 🌟 Key Features

### Teacher Dashboard
- **Emotion Snapshot**: Real-time visualization of classroom sentiment (angry, contempt, disgust, fear, happy, neutral, sad, surprise).
- **Access Integrity**: Monitor entry/exit grant rates and security alerts.
- **Attendance Tracking**: Historical data analysis (24h, 7d, 30d, 365d).
- **Live Event Log**: Real-time stream of access events and decisions.

### Admin Dashboard
- **Student Enrollment**: Register students by uploading reference photos. The system automatically generates 512-d face embeddings.
- **Live Camera Feed**: Real-time surveillance with **auto-scanning** every 1.4 seconds.
- **Manual Override**: Ability to force-grant access for denied attempts.
- **Model Sandbox**: Test specific ML endpoints (Face Detection, Face Recognition, Emotion) independently.
- **System Health**: Monitor uptime and global event counters.

### Advanced ML Engine
- **Multi-Model Pipeline**:
  - **Face Detection**: YOLOv8-Face (PyTorch) for locating faces in frames.
  - **Face Recognition**: Custom IResNet + ArcFace backbone for 512-d embeddings.
  - **Emotion Analysis**: FER+ (ONNX) for 8-class emotion classification.
- **Vector Search**: Cosine similarity for fast identity matching against enrolled embeddings.

---

## 🏗 System Architecture

The project is architected as containerized microservices:

| Service     | Technology                         | Port | Role                                              |
|-------------|-------------------------------------|------|---------------------------------------------------|
| `vultus-web`  | React 18 + Vite, Tailwind CSS, Framer Motion | 5173 | Frontend dashboards (Admin, Teacher)               |
| `vultus-api`  | Node.js Express                      | 4000 | Auth, business logic, orchestration, DB access     |
| `vultus-ml`   | Python FastAPI, PyTorch, ONNX Runtime | 8000 | ML inference (face detection, embeddings, emotion) |
| `postgres`    | PostgreSQL 16                        | 5432 | Users, students, face_embeddings, access_events, emotion_events |

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  vultus-web │────▶│  vultus-api │────▶│  vultus-ml  │     │   postgres  │
│  (React)    │     │  (Express)  │     │  (FastAPI)  │     │             │
└─────────────┘     └──────┬──────┘     └─────────────┘     └──────▲──────┘
                           │                      │                │
                           └──────────────────────┴────────────────┘
```

---

## 🧠 The Three Models

### 1. Face Recognition Model (Custom Trained)

**File:** `backbone_best.pt`  
**Role:** Generate 512-dimensional L2-normalized face embeddings for identity matching.

#### Architecture
- **Backbone:** IResNet (InsightFace-style IR-SE with Squeeze-and-Excitation attention)
- **Variants:** IR-SE-100 (`iresnet100`) or IR-SE-200 (`iresnet200`) — auto-detected from checkpoint
- **Input:** 112×112 RGB, normalized to [-1, 1]
- **Output:** 512-d float32 vector (L2-normalized)

#### Training (`train.py`)

| Aspect              | Detail                                                                 |
|---------------------|-----------------------------------------------------------------------|
| **Dataset**         | VGGFace2 (`vggface2_112x112/`) — 8631 train identities, 500 val       |
| **Structure**       | `id_0/`, `id_1/`, … with `0.jpg`, `1.jpg`, etc. per identity          |
| **Loss**            | ArcFace (Additive Angular Margin) with dynamic margin ramp            |
| **Margin schedule** | 0.2 → 0.50 over 12 epochs                                            |
| **Scale (s)**       | 64 (fixed)                                                            |
| **Optimizer**       | SGD (Nesterov), lr=0.05, weight_decay=5e-4, momentum=0.9              |
| **LR schedule**     | 3-epoch warmup, then cosine decay to 1e-5                             |
| **Batch sampling**  | P-K sampling (P identities × K images per batch)                      |
| **Validation**      | Fixed evaluation indices (reproducible epoch-to-epoch comparison)     |
| **Early stopping**  | Patience 10 epochs on ROC AUC                                         |
| **Distributed**     | DDP (NCCL) for multi-GPU                                              |
| **Mixed precision** | AMP (FP16) when CUDA available                                        |

#### Training command
```bash
python train.py --data_root /path/to/vggface2_112x112 --save_dir ./checkpoints
```

#### Post-training benchmark evaluation
```bash
python evaluate_benchmarks.py \
  --checkpoint ./checkpoints/backbone_best.pt \
  --lfw_path /path/to/lfw \
  --cfp_fp_path /path/to/cfp_fp
```

---

### 2. Face Detection Model (YOLOv8-Face)

**File:** `yolov8l_100e.pt`  
**Role:** Detect face bounding boxes in camera frames (used as the first stage before recognition).

#### Source
- **Project:** [YOLOv8-Face](https://github.com/yusepp/yolov8-face) (also: [lindevs/yolov8-face](https://github.com/lindevs/yolov8-face), [andrisan/yolov8-face](https://github.com/andrisan/yolov8-face))
- **Training:** YOLOv8-large variant trained on WIDERFace for 100 epochs (`yolov8l_100e`)
- **Format:** PyTorch (`.pt`), loaded via Ultralytics `YOLO` API

#### Specifications
- **Input:** 640×640 (configurable via `FACE_DETECT_INPUT_SIZE`)
- **Output:** Face bounding boxes `[x1, y1, x2, y2]` + confidence
- **Confidence threshold:** 0.35 (configurable via `FACE_DETECT_CONFIDENCE`)

#### Note on naming
The API endpoint is named `/predict/person` for historical compatibility, but it performs **face detection** (not full-body person detection). It returns the best face bbox when a face is present.

---

### 3. Emotion Classification Model (FER+)

**File:** `emotion-ferplus-8.onnx`  
**Role:** Classify facial expressions into 8 emotions from a cropped face.

#### Source
- **Model:** FER+ (Facial Expression Recognition Plus)
- **Origin:** [Microsoft FER+](https://github.com/microsoft/FERPlus), [ONNX Model Zoo](https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus)
- **Hugging Face:** [onnxmodelzoo/emotion-ferplus-8](https://huggingface.co/onnxmodelzoo/emotion-ferplus-8)
- **Format:** ONNX (Opset 8)

#### Specifications
- **Input:** 64×64 grayscale, normalized [0, 1]
- **Output:** 8 scores → softmax → probabilities
- **Classes:** `angry`, `contempt`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

---

## 🔄 Production Pipeline

The three models are chained into a single production-ready pipeline.

### Flow

```
Camera frame (base64)
        │
        ▼
┌───────────────────────────────────┐
│  1. Face Detection (YOLOv8-Face)  │  ← Fast gate: no face → skip rest
└───────────────────────────────────┘
        │
        │  face bbox [x1,y1,x2,y2]
        ▼
┌───────────────────────────────────┐
│  2. Face Crop                     │  Add padding, make square
└───────────────────────────────────┘
        │
        │  cropped face (112×112)
        ▼
┌───────────────────────────────────┐
│  3. Face Recognition (IResNet)    │  → 512-d embedding
└───────────────────────────────────┘
        │
        │  embedding
        ▼
┌───────────────────────────────────┐
│  4. Vector Match (vultus-api)     │  Cosine similarity vs enrolled embeddings
└───────────────────────────────────┘
        │
        │  match ≥ threshold → "granted"
        ▼
┌───────────────────────────────────┐
│  5. Emotion (FER+)                │  Only if access granted
└───────────────────────────────────┘
        │
        │  emotion + confidence
        ▼
  access_events + emotion_events
```

### End-to-end behavior

1. **Admin camera** captures frames; every 1.4s the frontend sends a frame to the API.
2. **Step 1:** `POST /api/detections/person` → YOLOv8-Face. If no face detected, stop.
3. **Step 2:** `POST /api/detections/face` → YOLOv8-Face + crop + IResNet embedding. API matches embedding against `face_embeddings` via cosine similarity.
4. **Step 3:** If decision is `granted`, API calls `POST /api/detections/emotion` with the face crop; FER+ returns emotion; result stored in `emotion_events`.

### Thresholds

- **Face match:** `FACE_MATCH_THRESHOLD` (default 0.4) — cosine similarity above this → grant access.
- **Face detection:** `FACE_DETECT_CONFIDENCE` (default 0.35) — minimum YOLOv8-Face confidence.

---

## 🚀 Getting Started

### Prerequisites

- Docker & Docker Compose
- Node.js 20+ (for local backend/frontend development)
- Python 3.11+ (for local ML development)

### Model files

Place these in the project root before running:

| File                   | Description                          |
|------------------------|--------------------------------------|
| `backbone_best.pt`     | Custom face recognition (IResNet)    |
| `yolov8l_100e.pt`      | YOLOv8-Face detection                |
| `emotion-ferplus-8.onnx` | FER+ emotion classification        |

### Run with Docker

```bash
git clone <repository-url>
cd FaceDetection
docker-compose up --build
```

**Access:**
- **Frontend:** http://localhost:5173
- **API:** http://localhost:4000
- **ML service:** http://localhost:8000

**Default login:**
- Email: `admin@vultus.ai`
- Password: `Admin@123`

---

## 🧠 Model Management

### Environment variables (vultus-ml)

| Variable                 | Default                    | Description                  |
|--------------------------|----------------------------|------------------------------|
| `FACE_MODEL_PATH`        | `./backbone_best.pt`       | Face recognition checkpoint  |
| `FACE_DETECT_MODEL_PATH` | `./yolov8l_100e.pt`        | Face detection model         |
| `EMOTION_MODEL_PATH`     | `./emotion-ferplus-8.onnx` | Emotion ONNX model           |
| `FACE_INPUT_SIZE`        | `112`                      | Input size for IResNet       |
| `FACE_DETECT_INPUT_SIZE` | `640`                      | Input size for YOLOv8-Face   |
| `EMOTION_INPUT_SIZE`     | `64`                       | Input size for FER+          |
| `FACE_DETECT_CONFIDENCE` | `0.35`                     | Detection confidence threshold |

### Updating models

1. Replace the corresponding file in the project root.
2. Restart the ML service:
   ```bash
   docker-compose restart vultus-ml
   ```

---

## 📡 API Reference

### Authentication
- `POST /api/auth/login` — Admin/Teacher login
- `POST /api/auth/register` — Create users (Admin only)

### Students & Enrollment
- `POST /api/students/enroll` — Register student with reference images (generates embeddings)
- `GET /api/students` — List students with access/emotion stats
- `DELETE /api/students/:id` — Remove student and embeddings

### Detections (ML pipeline)
- `POST /api/detections/face` — Full pipeline: Detect → Crop → Embed → Match → [Emotion if granted]
- `POST /api/detections/person` — Face detection only (YOLOv8-Face)
- `POST /api/detections/emotion` — Emotion only (requires face crop)

### ML service (vultus-ml)
- `GET /health` — Model load status
- `POST /predict/face` — Full face pipeline (detect + embed)
- `POST /predict/person` — Face detection
- `POST /predict/embedding` — Embedding from face crop
- `POST /predict/emotion` — Emotion from face crop

---

## 🧪 Development & Testing

### Local ML service

```bash
cd vultus-ml
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

### Operations

| Action          | Command                    |
|-----------------|----------------------------|
| Start (background) | `docker-compose up -d`  |
| View logs       | `docker-compose logs -f`   |
| Stop            | `docker-compose stop`      |
| Full reset      | `docker-compose down -v`   |

### Database

- Schema and seed data: `vultus-api/db/init.sql`
- Tables: `users`, `students`, `face_embeddings`, `access_events`, `emotion_events`, `system_metrics`

---

## 📄 License

This project is private. All rights reserved.
