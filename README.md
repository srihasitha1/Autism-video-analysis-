# AutiSense

Early autism screening platform combining video behavioral analysis and parental questionnaire scoring into a fused risk assessment.

---

## What this does

A parent uploads a short video of their child playing and completes a 40-question behavioral questionnaire. Two independent ML models process each input separately — a MobileNet-GRU video model classifies on-screen behaviors, and an ensemble of five scikit-learn classifiers scores the questionnaire responses. A fusion algorithm combines both scores using dynamic weighting driven by the video model's prediction confidence. The result is a risk band (low, medium, or high), a breakdown of contributing behavioral signals, an AI-generated plain-language summary, a downloadable report, and a map of nearby autism clinics.

---

## Disclaimer

> AutiSense is an early screening tool only. It does not provide a medical
> diagnosis. All results should be reviewed by a qualified pediatrician or
> child psychiatrist. Early screening indicators do not confirm or rule out
> autism spectrum disorder.

---

## Architecture overview

The application is composed of six Docker services coordinated by `docker-compose.yml`. The React frontend sends REST API calls to the FastAPI backend. The backend delegates long-running video inference to a Celery worker via a Redis message queue. Both the backend and worker share a PostgreSQL database. ML model files are not baked into any Docker image — they are mounted from the host machine at runtime.

```
Browser
  │
  ▼
React Frontend (port 5173)
  │  REST API calls
  ▼
FastAPI Backend (port 8000)
  │                    │
  │ async DB writes     │ enqueue task
  ▼                    ▼
PostgreSQL ◄──── Celery Worker
                       │
                  Redis (broker + result backend)
                       │
                  ML Models (volume mount: ./model → /app/model)
                       │
                  autism_final.h5        (video classification)
                  label_encoder.pkl      (video class labels)
                  autism_model.pkl       (questionnaire ensemble)

External APIs (called directly from the frontend):
  Groq API (llama-3.3-70b-versatile) ──► AI summaries + chatbot responses
  Geoapify Places API                 ──► Nearby clinic search
  Geoapify Geocoding API              ──► Convert typed location to coordinates
  Geoapify Static Maps API            ──► Render map image of clinic results
```

### Component details

**Frontend — React + Vite (port 5173)**
Single-page application served by a Vite dev server inside Node 20. All screen state and navigation logic lives in `AppContent.jsx`. Three post-analysis components handle results display:
- `PostAnalysisResult.jsx` — renders the fused risk score, behavioral breakdowns, and a downloadable plain-text report
- `LocationSpecialistWidget.jsx` — Geoapify-powered clinic finder with browser geolocation, manual address entry, and a static map
- The chatbot is embedded directly inside `AppContent.jsx` using Groq's API

**Backend — FastAPI (port 8000)**
Async REST API with four router groups: `auth`, `video`, `questionnaire`, and `fusion`. Uses an async SQLAlchemy engine (asyncpg driver) for all FastAPI route handlers. Swagger UI is available at `/docs` in development mode.

**Celery worker**
A separate process consuming tasks from the Redis queue. Runs video ML inference synchronously using `SyncSessionLocal` (psycopg2 driver), because TensorFlow is not async-compatible. The worker restarts itself after every 50 tasks to prevent memory accumulation from repeated model inference.

**Redis**
Acts as both the Celery task broker and the result backend. Runs as `redis:7-alpine`.

**PostgreSQL**
Primary relational database running as `postgres:16`. Two tables: `users` and `assessment_sessions`. Schema is versioned through two Alembic migrations — `001_initial_tables` creates both tables and their indexes; `002_add_video_inference_cols` adds `video_score`, `video_error`, and `celery_task_id` columns idempotently.

**ML models (runtime volume mount)**
Model files are mounted from the host's `model/` directory into `/app/model` inside the containers. The video inference service adds this directory to `sys.path` at runtime to resolve the existing ML pipeline imports (`video_loader`, `tta`, `config`).

---

## Full analysis pipeline

### Step 1 — Video upload

The parent selects a video file in the frontend.

```
POST /api/v1/analyze/video/upload
File: backend/app/routers/video.py → upload_video()
```

The file passes a five-stage validation pipeline: session status check, file extension check, Content-Length size check, magic bytes check, and streaming write with byte-count enforcement. Saved to the shared Docker volume at `/tmp/autisense_videos/{session_uuid}/upload.<ext>`. Session status set to `video_uploaded`.

### Step 2 — Analysis start

```
POST /api/v1/analyze/video/start
File: backend/app/routers/video.py → start_video_analysis()
```

FastAPI enqueues a Celery task (`app.tasks.video_task.process_video`) and sets the session status to `video_processing`. Returns the Celery task ID immediately — inference runs asynchronously.

### Step 3 — Video inference (Celery worker)

```
File: backend/app/tasks/video_task.py → process_video()
File: backend/app/services/video_inference.py → run_inference()
```

**3a.** Load model singleton using tf_keras first (for compatibility with the `.h5` saved format). Falls back to `tf.keras.models.load_model(compile=False)` if tf_keras is not importable. The model and label encoder are cached as module-level singletons — loaded once per worker process lifetime.

**3b.** Extract sliding window clips from the video using `extract_sliding_window_clips()` from `model/video_loader.py`. Clip parameters read from `model/config.py`: `sequence_length=8`, `img_height=96`, `img_width=96`, `overlap=0.5`.

**3c.** Cap the clip count at `MAX_CLIPS = 10`, sampled evenly across the full clip list using `numpy.linspace`. This prevents unbounded inference time on longer videos.

**3d.** Run test-time augmentation (TTA) on each clip via `tta_predict()` from `model/tta.py`, with `n_augments=2`. This was reduced from the training default of 8 for performance — reducing total forward passes from ~800 to ~20.

**3e.** Average probabilities across all clips to produce a single prediction vector.

**3f.** Compute video prediction confidence from cross-clip variance using exponential decay:
```
confidence = exp(-8.0 × mean_variance)
```
Low variance (consistent predictions) → high confidence. High variance → low confidence.

**3g.** Compute the composite autism score using a weighted sum of behavioral class probabilities:
```
score = 0.4 × arm_flapping + 0.3 × spinning + 0.4 × head_banging
```

**3h.** Classify video risk band: Low (< 0.3) / Moderate (0.3–0.6) / High (≥ 0.6).

**3i.** Write `video_class_probabilities`, `video_score`, `video_confidence`, `video_confidence_score`, `video_variance`, and `risk_level` to the `assessment_sessions` table. Set status to `video_done`. The uploaded video file is **always** deleted in the `finally` block of `process_video()`, regardless of inference outcome.

### Step 4 — Frontend polls status

```
GET /api/v1/analyze/video/status/{session_uuid}
File: backend/app/routers/video.py → get_video_status()
```

The frontend polls every 3 seconds, up to 100 attempts (5 minutes total). If Celery reports `FAILURE` but the database still shows `video_processing`, the endpoint reconciles the state and writes `error_video` to the database. Polling stops when status is `video_done` or `error_video`.

### Step 5 — Questionnaire submission

```
POST /api/v1/analyze/questionnaire/submit
File: backend/app/routers/questionnaire.py
```

The parent completes 40 behavioral questions across four sections — Social Interaction, Communication, Behavior Patterns, and Sensory & Emotional — each scored 0–4. The questionnaire model (`autism_model.pkl`) scores the responses and writes `questionnaire_probability` and `category_scores` to the session.

### Step 6 — Fusion

```
POST /api/v1/analyze/fuse
File: backend/app/routers/fusion.py → fuse_analysis()
File: backend/app/services/fusion_engine.py → fuse()
```

The fusion engine is a pure stateless function (no database access, no model loading). It combines video and questionnaire scores using a four-tier confidence-based weighting system:

| Video confidence | Video weight | Questionnaire weight |
|-----------------|-------------|---------------------|
| ≥ 0.85 (very high) | 0.50 | 0.50 |
| 0.70–0.84 (high) | 0.40 | 0.60 |
| 0.50–0.69 (moderate) | 0.30 | 0.70 |
| < 0.50 (low) | 0.20 | 0.80 |

If `video_variance > 0.1`, the effective confidence is reduced by `min(0.20, variance × 0.5)` before weight selection.

If the child is under 24 months, the questionnaire weight receives an additional `+0.10` bias (capped at 1.0), because behavioral markers in very young children are less reliably captured on video.

```
final_score = (video_score × video_weight) + (questionnaire_score × questionnaire_weight)
```

Risk bands on the final score: low (< 0.35) / medium (0.35–0.70) / high (> 0.70). Session status is set to `complete`.

### Step 7 — Results displayed

```
GET /api/v1/analyze/report/{session_uuid}
File: frontend/src/components/PostAnalysisResult.jsx
```

`PostAnalysisResult` renders the fusion score with a donut chart, the risk band label, and progress bars for each behavioral signal. On mount, it fires three parallel Groq API calls to generate a personalized summary, a detailed explanation, and actionable next steps for the parent. The report can be downloaded as a plain-text `.txt` file generated entirely in the browser — no server round-trip required.

### Step 8 — Support features

**Chatbot:** The chatbot is embedded inside `AppContent.jsx` and calls `llama-3.3-70b-versatile` via the Groq API directly from the browser. It receives the session's risk scores as context so it can answer parent questions in the context of their specific result.

**Clinic finder:** `LocationSpecialistWidget.jsx` uses the browser's Geolocation API or a manually typed address (geocoded via Geoapify) to search for nearby autism clinics using the Geoapify Places API. Results are displayed as a list alongside a static map image rendered via the Geoapify Static Maps API. If the API key is absent or the query returns no results, the widget shows pre-defined demo clinic data.

---

## Tech stack

| Layer | Technology | Version |
|-------|------------|---------|
| Frontend framework | React | ^19.2.4 |
| Frontend build tool | Vite | ^8.0.3 |
| UI icons | Lucide React | ^1.7.0 |
| Charts | Recharts | ^3.8.1 |
| CSS framework | Tailwind CSS | ^4.2.2 |
| Backend framework | FastAPI | 0.115.12 |
| Async server | Uvicorn | 0.34.2 |
| Task queue | Celery | 5.5.2 |
| Redis client | redis | 6.2.0 |
| Message broker / cache | Redis | 7-alpine (Docker image) |
| Database | PostgreSQL | 16 (Docker image) |
| ORM | SQLAlchemy | 2.0.40 |
| Migrations | Alembic | 1.15.2 |
| Async DB driver | asyncpg | 0.30.0 |
| Sync DB driver (Celery) | psycopg2-binary | 2.9.10 |
| Settings management | pydantic-settings | 2.9.1 |
| Auth (JWT) | python-jose | 3.4.0 |
| Auth (password hashing) | passlib + bcrypt | 1.7.4 + 4.0.1 |
| ML framework | tensorflow-cpu | 2.21.0 |
| Legacy model loader | tf_keras | latest (installed separately) |
| Video processing | opencv-python-headless | 4.11.0.86 |
| Numerical computing | numpy | 2.2.5 |
| Questionnaire ML | scikit-learn | 1.7.0 |
| Data handling | pandas | 2.2.3 |
| Model serialization | joblib | 1.5.1 |
| AI summaries / chatbot | Groq API (llama-3.3-70b-versatile) | groq 0.25.0 |
| Clinic search | Geoapify Places API | — |
| HTTP client | httpx | 0.28.1 |
| Containerization | Docker + Docker Compose | — |

---

## ML models

### Video model — `autism_final.h5`

**Architecture:** MobileNetV2 (pretrained on ImageNet) as a spatial feature extractor, followed by a two-layer GRU head (`[128, 64]` units) for temporal sequence modeling. GRU was chosen over LSTM for its lower parameter count, which is more appropriate for small datasets.

**Input:** Sliding window clips of 8 frames at 96×96 pixels, extracted with 50% overlap between windows.

**Output:** Class probabilities for four behavioral categories — `arm_flapping`, `spinning`, `head_banging`, `normal`.

**Training improvements applied:** Mixup augmentation (α=0.4), label smoothing (0.1), discriminative learning rates (head: 1e-4, top MobileNet layers: 2e-5), cosine annealing LR schedule, and stochastic weight averaging (SWA) over 15 final epochs.

**Loading strategy:** tf_keras is attempted first (handles models saved with older Keras serialization formats). Falls back to `tf.keras.models.load_model(compile=False)` for Keras 3.x compatibility.

**Performance tuning at inference time:** Clips capped at 10 (evenly sampled), TTA augments reduced from 8 to 2, reducing the maximum forward pass count from ~800 to ~20.

### Questionnaire model — `autism_model.pkl`

**Architecture:** An ensemble of five scikit-learn classifiers, each trained in a StandardScaler pipeline:

| Classifier | Key hyperparameters |
|-----------|---------------------|
| Random Forest | 100 estimators, max_depth=5 |
| Gradient Boosting | 100 estimators, max_depth=3 |
| Support Vector Machine (RBF kernel) | probability=True |
| K-Nearest Neighbours | n_neighbors=10 |
| Logistic Regression | C=0.5, class_weight='balanced' |

All five models are trained on a 40-question behavioral questionnaire plus child age and gender. The saved `autism_model.pkl` contains the best-performing model as selected by validation AUC during training — in the final training run this was a Random Forest pipeline. The model is evaluated using 5-fold stratified cross-validation on 60% training data, with 20% validation and 20% holdout test sets.

Five questions from the original 40 (Q7, Q11, Q19, Q30, Q31) were removed due to high inter-feature correlation identified during correlation analysis.

### Fusion algorithm

The fusion engine (`app/services/fusion_engine.py`) is a pure mathematical function, not a trained model.

**Weighting logic:** The video model's numeric confidence score (derived from cross-clip prediction variance) determines which of four weight tiers applies:

```
video_certainty = numeric confidence in [0, 1]
  computed as: exp(-8.0 × mean_variance_across_clips)

Weight selection:
  ≥ 0.85 → video: 0.50, questionnaire: 0.50
  0.70–0.84 → video: 0.40, questionnaire: 0.60
  0.50–0.69 → video: 0.30, questionnaire: 0.70
  < 0.50   → video: 0.20, questionnaire: 0.80

Variance penalty: if video_variance > 0.1
  adjusted_confidence = confidence - min(0.20, variance × 0.5)
  (applied before weight tier selection)

final_score = (video_score × video_weight)
            + (questionnaire_score × questionnaire_weight)
```

**Plain English:** When the video model produces consistent predictions across clips (high confidence), it contributes up to 50% of the final score. When the video model is inconsistent or ambiguous, the questionnaire carries up to 80% of the final score, because parental behavioral observation across 40 questions is a more reliable signal than an uncertain video reading.

**Age rule:** Children under 24 months receive an additional +10% bias toward the questionnaire weight, because behavioral markers in very young children are harder to reliably capture on video. This cap is enforced at a maximum questionnaire weight of 1.0.

**Risk bands on final score:**
- Low: < 0.35
- Medium: 0.35–0.70
- High: > 0.70

---

## Project structure

```
Autism-video-analysis-/
├── docker-compose.yml              # Orchestrates all 6 Docker services
├── .dockerignore                   # Excludes model files, venvs, datasets from build context
├── .gitignore                      # Git exclusions for both backend and frontend
├── video_score.py                  # Standalone video scoring script (development utility)
│
├── backend/
│   ├── Dockerfile                  # Python 3.11-slim; pins bcrypt==4.0.1, installs tf_keras
│   ├── requirements.txt            # All Python dependencies with pinned versions
│   ├── alembic.ini                 # Alembic configuration
│   ├── celery_worker.py            # Celery app factory (broker + backend = Redis)
│   ├── .env                        # Local environment variables (not committed)
│   ├── .env.example                # Template for required environment variables
│   ├── app/
│   │   ├── main.py                 # FastAPI app factory, CORS, security headers, router registration
│   │   ├── config.py               # Pydantic settings — all env vars with defaults
│   │   ├── db/
│   │   │   ├── base.py             # SQLAlchemy DeclarativeBase
│   │   │   ├── session.py          # Async engine + session (used by FastAPI routes)
│   │   │   └── sync_session.py     # Sync engine + session (used by Celery tasks only)
│   │   ├── models/
│   │   │   ├── user.py             # User ORM model (email stored as SHA-256 hash only)
│   │   │   └── session.py          # AssessmentSession ORM model (no PII columns)
│   │   ├── routers/
│   │   │   ├── auth.py             # Registration, login, JWT token endpoints
│   │   │   ├── video.py            # Upload, delete, start inference, poll status
│   │   │   ├── questionnaire.py    # Submit 40-question responses, score via RF model
│   │   │   └── fusion.py           # Fuse results, return full report and summary
│   │   ├── schemas/                # Pydantic request/response schemas
│   │   ├── services/
│   │   │   ├── video_inference.py  # ML adapter: loads model, runs TTA inference pipeline
│   │   │   └── fusion_engine.py    # Pure fusion function: confidence-weighted score combination
│   │   ├── tasks/
│   │   │   └── video_task.py       # Celery task: orchestrates full video inference pipeline
│   │   └── utils/
│   │       ├── privacy.py          # Temp file management, session directory helpers
│   │       └── validators.py       # File extension, size, and magic bytes validation
│   └── alembic/
│       └── versions/
│           ├── 001_initial_tables.py           # Creates users + assessment_sessions tables
│           └── 002_add_video_inference_cols.py # Adds video_score, video_error, celery_task_id (idempotent)
│
├── frontend/
│   ├── Dockerfile                  # Node 20-slim; Vite dev server on 0.0.0.0
│   ├── package.json                # React 19, Vite 8, Recharts, Lucide React, Tailwind CSS
│   └── src/
│       ├── App.jsx                 # Root component — wraps AppContent in AuthProvider
│       ├── AppContent.jsx          # All screens and navigation state, chatbot, questionnaire
│       ├── main.jsx                # React DOM entry point
│       ├── index.css               # Global styles
│       ├── components/
│       │   ├── PostAnalysisResult.jsx       # Fusion score UI, Groq AI summaries, downloadable report
│       │   ├── LocationSpecialistWidget.jsx  # Geoapify clinic finder with static map
│       │   └── BottomTabBar.jsx             # Mobile navigation tab bar
│       ├── hooks/
│       │   └── useAuth.jsx         # Auth context and JWT token management
│       └── services/               # API service layer for backend calls
│
└── model/
    ├── autism_final.h5             # MobileNet-GRU behavioral classification model (~29 MB)
    ├── autism_model.pkl            # Questionnaire ensemble model (~344 KB)
    ├── config.py                   # ML pipeline configuration (sequence length, image size, etc.)
    ├── video_loader.py             # Sliding window clip extraction from video files
    ├── tta.py                      # Test-time augmentation prediction
    ├── augmentation.py             # Frame-level augmentation functions
    ├── model.py                    # Model architecture definition (MobileNet-GRU)
    ├── trainer.py                  # Training loop with two-phase transfer learning
    ├── improve.py                  # Training improvement techniques (Mixup, SWA, etc.)
    ├── dataset_builder.py          # Dataset loading and preprocessing
    ├── predictor.py                # Standalone prediction utilities
    ├── optical_flow.py             # Optical flow dual-stream (improvement 7, disabled by default)
    ├── main.py                     # Training entry point
    ├── questionnarie_model_train.py # Questionnaire ensemble training and evaluation script
    ├── questionnarie_model_test.py  # Questionnaire model testing and interactive CLI
    ├── requirements.txt            # ML-only Python dependencies
    └── COLAB_SETUP.md              # Instructions for running training in Google Colab
```

---

## Environment variables

| Variable | Service | Required | Default | Description |
|----------|---------|----------|---------|-------------|
| `DATABASE_URL` | backend, celery, migrate | yes | — | PostgreSQL asyncpg connection string |
| `REDIS_URL` | backend, celery | yes | `redis://localhost:6379` | Redis connection string |
| `SECRET_KEY` | backend | yes | insecure default | JWT signing key (minimum 32 characters) |
| `ALGORITHM` | backend | no | `HS256` | JWT signing algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | backend | no | `60` | JWT token lifetime in minutes |
| `GROQ_API_KEY` | backend | no | `""` | Groq API key (backend; currently unused at runtime) |
| `VITE_GROQ_API_KEY` | frontend | no | — | Groq API key for AI summaries and chatbot |
| `VITE_GEOAPIFY_API_KEY` | frontend | no | — | Geoapify API key for clinic search, geocoding, and static maps |
| `MODEL_VIDEO_PATH` | backend, celery | yes | `../model/autism_final.h5` | Path to video classification model |
| `MODEL_ENCODER_PATH` | backend, celery | yes | `ml_models/video_model/label_encoder.pkl` | Path to video label encoder |
| `MODEL_RF_PATH` | backend, celery | yes | `ml_models/questionnaire_model/autism_model.pkl` | Path to questionnaire model |
| `TEMP_VIDEO_DIR` | backend, celery | no | `/tmp/autisense_videos` | Directory for temporary uploaded video files |
| `MAX_VIDEO_SIZE_MB` | backend | no | `50` | Maximum allowed video file size in megabytes |
| `VIDEO_INFERENCE_SOFT_TIMEOUT` | backend, celery | no | `300` | Celery soft time limit in seconds (raises SoftTimeLimitExceeded) |
| `VIDEO_INFERENCE_HARD_TIMEOUT` | backend, celery | no | `600` | Celery hard kill limit in seconds |
| `ENVIRONMENT` | backend | no | `development` | Controls Swagger UI visibility and SQL echo |
| `LOG_LEVEL` | backend | no | `INFO` | Logging verbosity: DEBUG / INFO / WARNING |
| `ALLOWED_ORIGINS` | backend | no | localhost:3000, 5173 | CORS allowed origins list |

**Note:** `VITE_GROQ_API_KEY` and `VITE_GEOAPIFY_API_KEY` are optional. When absent, AI content falls back to static text and the clinic finder shows demo data.

---

## Local development setup

### Prerequisites

- Docker Desktop installed and running
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/srihasitha1/Autism-video-analysis-
cd Autism-video-analysis-

# 2. Add your API keys to docker-compose.yml (optional — app works without them)
#    Edit the frontend service environment section:
#      VITE_GROQ_API_KEY: "your_groq_key_here"
#      VITE_GEOAPIFY_API_KEY: "your_geoapify_key_here"
#
#    Without these keys:
#      - AI summaries fall back to static placeholder text
#      - Clinic finder displays hardcoded demo clinic data

# 3. Build and start all services
docker compose up --build

# 4. Wait for this sequence in the logs:
#    autisense_postgres  | database system is ready to accept connections
#    autisense_migrate   | Running upgrade -> 001_initial_tables
#    autisense_migrate   | Running upgrade 001_initial_tables -> 002_add_video_inference_cols
#    autisense_migrate   | Done
#    autisense_backend   | Application startup complete
#    autisense_celery    | celery@... ready.
#    autisense_frontend  | VITE v8.x.x  ready

# 5. Open the application
#    Frontend:   http://localhost:5173
#    API:        http://localhost:8000
#    API docs:   http://localhost:8000/docs
```

### Stopping the application

```bash
docker compose down
```

### Wiping the database and starting fresh

```bash
# Stops containers and removes named volumes (postgres_data + video_temp)
docker compose down -v
docker compose up --build
```

---

## Known limitations

- Video inference is capped at 10 clips sampled evenly across the video. For videos longer than approximately 30 seconds, temporal segments between sampled clips are not evaluated.
- TTA augmentations are reduced to 2 per clip (from the training default of 8) for performance. This trades some calibration accuracy for approximately a 4x speed improvement.
- The questionnaire model was trained with five highly correlated questions removed (Q7, Q11, Q19, Q30, Q31). Questionnaire inputs must match this reduced feature set — 35 question responses plus age and gender.
- The questionnaire model does not account for cultural or developmental context beyond the child's age in years.
- Geoapify clinic results depend on OpenStreetMap data quality, which varies significantly by region. Rural areas or regions with sparse OpenStreetMap coverage may return few or no clinic results.
- The fusion algorithm uses a hand-crafted confidence-tier weighting formula, not a trained meta-model. The age-based rule for children under 24 months is a heuristic, not clinically validated.
- The video model recognizes four behavioral classes only (arm flapping, spinning, head banging, normal). Behaviors outside this vocabulary do not influence the video score.
- The frontend runs a Vite development server in Docker. For production deployment, the frontend Dockerfile should be replaced with a multi-stage build that compiles static assets and serves them via Nginx.
- This tool has not been clinically validated. It is a prototype built for educational and research purposes.

---

## Contributing

This project was built as a prototype. If you are extending it:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and test with `docker compose up --build`
4. Submit a pull request with a clear description of what changed and why

All ML model changes must be accompanied by updated inference timing benchmarks.

---

## License

License not yet specified.
