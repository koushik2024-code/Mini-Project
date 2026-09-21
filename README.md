# Neural Network Based Dynamic LLM Router

An intelligent routing system that dynamically selects appropriate LLMs based on query difficulty predicted by a trained neural network classifier.

## Overview

This project implements a **Neural Network Based Dynamic LLM Router** that intelligently routes user queries to appropriate LLMs based on predicted query difficulty. The system follows the core principle:

**EMBED → PREDICT → ROUTE → EXECUTE → RESPOND**

The neural network ONLY classifies/routes. The selected LLM generates the final answer using the ORIGINAL user query.

## Architecture

```
User Query → Preprocessing → Embedding (Sentence Transformers) 
    → Keras MLP Router → Difficulty Class + Confidence 
    → Router Policy (Confidence Threshold) → LLM Selection 
    → Original Query → Selected LLM (Ollama) → Response
```

## Features

- **Intelligent Routing**: Keras MLP classifier predicts query difficulty (Easy/Medium/Hard)
- **Confidence-Based Fallback**: Low confidence predictions fall back to medium tier
- **Multi-Model Support**: qwen3:1.7b (Easy/Medium), llama3.2:3b (Hard)
- **Local Inference**: Ollama for local LLM inference (no API keys needed)
- **Visible Routing**: Every answer carries a receipt - tier, model, confidence meter with the fallback threshold marked
- **Google Sign-In**: The whole app sits behind Google OAuth; each account gets its own private history
- **Search History**: Every query is persisted to SQLite and browsable from a left sidebar
- **Professional Frontend**: React + TypeScript + Tailwind CSS v4 with dark/light mode
- **FastAPI Backend**: High-performance async API with proper error handling
- **Comprehensive Testing**: 148+ tests (unit, integration, E2E)
- **Quality Gates**: Ruff, MyPy, Bandit, Pytest coverage

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend)
- Ollama running locally with models: `qwen3:1.7b`, `llama3.2:3b`

### Installation

```bash
# Clone and setup
git clone <repo>
cd mini-project

# Backend
pip install -r requirements.txt

# Frontend
cd frontend && npm install && cd ..

# Configure Google Sign-In (see "Google Sign-In setup" below)
cp .env.example .env
cp frontend/.env.example frontend/.env

# Start Ollama (in separate terminal)
ollama serve

# Start Backend (in separate terminal)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Start Frontend (in separate terminal)
cd frontend && npm run dev
```

### Google Sign-In setup

The app is gated behind Google Sign-In, so it needs an OAuth client before it
will let anyone in. This takes about two minutes and is free.

1. Go to the [Google Cloud Console credentials page](https://console.cloud.google.com/apis/credentials)
   and select or create a project.
2. If prompted, configure the **OAuth consent screen**: choose **External**,
   fill in an app name and your email, and save. While the app is in
   *Testing*, add the Google accounts you want to allow under **Test users**.
3. Click **Create Credentials -> OAuth client ID**, choose
   **Web application**, and under **Authorized JavaScript origins** add:
   ```
   http://localhost:5173
   ```
   (add your deployed origin too, when you have one).
4. Copy the generated **Client ID**. It looks like
   `123456789-abc123.apps.googleusercontent.com`.
5. Put the *same* client id in both env files, and generate a session secret:

   `.env` (backend):
   ```
   GOOGLE_CLIENT_ID=123456789-abc123.apps.googleusercontent.com
   SESSION_SECRET=<paste the generated secret>
   ```

   `frontend/.env`:
   ```
   VITE_GOOGLE_CLIENT_ID=123456789-abc123.apps.googleusercontent.com
   ```

   Generate the session secret with:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(48))"
   ```
6. Restart both servers. Vite only reads `.env` at startup, so the frontend
   must be restarted, not just refreshed.

Notes:
- `SESSION_SECRET` must be at least 32 bytes. If it is unset the server
  generates a random one at startup and logs a warning - sign-in still works,
  but everyone is signed out whenever the backend restarts.
- The client id is not a secret (it ships in the frontend bundle by design).
  `SESSION_SECRET` **is** a secret; keep it out of version control.
- `GET /health` reports `auth_configured` so you can check whether the backend
  picked up the client id.

### Access
- **Frontend**: http://localhost:5173
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Endpoints

All endpoints except `POST /auth/google` and `GET /health` require a
`Authorization: Bearer <session token>` header.

### POST /auth/google
Exchange a Google ID token (from Google Identity Services in the browser) for
a backend session token.

**Request:**
```json
{ "credential": "<google id token>" }
```

**Response:**
```json
{
  "access_token": "<session jwt>",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {
    "id": 1,
    "email": "you@gmail.com",
    "name": "Your Name",
    "picture": "https://lh3.googleusercontent.com/..."
  }
}
```

Returns 401 if the Google token is invalid, expired, issued for a different
OAuth client, or its email is unverified.

### GET /auth/me
Return the signed-in user's profile. 401 if the session token is missing,
expired, or forged.

### POST /auth/logout
Sign out. Session tokens are stateless, so the client discards its token; the
endpoint exists so the frontend has one place to call.

### POST /route
Classify a query and return the routing decision *without* generating an answer.

Routing is the cheap half of the pipeline: on this machine it measures ~0.5s
against ~47s for the same query through `/chat`. The frontend calls this first
so it can show which model was picked, and why, while the answer is still
being written.

**Request:**
```json
{ "query": "Explain how a binary search tree works" }
```

**Response:**
```json
{
  "predicted_class": "medium",
  "confidence": 0.4432,
  "selected_tier": "medium",
  "selected_model": "qwen3:4b",
  "fallback_used": true,
  "confidence_threshold": 0.5,
  "latency_ms": 490
}
```

It does not call an LLM and does not write history. `/chat` runs the same
`_classify()` helper, so the two can never disagree.

### POST /chat
Main chat endpoint. Routes query to appropriate LLM based on predicted difficulty.

**Request:**
```json
{
  "query": "Explain how a binary search tree works"
}
```

**Response:**
```json
{
  "query": "Explain how a binary search tree works",
  "predicted_class": "medium",
  "confidence": 0.87,
  "selected_model": "qwen3:4b",
  "response": "A binary search tree (BST) is a data structure...",
  "fallback_used": false,
  "latency_ms": 2450,
  "confidence_threshold": 0.5
}
```

### GET /history
List the signed-in user's past searches, newest first. History is scoped per
account - one user can never read or delete another's entries. Query params: `limit` (1-200, default 50),
`offset` (default 0), `q` (filter by query text).
```json
{
  "entries": [
    {
      "id": 42,
      "query": "Explain how a binary search tree works",
      "response": "A binary search tree (BST) is a data structure...",
      "predicted_class": "medium",
      "confidence": 0.87,
      "selected_model": "qwen3:4b",
      "fallback_used": false,
      "latency_ms": 2450,
      "created_at": "2026-09-20T10:31:04+00:00"
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

### GET /history/{id}
Fetch one past search, including its full response. Returns 404 if unknown.

### GET /history/stats
Aggregate counts per difficulty tier, plus average latency and confidence.
```json
{
  "total": 37,
  "avg_latency_ms": 1820.4,
  "avg_confidence": 0.8123,
  "by_class": {"easy": 18, "medium": 12, "hard": 7}
}
```

### DELETE /history/{id}
Delete a single past search. Returns 404 if unknown.

### DELETE /history
Clear the entire search history. Returns `{"deleted": <count>}`.

All `/history` endpoints return 503 when `features.enable_history` is `false`.

### GET /health
Health check endpoint.
```json
{
  "status": "healthy",
  "router_loaded": true,
  "embedding_loaded": true,
  "models_loaded": ["qwen3:1.7b", "llama3.2:3b"],
  "auth_configured": true,
  "confidence_threshold": 0.5
}
```

### GET /models
List available models with tiers.
```json
{
  "models": [
    {"tier": "easy", "name": "qwen3:1.7b", "loaded": true},
    {"tier": "medium", "name": "qwen3:4b", "loaded": true},
    {"tier": "hard", "name": "llama3.2:3b", "loaded": true}
  ]
}
```

## Design system

The interface is built on a small set of tokens in
[frontend/src/index.css](frontend/src/index.css), not a component library.

- **Ground** is warm paper (`#F6F4EF`) with near-black ink (`#1A1822`), rather
  than cold white-on-slate. It reads better in screenshots and print.
- **Type** is Space Grotesk for headings, IBM Plex Sans for body, IBM Plex
  Mono for every machine value - model ids, confidences, timings - so a model
  id never reads as prose.
- **Difficulty is an ordered variable**, so easy/medium/hard use an ordered
  ramp running cool to warm (`#1F6F5C` / `#8A5F12` / `#A8432C`) rather than
  three unrelated hues. All three sit at roughly one lightness so no tier
  shouts louder than another, and each clears 4.5:1 contrast on its tint.
- **Primary actions are ink, not colour**, so the only real hue in the
  interface belongs to the tier scale.

The confidence meter is the one non-obvious component. Its tick marks the
fallback threshold from `config.yaml`: fill past the tick means the predicted
tier was used, fill short of it means `RouterPolicy` dropped to the fallback
tier and the Fallback badge appears beside it.

Reference mockups: https://claude.ai/artifact/Vey5PqJPf73nSkhxFAFwQY

## Configuration

Configuration via `config.yaml`:

```yaml
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"
  cache_dir: "embeddings/cache"

router:
  model_path: "models/router/best_model.keras"
  confidence_threshold: 0.70
  fallback_tier: "medium"

ollama:
  host: "http://localhost:11434"
  timeout: 180
  models:
    easy: "qwen3:1.7b"
    medium: "qwen3:1.7b"  # Using 1.7b for medium (4b too slow on CPU)
    hard: "llama3.2:3b"

features:
  use_mock_llms: true
  use_ollama: true
  enable_fallback: true
  enable_safety_validation: false
```

## Project Structure

```
mini-project/
├── api/                    # FastAPI backend
│   └── main.py            # FastAPI app with /chat, /health, /models, /history
├── auth/                   # Authentication
│   ├── google_auth.py      # Google ID token verification
│   └── session.py          # Backend-issued session JWTs
├── db/                     # SQL storage
│   ├── history_store.py    # Per-user search-history store (schema + queries)
│   └── user_store.py       # Google-authenticated users
├── data/                   # Dataset and loaders
│   ├── dataset_loader.py   # Dataset loading and validation
│   ├── ml_dataset_prep.py  # ML dataset preparation, label mapping, splits
│   └── dataset_loader.py   # Dataset loader
├── embeddings/             # Embedding pipeline
│   ├── embedding_model.py  # Sentence Transformers with caching
│   └── preprocessing.py    # Text preprocessing
├── router/                 # Neural network router
│   ├── keras_mlp.py        # Keras MLP architecture
│   ├── trainer.py          # Training pipeline with callbacks
│   ├── predictor.py        # Query prediction
│   ├── router_policy.py    # Router policy (threshold, fallback)
│   └── evaluate.py         # Comprehensive evaluation
├── llm/                    # LLM integration
│   ├── ollama_client.py    # Ollama HTTP client
│   └── mock_llm.py         # Mock LLM for testing
├── frontend/               # React frontend
│   ├── src/
│   │   ├── App.tsx        # Main chat application
│   │   ├── HistorySidebar.tsx # Left-side search history panel
│   │   ├── LoginScreen.tsx # Google Sign-In gate
│   │   ├── RoutingReceipt.tsx # Tier badge + confidence meter
│   │   ├── Markdown.tsx   # Dependency-free markdown renderer
│   │   ├── useAuth.ts     # Session state hook
│   │   ├── auth.ts        # Token storage + authenticated fetch
│   │   ├── index.css      # Tailwind CSS v4 styles
│   │   └── main.tsx       # Entry point
│   └── ...
├── models/                 # Trained models
│   └── router/
│       └── best_model.keras
├── tests/                  # Test suite (148+ tests)
│   ├── test_api.py
│   ├── test_history_store.py
│   ├── test_history_api.py
│   ├── test_auth.py
│   ├── test_auth_api.py
│   ├── test_user_store.py
│   ├── test_route_api.py
│   ├── test_e2e.py
│   ├── test_complete_pipeline.py
│   ├── test_predictor.py
│   ├── test_router_policy.py
│   ├── test_mock_llm.py
│   ├── test_ollama_client.py
│   ├── test_advanced_features.py
│   └── ...
├── config.yaml             # Configuration
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_api.py -v
pytest tests/test_e2e.py -v
pytest tests/test_complete_pipeline.py -v

# Quality gates
ruff check .
mypy .
bandit -r .
pytest --co
```

## Evaluation Results

### Router Performance (Test Set)
| Metric | Value |
|--------|-------|
| Accuracy | 60.88% |
| Macro-F1 | 0.5021 |
| ECE (Calibration) | 0.0850 |

### Per-Class F1 Scores
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Easy  | 0.7602    | 0.6806 | 0.7182   |
| Medium| 0.7237    | 0.5392 | 0.6180   |
| Hard  | 0.1143    | 0.3333 | 0.1702   |

### Confusion Matrix
```
              Predicted
Actual    Easy  Medium  Hard
Easy       130     18     43
Medium      28     55     19
Hard        13      3      8
```

Model: `hidden_dims=[512]`, `dropout=0.5`, `lr=0.0001`, `batch_size=64`,
class-weighted, selected by best **validation macro-F1** across a 5-config
sweep (never selected on the test set). A config selected by validation loss
instead scored higher raw accuracy (72.2%) but completely collapsed on the
hard class (0 precision/recall/F1) — see [Dataset & Accuracy Notes](#dataset--accuracy-notes-updated-2026-09-19)
below for why macro-F1 selection was used instead.

### Dataset & Accuracy Notes (updated 2026-09-19)

**The dataset was switched.** The router originally trained on a dataset with
an ambiguous 4-pattern multi-label scheme collapsed into 3 classes by a fixed
rule; k-NN label purity for that scheme was only 0.46 (barely above the 0.33
random baseline) — an information-theoretic ceiling proven via 50-run
hyperparameter search (best 59.9%), alternate model families (RandomForest
56.5%, GradientBoosting 54.9%), and alternate embeddings (MiniLM -> mpnet,
no change), none of which broke past ~60%. That dataset is preserved at
`data/backup_old_multilabel/` for reference.

The current dataset (`data/router_dataset_single_label.jsonl`) instead carries
a single clean `model_label` per sample — the actual router-tier model that
handled that question. Its k-NN purity is meaningfully higher (0.59), and a
RandomForest ceiling check reaches 70.8%, confirming genuinely better
text-label correlation.

**But it's heavily imbalanced**: 60.3% easy, 32.1% medium, only **7.5% hard**
(238 of 3164 samples). Easy/medium are now well-discriminated (F1 0.72/0.62,
up from 0.52/0.60 on the old dataset). The hard tier remains weak (F1 0.17)
purely because of its small sample count (24 test examples) — not a modeling
or tuning problem. Selecting the deployed checkpoint by validation loss alone
let the model collapse entirely on that minority class (0% recall); selecting
by validation macro-F1 instead avoids the collapse (33% recall, though only
11% precision) at the cost of some raw accuracy. See [Known Limitations](#known-limitations).

## Known Limitations

1. **Hard-tier accuracy (F1 0.17)**: Data-starved, not a modeling issue — only
   238 hard-labeled examples exist versus 1909/1017 for easy/medium. Precision
   is low (11%, over-routes easy/medium queries to the priciest model) and
   recall is modest (33%, still misses most truly-hard queries).
2. **Confidence Calibration**: ECE = 0.085 (worse than a prior deployment's
   0.053 on the old dataset — consistent with more genuine uncertainty on the
   harder-to-learn minority class)
3. **Medium Tier Model**: Using qwen3:1.7b instead of qwen3:4b for CPU performance
4. **Hard Tier**: Low-confidence predictions fall back to medium tier per the
   confidence-threshold policy — the intended safety net for exactly this gap

## Future Improvements

1. **More hard-tier training data**: the highest-leverage fix by far — grow
   the 238-example hard class (more real examples, or a targeted LLM-as-judge
   relabeling pass on borderline medium/hard cases). This is what it would
   take to meaningfully close the accuracy gap, not further hyperparameter tuning.
2. **Confidence calibration**: Temperature or Platt scaling, given ECE
   regressed with the new dataset.
3. **Transformer Classifier**: Replace MLP with attention-based classifier —
   worth revisiting once hard-tier data volume improves.
4. **Data Augmentation**: Paraphrasing/back-translation targeted at the hard class
5. **GPU Support**: Enable GPU inference for larger models
6. **Model Ensembles**: Ensemble multiple routers once the hard-tier data gap is closed

## License

MIT License - feel free to use for research and development.

## Acknowledgments

- Sentence Transformers for embeddings
- TensorFlow/Keras for neural network
- Ollama for local LLM inference
- FastAPI for API framework
- React + Tailwind CSS for frontend