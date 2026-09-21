# CLAUDE.md - Project Architecture Baseline

## Project Overview
Neural Network Based Dynamic LLM Router - An intelligent routing system that dynamically selects appropriate LLMs based on query difficulty predicted by a trained neural network classifier.

## Tech Stack
- **Language:** Python 3.10+
- **Backend:** FastAPI, Uvicorn
- **Auth:** Google Identity Services (frontend), google-auth + PyJWT (backend)
- **ML Framework:** TensorFlow/Keras (MLP classifier)
- **Embeddings:** Sentence Transformers (local embedding model)
- **LLM Framework:** Hugging Face Transformers
- **Data Processing:** Pandas, NumPy
- **Evaluation:** Scikit-learn (Accuracy, Precision, Recall, F1, Macro-F1, Confusion Matrix)
- **Frontend:** React
- **Deployment:** AWS

## Architecture Type
Monolithic application with modular components:
- Google Sign-In authentication + session tokens
- Embedding service
- Neural network router (Keras MLP)
- Router policy engine
- LLM model registry and inference
- FastAPI backend
- React frontend

## Data Stores
- **Dataset:** JSON file (merged_llm_router_dataset_no_difficulty.json)
- **Trained Models:** Saved Keras models in models/trained_router/
- **Configuration:** YAML config file
- **Search History:** SQLite database (data/history.db) via stdlib `sqlite3`, table `search_history` (scoped per user)
- **Users:** Same SQLite database, table `users` (Google-authenticated accounts)
- **No external database server required** - file-based storage + embedded SQLite

## Testing Strategy
- **Unit Tests:** REQUIRED for all business logic (preprocessing, embedding, router prediction, policy, model registry)
- **Integration Tests:** REQUIRED for API endpoints, model loading, LLM inference pipeline
- **E2E Tests:** REQUIRED for frontend-backend chat flow (user query → response)
- **Framework:** pytest for backend, React Testing Library + Playwright for frontend
- **Coverage Target:** 80% minimum

## Quality Gates
- Lint: ruff (Python), ESLint (React)
- Type Check: mypy (Python), TypeScript (React)
- Security: bandit (Python)
- Test Coverage: pytest-cov >= 80%

## Feature Flags
- `USE_MOCK_LLMS` - Use mock LLM responses for development/testing
- `ENABLE_FALLBACK` - Enable confidence-based fallback routing
- `ENABLE_SAFETY_VALIDATION` - Enable Llama Guard safety validation
- `ENABLE_HISTORY` - Persist user search history to SQLite (config: `features.enable_history`)

## Model Registry (LLMs)
- **Easy:** microsoft/Phi-3-mini-128k-instruct
- **Medium:** google/gemma-7b
- **Hard:** meta-llama/Llama-Guard-4-12B (with fallback documented)

## Difficulty Classes
- Easy, Medium, Hard (3 classes, one-hot encoded in dataset)

## Authentication
- All endpoints require `Authorization: Bearer <session token>` except
  `POST /auth/google` and `GET /health`
- Any Google account may sign in; each user sees only their own history
- Secrets come from the environment: `GOOGLE_CLIENT_ID`, `SESSION_SECRET`
  (>=32 bytes). Never commit them or put them in config.yaml
- Frontend needs `VITE_GOOGLE_CLIENT_ID` set to the same client id
- Tests mint real session tokens via `tests/auth_helpers.py`; auth is never
  bypassed or disabled in the test suite

## Confidence Threshold
Configurable via config.yaml (currently 0.50). Exposed to the frontend on
`GET /health` and on every `/chat` and `/route` response, so the confidence
meter can draw its threshold tick from the real value rather than hardcoding it.

## Frontend Design Tokens
Defined in `frontend/src/index.css` under `@theme`; no component library.
- Warm paper ground (#F6F4EF) + ink (#1A1822); primary buttons are ink, not colour
- Space Grotesk (display), IBM Plex Sans (body), IBM Plex Mono (all machine values)
- Difficulty is an ORDERED scale: easy #1F6F5C, medium #8A5F12, hard #A8432C,
  cool to warm, one lightness, each 4.5:1 on its tint. Do not swap these for
  arbitrary categorical hues.

## API Endpoints
- POST /auth/google - Exchange a Google ID token for a session token (public)
- GET /auth/me - Current user profile
- POST /auth/logout - Sign out
- POST /route - Routing decision only, no generation (fast path for the UI)
- POST /chat - Main chat endpoint
- GET /health - Health check (public)
- GET /models - List available models
- GET /history - List past searches (limit/offset/q)
- GET /history/{id} - Fetch one past search
- GET /history/stats - Aggregate history statistics
- DELETE /history/{id} - Delete one past search
- DELETE /history - Clear all history

## Project Structure (Target)
```
llm-router/
├── auth/
├── data/
├── db/
├── embeddings/
├── router/
├── llm/
├── api/
├── models/
├── frontend/
├── requirements.txt
├── config.yaml
└── README.md
```