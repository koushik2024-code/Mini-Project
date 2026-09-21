# PRD-0001: Neural Network Based Dynamic LLM Router

## Metadata
- **PRD ID:** PRD-0001
- **Version:** 1.0
- **Status:** 📋 Draft
- **Created:** 2026-09-13
- **Last Updated:** 2026-09-13
- **Complexity:** Complex
- **Author:** System Architect

---

## 1. Introduction

### 1.1 Problem Statement
Current LLM applications route all user queries to a single large language model, resulting in:
- Unnecessary computational cost for simple queries
- Suboptimal response quality for complex queries
- No intelligent adaptation to query difficulty

### 1.2 Solution Overview
Build an intelligent dynamic LLM routing system that:
1. Receives a user query
2. Preprocesses and embeds the query
3. Uses a trained neural network (TensorFlow/Keras MLP classifier) to predict query difficulty (Easy/Medium/Hard)
4. Applies a Router Policy with confidence threshold to select the appropriate LLM
5. Sends the **original user query** to the selected LLM for response generation
6. Returns the final answer to the user

### 1.3 Core Principle
**EMBED → PREDICT → ROUTE → EXECUTE → RESPOND**

### 1.4 Critical Architectural Rules

| Rule | Description |
|------|-------------|
| **ML Framework** | TensorFlow + Keras (NOT PyTorch) |
| **Router Architecture** | Embedding → Dense → ReLU → Dense → ReLU → Dense(3) → Softmax |
| **Router Learns** | Query Embedding → Difficulty Class (Easy/Medium/Hard) |
| **Router Does NOT Learn** | Query → Answer |
| **Data Leakage Rule** | The "answer" field MUST NOT be used as input feature during router training |
| **Label Mapping** | Verify dataset's one-hot label mapping before training. Do NOT assume [1,0,0]=Easy, [0,1,0]=Medium, [0,0,1]=Hard. **New dataset uses multi-label format - must map to 3 difficulty classes** |
| **Data Split** | 80/10/10 stratified split (Train/Validation/Test) |
| **Optimizer** | Adam |
| **Loss Function** | Categorical Crossentropy |
| **Output Activation** | Softmax |
| **Embedding Caching** | Dataset embeddings generated once and reused |
| **No RAG Chunking** | Each query embedded as one complete semantic unit. NO fixed-size chunking |
| **Component Separation** | Embedding Model: Text → Vector. Keras MLP: Vector → Class + Confidence. Router Policy: Class + Confidence → Selected LLM |
| **LLM Input** | Original user query sent to LLM. NEVER send embedding vector or class name |
| **Confidence Threshold** | Configurable (default 0.70). confidence ≥ threshold → predicted tier; confidence < threshold → fallback tier |
| **Mock LLMs First** | Complete and test: Query → Embedding → Router → Policy → Mock LLM before real models |
| **Modular LLM Integration** | Each model tested independently before pipeline integration |
| **Model Mapping** | **Easy→Qwen3-1.7B-Instruct, Medium→Qwen3-4B-Instruct, Hard→Llama-3.2-3B-Instruct** |
| **Llama Guard Note** | **Removed** - New Hard-tier model is Llama-3.2-3B-Instruct (general-purpose) |

---

## 2. Goals & Success Criteria

### 2.1 Primary Goals
- [ ] Route queries to appropriate LLM tier based on predicted difficulty
- [ ] Achieve ≥85% routing accuracy on test set (TARGET, not guaranteed)
- [ ] Reduce average computational cost vs. single-LLM baseline
- [ ] Maintain response quality across all difficulty tiers

### 2.2 Success Metrics
| Metric | Target |
|--------|--------|
| Router Accuracy | ≥85% (TARGET) |
| Macro-F1 Score | ≥0.80 |
| Confidence Calibration | ECE ≤ 0.10 |
| Fallback Rate | ≤15% |
| Avg Latency (Easy) | <2s |
| Avg Latency (Medium) | <5s |
| Avg Latency (Hard) | <15s |
| Cost Reduction vs Baseline | ≥40% |

---

## 3. User Stories

| ID | Story | Priority |
|----|-------|----------|
| US-001 | As a user, I want my query answered by the most appropriate LLM so that I get quality responses efficiently | Critical |
| US-002 | As a developer, I want to see which model was selected and why so that I can debug routing decisions | High |
| US-003 | As a system operator, I want configurable confidence thresholds so that I can tune the fallback behavior | High |
| US-004 | As a user, I want a simple chat interface to submit queries and see responses | Critical |
| US-005 | As a researcher, I want evaluation metrics comparing dynamic routing vs single-LLM baseline | Medium |

---

## 4. Functional Requirements

### 4.1 Data Pipeline (Offline Training)

| FR ID | Requirement | Description |
|-------|-------------|-------------|
| PRD-0001-FR-1 | Load and validate dataset | Load `combined_labels_one_hot.jsonl`, validate schema, check for missing values |
| PRD-0001-FR-2 | Verify difficulty label mapping | **MUST** confirm actual one-hot encoding mapping from data. Do NOT assume [1,0,0]=Easy, [0,1,0]=Medium, [0,0,1]=Hard |
| PRD-0001-FR-3 | Preprocess training prompts | Clean whitespace, normalize text, preserve math/code/symbols, **NO RAG-style chunking** |
| PRD-0001-FR-4 | Generate embeddings with caching | Use Sentence Transformers to convert prompts to vectors. **Cache embeddings** so generated once and reused |
| PRD-0001-FR-5 | Split data | **80/10/10 stratified split** (Train/Validation/Test) |
| PRD-0001-FR-6 | Build Keras MLP classifier | **Embedding → Dense → ReLU → Dense → ReLU → Dense(3) → Softmax** |
| PRD-0001-FR-7 | Train with validation | **Adam optimizer, Categorical Crossentropy loss, Softmax output**, Early Stopping, Model Checkpointing, Class Weights |
| PRD-0001-FR-8 | Evaluate router | **Accuracy, Precision, Recall, F1-score, Macro-F1, Per-class F1, Confusion Matrix**, Confidence Calibration |
| PRD-0001-FR-9 | Save trained router | Save Keras model to **`models/router/best_model.keras`** with embedding model metadata and label mapping |

### 4.2 Online Inference Pipeline

| FR ID | Requirement | Description |
|-------|-------------|-------------|
| PRD-0001-FR-10 | Accept user query via API | POST `/chat` with `{ "query": "..." }` |
| PRD-0001-FR-11 | Preprocess query | Same preprocessing as training (FR-3) |
| PRD-0001-FR-12 | Generate query embedding | Use same embedding model as training (FR-4), cached |
| PRD-0001-FR-13 | Predict difficulty class | Pass embedding through trained Keras MLP → class probabilities |
| PRD-0001-FR-14 | Calculate confidence | Max probability as confidence score |
| PRD-0001-FR-15 | Apply Router Policy | Map predicted class + confidence → LLM selection (separate from neural network) |
| PRD-0001-FR-16 | Confidence-based fallback | If confidence < threshold (default 0.70) → use configured fallback tier |
| PRD-0001-FR-17 | Select LLM from registry | Easy→Qwen/Qwen3-1.7B-Instruct, Medium→Qwen/Qwen3-4B-Instruct, Hard→meta-llama/Llama-3.2-3B-Instruct |
| PRD-0001-FR-18 | Generate response | Send **ORIGINAL USER QUERY** to selected LLM (NOT embedding, NOT class name) |
| PRD-0001-FR-19 | Optional safety validation | If enabled, run Llama Guard on response |
| PRD-0001-FR-20 | Return structured response | `{ query, predicted_class, confidence, selected_model, response, fallback_used, latency_ms }` |

### 4.3 Router Policy Engine

| FR ID | Requirement | Description |
|-------|-------------|-------------|
| PRD-0001-FR-21 | Model registry | Centralized `MODEL_REGISTRY` mapping difficulty → model name |
| PRD-0001-FR-22 | Confidence threshold | Configurable via `config.yaml` (default: 0.70) |
| PRD-0001-FR-23 | Fallback strategy | Low confidence → configured fallback tier (default: medium) |
| PRD-0001-FR-24 | Llama Guard documentation | **Removed** - New Hard-tier model is Llama-3.2-3B-Instruct (general-purpose) |

### 4.4 API Endpoints

| FR ID | Requirement | Description |
|-------|-------------|-------------|
| PRD-0001-FR-25 | POST `/chat` | Main chat endpoint (see FR-10, FR-20) |
| PRD-0001-FR-26 | GET `/health` | Health check endpoint |
| PRD-0001-FR-27 | GET `/models` | List available models with their tiers |

### 4.5 Frontend

| FR ID | Requirement | Description |
|-------|-------------|-------------|
| PRD-0001-FR-28 | Chat interface | Input field, submit button, response display |
| PRD-0001-FR-29 | Show routing info | Display predicted difficulty, confidence, selected model |
| PRD-0001-FR-30 | Error handling | User-friendly error messages for failures |

### 4.6 Evaluation & Monitoring

| FR ID | Requirement | Description |
|-------|-------------|-------------|
| PRD-0001-FR-31 | Router evaluation | Comprehensive metrics (FR-8) + confidence analysis |
| PRD-0001-FR-32 | End-to-end evaluation | Routing accuracy, response quality, latency, cost, fallback frequency |
| PRD-0001-FR-33 | Baseline comparison | **Compare dynamic routing vs. single fixed LLM (Qwen3-4B-Instruct for all queries)** |
| PRD-0001-FR-34 | Dataset bias check | **Verify router learns difficulty patterns, not dataset/source/task identity**. Dataset-only baseline, Task-only baseline, Cross-dataset evaluation |
| PRD-0001-FR-35 | Model artifact metadata | Trained router records: embedding model name, label mapping used, training config |

---

## 5. Non-Functional Requirements

| NFR ID | Requirement | Target |
|--------|-------------|--------|
| PRD-0001-NFR-1 | Router inference latency | <100ms (embedding + MLP prediction) |
| PRD-0001-NFR-2 | API response time (Easy) | <2s p95 |
| PRD-0001-NFR-3 | API response time (Medium) | <5s p95 |
| PRD-0001-NFR-4 | API response time (Hard) | <15s p95 |
| PRD-0001-NFR-5 | Memory usage (router + embedding) | <2GB |
| PRD-0001-NFR-6 | Model loading | Lazy loading; don't load all LLMs simultaneously |
| PRD-0001-NFR-7 | Test coverage | ≥80% (unit + integration + E2E) |
| PRD-0001-NFR-8 | Security | Input validation, **minimal logging (no raw queries by default)**, basic API auth |
| PRD-0001-NFR-9 | Containerization | Docker multi-stage build, runs on CPU/GPU |
| PRD-0001-NFR-10 | Configuration | All tunable params in `config.yaml` |

---

## 6. Acceptance Criteria

### 6.1 Training Pipeline
- [ ] AC-001: Dataset loads without errors, schema validated
- [ ] AC-002: **Label mapping confirmed from data (not assumed) and documented**
- [ ] AC-003: Preprocessing preserves code/math/symbols, **no RAG chunking**
- [ ] AC-004: Embeddings generated for all samples, **cached for reuse**
- [ ] AC-005: **80/10/10 stratified splits**, no leakage
- [ ] AC-006: **MLP trains with Adam, Categorical Crossentropy, Softmax**, early stopping, best checkpoint saved
- [ ] AC-007: Evaluation report with **Accuracy, Precision, Recall, F1, Macro-F1, Per-class F1, Confusion Matrix**
- [ ] AC-008: Model saves to `models/router/best_model.keras` **with embedding model metadata and label mapping**, loads correctly
- [ ] AC-009: **Answer field NOT used as input feature** (data leakage check)

### 6.2 Inference Pipeline
- [ ] AC-010: POST `/chat` accepts query, returns structured response
- [ ] AC-011: Preprocessing matches training exactly
- [ ] AC-012: Embedding model reused (not reloaded per request)
- [ ] AC-013: Router predicts class + confidence correctly
- [ ] AC-014: **Router Policy (separate from NN) selects correct LLM per mapping**
- [ ] AC-015: **Confidence threshold respected (fallback when < 0.70 default)**
- [ ] AC-016: **Original query sent to LLM** (not embedding, not class name)
- [ ] AC-017: LLM response returned in final output
- [ ] AC-018: Error handling for empty query, model failures, OOM, timeouts
- [ ] AC-019: **Mock LLM pipeline works end-to-end before real model integration**
- [ ] AC-020: **Each LLM tested independently before complete pipeline integration**

### 6.3 Frontend
- [ ] AC-021: User can enter query, submit, see response
- [ ] AC-022: Predicted difficulty, confidence, model displayed
- [ ] AC-023: Errors displayed user-friendly

### 6.4 Evaluation
- [ ] AC-024: Router evaluation report with all metrics (Accuracy, Precision, Recall, F1, Macro-F1, Per-class F1, Confusion Matrix)
- [ ] AC-025: E2E evaluation with **baseline comparison (dynamic vs fixed LLM)**
- [ ] AC-026: **Bias check report showing router learns difficulty not dataset ID**
- [ ] AC-027: Model artifact includes embedding model and label mapping metadata

---

## 7. API Contract

### 7.1 POST `/chat`

**Request:**
```json
{
  "query": "string (required, 1-4096 chars)"
}
```

**Response (200):**
```json
{
  "query": "string",
  "predicted_class": "easy|medium|hard",
  "confidence": "float (0.0-1.0)",
  "selected_model": "string",
  "response": "string",
  "fallback_used": "boolean",
  "latency_ms": "integer"
}
```

**Error Responses:**
- 400: Empty/invalid query
- 500: Internal error (embedding, router, LLM failure)
- 503: Model loading/unavailable

### 7.2 GET `/health`
```json
{
  "status": "healthy|degraded",
  "router_loaded": "boolean",
  "embedding_loaded": "boolean",
  "models_loaded": "string[]"
}
```

### 7.3 GET `/models`
```json
{
  "models": [
    { "tier": "easy", "name": "Qwen/Qwen3-1.7B-Instruct", "loaded": "boolean" },
    { "tier": "medium", "name": "Qwen/Qwen3-4B-Instruct", "loaded": "boolean" },
    { "tier": "hard", "name": "meta-llama/Llama-3.2-3B-Instruct", "loaded": "boolean" }
  ]
}
```

---

## 8. Architecture & Data Flow

### 8.1 Offline Training Flow
```
Dataset (JSON)
    ↓
Data Validation
    ↓
Preprocessing
    ↓
Label Validation (verify one-hot mapping)
    ↓
Train/Val/Test Split (80/10/10 stratified)
    ↓
Text Embedding (Sentence Transformers) + Caching
    ↓
Keras MLP Training (Adam, Categorical Crossentropy, Softmax)
    ↓
Validation + Evaluation (Accuracy, Precision, Recall, F1, Macro-F1, Per-class F1, Confusion Matrix)
    ↓
Save Trained Router (models/router/best_model.keras + metadata)
```

### 8.2 Online Inference Flow
```
USER QUERY
    ↓
VALIDATION
    ↓
PREPROCESSING
    ↓
EMBEDDING (cached model)
    ↓
KERAS MLP ROUTER
    ↓
PREDICTED CLASS + CONFIDENCE
    ↓
ROUTER POLICY (threshold check)
    ↓
SELECTED LLM (from MODEL_REGISTRY)
    ↓
ORIGINAL USER QUERY
    ↓
LLM RESPONSE
    ↓
FINAL RESPONSE
```

### 8.3 Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| Embedding Model | Text → Vector |
| Keras MLP Router | Vector → Class Probabilities (Difficulty Class) |
| Router Policy | Class + Confidence → Selected LLM |
| Model Registry | Tier → Model Name mapping |
| Selected LLM | Original Query → Final Answer |

---

## 9. Model Registry

```python
MODEL_REGISTRY = {
    "easy": "qwen3:1.7b",
    "medium": "qwen3:4b",
    "hard": "llama3.2:3b"
}
```

**Note:** All three models are served via **Ollama** (local inference, no API calls). Models are pre-downloaded and quantized (Q4_K_M).

---

## 10. Configuration (config.yaml)

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
  timeout: 120
  models:
    easy: "qwen3:1.7b"
    medium: "qwen3:4b"
    hard: "llama3.2:3b"

# Kept for compatibility but not used when use_ollama=true
models:
  easy: "Qwen/Qwen3-1.7B-Instruct"
  medium: "Qwen/Qwen3-4B-Instruct"
  hard: "meta-llama/Llama-3.2-3B-Instruct"
  device_map: "auto"
  load_in_8bit: true
  max_memory_gb: 12

generation:
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  do_sample: true

api:
  host: "0.0.0.0"
  port: 8000
  timeout: 120
  workers: 1

features:
  use_mock_llms: true
  use_ollama: true
  enable_fallback: true
  enable_safety_validation: false

logging:
  log_raw_queries: false
  log_fields: ["request_id", "predicted_class", "confidence", "selected_model", "fallback_used", "latency_ms", "error"]
  level: "INFO"

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 10
  validation_split: 0.1
  test_split: 0.1
  class_weights: true
  random_seed: 42

data:
  dataset_path: "data/combined_labels_one_hot.jsonl"
  text_field: "prompt"
  label_field: "labels_one_hot"
  preprocessing:
    strip_whitespace: true
    normalize_unicode: true
    preserve_math: true
    preserve_code: true
    preserve_symbols: true
```

---

## 11. Dependencies & Predecessors

### 11.1 External Dependencies
- Dataset: `data/combined_labels_one_hot.jsonl`
- **Ollama** running locally (models pre-downloaded: qwen3:1.7b, qwen3:4b, llama3.2:3b)
- GPU/CPU with sufficient VRAM/RAM for LLMs (Ollama handles quantization)

### 11.2 Internal Dependencies
- Training pipeline must complete before inference works
- Embedding model must be same for training and inference
- Model registry must be consistent across training/policy/inference
- Mock LLM pipeline must work before real model integration
- Each LLM tested independently before complete pipeline integration

---

## 12. Implementation Order (Checkpoint Rule: BUILD → TEST → VERIFY → FIX → CHECKPOINT → CONTINUE)

Do NOT continue to the next task until the current task works.

| Task | Description | Dependencies |
|------|-------------|--------------|
| **Task 0** | **Project Setup** | - |
| **Task 1** | **Dataset Loading & Validation** | Task 0 |
| **Task 2** | **Text Preprocessing** | Task 1 |
| **Task 3** | **Embedding Pipeline (with caching)** | Task 2 |
| **Task 4** | **ML Dataset Preparation (80/10/10 split, label verification)** | Task 3 |
| **Task 5** | **Keras MLP Router (Embedding→Dense→ReLU→Dense→ReLU→Dense(3)→Softmax)** | Task 4 |
| **Task 6** | **Router Training & Evaluation (Adam, CatCrossentropy, metrics)** | Task 5 |
| **Task 7** | **New Query Prediction** | Task 6 |
| **Task 8** | **Router Policy (threshold, fallback, registry)** | Task 7 |
| **Task 9** | **Mock LLM Pipeline (end-to-end with mocks)** | Task 8 |
| **Task 10** | **Ollama LLM Client** | Task 9 |
| **Task 11** | **qwen3:1.7b Integration (Easy Tier)** | Task 10 |
| **Task 12** | **qwen3:4b Integration (Medium Tier)** | Task 10 |
| **Task 13** | **llama3.2:3b Integration (Hard Tier)** | Task 10 |
| **Task 14** | **Complete Inference Pipeline** | Tasks 11-13 |
| **Task 15** | **FastAPI Backend** | Task 14 |
| **Task 16** | **Required Advanced Features** | Task 15 |
| **Task 17** | **Testing (unit, integration, E2E)** | Task 16 |
| **Task 18** | **React Frontend** | Task 15 |
| **Task 19** | **Docker** | Task 17 |
| **Task 20** | **AWS Deployment** | Task 19 |
| **Task 21** | **Final Evaluation (router metrics, E2E, baseline, bias check)** | Task 17 |

---

## 13. Test Strategy

### 13.1 Unit Tests (Required per CLAUDE.md)
- Preprocessing functions
- Embedding generation + caching
- Router prediction logic
- Router policy (threshold, fallback, registry)
- Model registry
- Configuration loading

### 13.2 Integration Tests (Required per CLAUDE.md)
- API endpoints (`/chat`, `/health`, `/models`)
- Model loading (embedding, router, **Ollama client**)
- Full inference pipeline (mocked LLMs)
- Training pipeline components

### 13.3 E2E Tests (Required - Frontend-Backend)
- User submits query → sees response with routing info
- Different difficulty queries route to correct models
- Fallback behavior when confidence low
- Error handling display

### 13.4 Evaluation Tests
- Router metrics on test set: **Accuracy, Precision, Recall, F1-score, Macro-F1, Per-class F1, Confusion Matrix**
- E2E routing accuracy
- **Baseline comparison: dynamic routing vs. single fixed LLM (qwen3:4b for all)**
- **Dataset bias check: verify router learns difficulty, not dataset/source/task**

---

## 14. Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `USE_MOCK_LLMS` | true | **Use mock responses for dev/testing (start with true)** |
| `ENABLE_FALLBACK` | true | Enable confidence-based fallback |
| `ENABLE_SAFETY_VALIDATION` | false | Enable Llama Guard safety check |

---

## 15. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LLMs too large for memory | High | High | 8-bit quantization, lazy loading, CPU offload |
| Dataset bias (learns source not difficulty) | Medium | High | **Bias check evaluation (FR-34), cross-dataset eval** |
| Low router accuracy | Medium | High | Iterate on architecture, more data, ensemble |
| Confidence miscalibration | Medium | Medium | Temperature scaling, Platt scaling |
| Data leakage (answer field used) | Low | Critical | **Explicit prohibition in FR-3, AC-009** |

---

## 16. Operational Readiness

### 16.1 Deployment
- Docker container with multi-stage build
- Health check endpoint for orchestration
- Config via environment variables / config.yaml

### 16.2 Monitoring
- **Log: request_id, predicted_class, confidence, selected_model, fallback_used, latency_ms, error**
- **Do NOT log raw user queries by default**
- Metrics: routing distribution, fallback rate, latency percentiles, error rate

### 16.3 Rollback
- Previous model version in `models/router/`
- Blue-green deployment via Docker tags

---

## 17. Definition of Done

- [ ] All ACs in Section 6 passing
- [ ] Test coverage ≥80% (unit + integration + E2E)
- [ ] Quality gates: ruff, mypy, bandit, pytest-cov all pass
- [ ] **Checkpoint rule followed: each task BUILD→TEST→VERIFY→FIX→CHECKPOINT before next**
- [ ] Docker image builds and runs
- [ ] Evaluation reports generated (router metrics + E2E + baseline comparison + bias check)
- [ ] README with setup, usage, architecture docs
- [ ] No critical/security findings

---

## 18. Traceability Matrix

| FR/NFR | Unit Tests | Integration Tests | E2E Tests | Evaluation |
|--------|------------|-------------------|-----------|------------|
| FR-1 to FR-9 | ✅ | ✅ | - | ✅ |
| FR-10 to FR-20 | ✅ | ✅ | ✅ | ✅ |
| FR-21 to FR-24 | ✅ | ✅ | - | - |
| FR-25 to FR-27 | ✅ | ✅ | ✅ | - |
| FR-28 to FR-30 | - | - | ✅ | - |
| FR-31 to FR-35 | - | - | - | ✅ |
| NFR-1 to NFR-10 | - | ✅ | ✅ | ✅ |

---

## 19. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-09-13 | Initial PRD from master prompt |
| 1.1 | 2026-09-13 | Applied 28 required changes: TensorFlow/Keras, MLP architecture, data leakage rule, label verification, 80/10/10 split, Adam/CatCrossentropy/Softmax, embedding caching, no RAG chunking, component separation, original query to LLM, confidence threshold/fallback, mock LLMs first, modular LLM integration, model mapping, Llama Guard note, implementation order with checkpoint rule, router metrics, 85% target, bias check, baseline comparison, model artifact metadata, minimal logging, final pipeline |
| 1.2 | 2026-09-17 | Updated models: Easy→Qwen3-1.7B, Medium→Qwen3-4B, Hard→Llama-3.2-3B. New dataset: combined_labels_one_hot.jsonl (multi-label format). Removed Llama Guard note. |
| 1.3 | 2026-09-17 | **Switched to Ollama for local LLM inference. Models: qwen3:1.7b (Easy), qwen3:4b (Medium), llama3.2:3b (Hard). No Hugging Face API calls needed.** |