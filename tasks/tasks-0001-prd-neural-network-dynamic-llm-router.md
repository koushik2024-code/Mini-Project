# Task List: PRD-0001 Neural Network Based Dynamic LLM Router

## Parent Tasks with Sub-Tasks

### 1.0 [✅] Project Setup
**FR/NFR:** NFR-7, NFR-9, NFR-10

- [x] 1.1 Create project directory structure (data/, embeddings/, router/, llm/, api/, evaluation/, models/, config/, frontend/, tests/)
- [x] 1.2 Create requirements.txt with all dependencies (tensorflow, keras, sentence-transformers, transformers, fastapi, uvicorn, pandas, numpy, scikit-learn, pyyaml, pytest, ruff, mypy, bandit, pytest-cov, ollama)
- [x] 1.3 Create config.yaml with all configurable parameters (embedding, router, ollama, generation, api, features, logging)
- [x] 1.4 Create .gitignore, .env.example, Dockerfile (multi-stage), docker-compose.yml
- [x] 1.5 Set up pytest configuration (pytest.ini) with coverage target 80%
- [x] 1.6 Set up ruff, mypy, bandit configuration
- [x] 1.7 Verify environment: Python 3.10+, GPU/CPU detection, Ollama running
- [x] 1.8 Run initial quality gates: `ruff check .`, `mypy .`, `bandit -r .`, `pytest --collect-only`

### 2.0 [✅] Dataset Loading & Validation
**FR/NFR:** FR-1, AC-001

- [x] 2.1 Write unit tests for dataset loading (PRD-0001-FR-1)
- [x] 2.2 Implement dataset loader: load `data/combined_labels_one_hot.jsonl`
- [x] 2.3 Implement schema validation: check required fields (id, task, prompt, answer, dataset, labels_one_hot)
- [x] 2.4 Implement missing value detection and reporting
- [x] 2.5 Run unit tests and fix failures
- [x] 2.6 **CHECKPOINT:** Dataset loads without errors, schema validated

### 3.0 [✅] Text Preprocessing
**FR/NFR:** FR-3, AC-003, AC-009

- [x] 3.1 Write unit tests for preprocessing (PRD-0001-FR-3): whitespace cleaning, text normalization, math/code/symbol preservation, NO RAG chunking, NO answer field usage
- [x] 3.2 Implement preprocessing function: remove unnecessary whitespace, normalize unicode, preserve mathematical expressions, preserve programming code, preserve important symbols
- [x] 3.3 Add validation: ensure answer field is NEVER used as input feature
- [x] 3.4 Run unit tests and fix failures
- [x] 3.5 **CHECKPOINT:** Preprocessing preserves code/math/symbols, no RAG chunking, no answer field leakage

### 4.0 [✅] Embedding Pipeline with Caching
**FR/NFR:** FR-4, AC-004

- [x] 4.1 Write unit tests for embedding generation and caching (PRD-0001-FR-4)
- [x] 4.2 Implement embedding model loader (Sentence Transformers, configurable via config.yaml)
- [x] 4.3 Implement embedding generation for single query and batch
- [x] 4.4 Implement embedding cache: save/load embeddings to disk (embeddings/cache/), reuse for repeated queries
- [x] 4.5 Implement cache invalidation when embedding model changes
- [x] 4.6 Run unit tests and fix failures
- [x] 4.7 **CHECKPOINT:** Embeddings generated for all samples, cached for reuse

### 5.0 [✅] ML Dataset Preparation
**FR/NFR:** FR-2, FR-5, AC-002, AC-005

- [x] 5.1 Write unit tests for label mapping verification and data splitting (PRD-0001-FR-2, FR-5)
- [x] 5.2 Implement label mapping verification: extract unique one-hot vectors from labels_one_hot, map to Easy/Medium/Hard, document actual mapping (DO NOT assume)
- [x] 5.3 Implement 80/10/10 stratified split (train/validation/test) using labels_one_hot
- [x] 5.4 Verify no data leakage between splits
- [x] 5.5 Save split indices and label mapping for reproducibility
- [x] 5.6 Run unit tests and fix failures
- [x] 5.7 **CHECKPOINT:** Label mapping confirmed from data, 80/10/10 stratified splits, no leakage

### 6.0 [✅] Keras MLP Router
**FR/NFR:** FR-6, NFR-1

- [x] 6.1 Write unit tests for MLP model architecture (PRD-0001-FR-6)
- [x] 6.2 Implement Keras MLP: Input(embedding_dim) → Dense(256) → ReLU → Dense(128) → ReLU → Dense(3) → Softmax
- [x] 6.3 Compile model: Adam optimizer, Categorical Crossentropy loss, metrics=['accuracy']
- [x] 6.4 Implement model summary and parameter counting
- [x] 6.5 Run unit tests and fix failures
- [x] 6.6 **CHECKPOINT:** MLP architecture correct: Embedding → Dense → ReLU → Dense → ReLU → Dense(3) → Softmax

### 7.0 [✅] Router Training & Evaluation
**FR/NFR:** FR-7, FR-8, FR-31, AC-006, AC-007, AC-024

- [x] 7.1 Write unit tests for training loop and evaluation metrics (PRD-0001-FR-7, FR-8)
- [x] 7.2 Implement training: Adam optimizer, Categorical Crossentropy, Early Stopping (patience=10), Model Checkpointing (save best), Class Weights (if imbalanced)
- [x] 7.3 Implement validation during training
- [x] 7.4 Implement evaluation on test set: Accuracy, Precision, Recall, F1-score, Macro-F1, Per-class F1, Confusion Matrix
- [x] 7.5 Implement confidence calibration analysis (ECE)
- [x] 7.6 Generate evaluation report (JSON + visualizations)
- [x] 7.7 Run unit tests and fix failures
- [x] 7.8 **CHECKPOINT:** MLP trains with Adam/CatCrossentropy/Softmax, early stopping, best checkpoint saved; evaluation report with all metrics

### 8.0 [  ] Model Artifact & Metadata
**FR/NFR:** FR-9, FR-35, AC-008, AC-027

- [x] 8.1 Write unit tests for model saving/loading with metadata (PRD-0001-FR-9, FR-35)
- [x] 8.2 Save trained router to `models/router/best_model.keras`
- [x] 8.3 Save metadata: embedding model name, label mapping (one-hot → class), training config (optimizer, loss, epochs, split ratios)
- [x] 8.4 Implement model loading with metadata verification
- [x] 8.5 Run unit tests and fix failures
- [x] 8.6 **CHECKPOINT:** Model saves to models/router/best_model.keras with embedding model metadata and label mapping, loads correctly

### 9.0 [✅] New Query Prediction
**FR/NFR:** FR-11, FR-12, FR-13, FR-14, AC-011, AC-012, AC-013

- [x] 9.1 Write unit tests for prediction pipeline (PRD-0001-FR-11 to FR-14)
- [x] 9.2 Implement query preprocessing (reuse from Task 3.0)
- [x] 9.3 Implement query embedding (reuse cached embedding model from Task 4.0)
- [x] 9.4 Implement router prediction: embedding → class probabilities → predicted class + confidence (max prob)
- [x] 9.5 Verify preprocessing matches training exactly
- [x] 9.6 Run unit tests and fix failures
- [x] 9.7 **CHECKPOINT:** Router predicts class + confidence correctly, embedding model reused

### 10.0 [✅] Router Policy Engine
**FR/NFR:** FR-15, FR-16, FR-21, FR-22, FR-23, AC-014, AC-015

- [x] 10.1 Write unit tests for router policy (PRD-0001-FR-15, FR-16, FR-21, FR-22, FR-23)
- [x] 10.2 Implement MODEL_REGISTRY: {"easy": "qwen3:1.7b", "medium": "qwen3:4b", "hard": "llama3.2:3b"}
- [x] 10.3 Implement confidence threshold logic (configurable, default 0.70)
- [x] 10.4 Implement fallback strategy: confidence < threshold → fallback_tier (configurable, default "medium")
- [x] 10.5 Ensure Router Policy is SEPARATE from neural network (MLP predicts, Policy decides LLM)
- [x] 10.6 Run unit tests and fix failures
- [x] 10.7 **CHECKPOINT:** Router Policy selects correct LLM per mapping, confidence threshold respected, fallback works

### 11.0 [✅] Mock LLM Pipeline
**FR/NFR:** FR-18, FR-20, AC-016, AC-017, AC-019

- [x] 11.1 Write integration tests for mock LLM pipeline (PRD-0001-FR-18, FR-20)
- [x] 11.2 Implement mock LLM responses for each tier (Easy/Medium/Hard)
- [x] 11.3 Implement complete pipeline: Query → Preprocessing → Embedding → Router → Policy → Mock LLM → Response
- [x] 11.4 Verify ORIGINAL USER QUERY sent to mock LLM (not embedding, not class name)
- [x] 11.5 Verify structured response: {query, predicted_class, confidence, selected_model, response, fallback_used, latency_ms}
- [x] 11.6 Run integration tests and fix failures
- [x] 11.7 **CHECKPOINT:** Mock LLM pipeline works end-to-end, original query sent to LLM

### 12.0 [✅] Ollama LLM Client
**FR/NFR:** NFR-5, NFR-6

- [x] 12.1 Write unit tests for Ollama client (NFR-5, NFR-6)
- [x] 12.2 Implement Ollama client wrapper (connect to localhost:11434)
- [x] 12.3 Implement chat completion method with generation config
- [x] 12.4 Implement model listing and health check
- [x] 12.5 Handle connection errors, timeouts, model not found
- [x] 12.6 Run unit tests and fix failures
- [x] 12.7 **CHECKPOINT:** Ollama client connects, lists models, generates responses

### 13.0 [✅] qwen3:1.7b Integration (Easy Tier)
**FR/NFR:** FR-17, AC-020

- [x] 13.1 Write integration tests for qwen3:1.7b integration (PRD-0001-FR-17)
- [x] 13.2 Test qwen3:1.7b via Ollama client
- [x] 13.3 Implement generation with config parameters (max_new_tokens, temperature, top_p)
- [x] 13.4 Test qwen3:1.7b independently with sample queries
- [x] 13.5 Run integration tests and fix failures
- [x] 13.6 **CHECKPOINT:** qwen3:1.7b generates responses, tested independently

### 14.0 [✅] qwen3:1.7b Integration (Medium Tier)
**FR/NFR:** FR-17, AC-020

- [x] 14.1 Write integration tests for qwen3:1.7b integration (PRD-0001-FR-17)
- [x] 14.2 Test qwen3:1.7b via Ollama client
- [x] 14.3 Implement generation with config parameters
- [x] 14.4 Test qwen3:1.7b independently with sample queries
- [x] 14.5 Run integration tests and fix failures
- [x] 14.6 **CHECKPOINT:** qwen3:1.7b generates responses, tested independently

**Note:** Using qwen3:1.7b for medium tier as qwen3:4b is too slow on CPU.

### 15.0 [✅] llama3.2:3b Integration (Hard Tier)
**FR/NFR:** FR-17, AC-020

- [x] 15.1 Write integration tests for llama3.2:3b integration (PRD-0001-FR-17)
- [x] 15.2 Test llama3.2:3b via Ollama client
- [x] 15.3 Implement generation with config parameters
- [x] 15.4 Test llama3.2:3b independently with sample queries
- [x] 15.5 Run integration tests and fix failures
- [x] 15.6 **CHECKPOINT:** llama3.2:3b generates responses, tested independently

**Note:** Using llama3.2:3b instead of Llama Guard 4 12B (which is a safety model, not a generation model).

### 16.0 [✅] Complete Inference Pipeline
**FR/NFR:** FR-10, FR-18, FR-19, FR-20, AC-010, AC-016, AC-017, AC-018

- [x] 16.1 Write integration tests for complete inference pipeline (PRD-0001-FR-10, FR-18, FR-19, FR-20)
- [x] 16.2 Integrate all components: Validation → Preprocessing → Embedding → Router → Policy → Ollama LLM → Response
- [x] 16.3 Implement optional safety validation (feature flag)
- [x] 16.4 Implement error handling: empty query, model failures, OOM, timeouts
- [x] 16.5 Verify end-to-end flow with real Ollama LLMs
- [x] 16.6 Run integration tests and fix failures
- [x] 16.7 **CHECKPOINT:** Complete pipeline works with real Ollama LLMs, error handling functional

### 17.0 [✅] FastAPI Backend
**FR/NFR:** FR-25, FR-26, FR-27, NFR-8, NFR-10

- [x] 17.1 Write integration tests for API endpoints (PRD-0001-FR-25, FR-26, FR-27)
- [x] 17.2 Implement FastAPI app with config.yaml loading
- [x] 17.3 Implement POST /chat endpoint with request validation
- [x] 17.4 Implement GET /health endpoint
- [x] 17.5 Implement GET /models endpoint
- [x] 17.6 Implement minimal logging: request_id, predicted_class, confidence, selected_model, fallback_used, latency_ms, error (NO raw queries)
- [x] 17.7 Implement input validation and error responses (400, 500, 503)
- [x] 17.8 Run integration tests and fix failures
- [x] 17.9 **CHECKPOINT:** All API endpoints work, validation, logging, error handling

### 18.0 [✅] Required Advanced Features
**FR/NFR:** NFR-2, NFR-3, NFR-4, NFR-8

- [x] 18.1 Write tests for advanced features (NFR-2, NFR-3, NFR-4)
- [x] 18.2 Implement request timeout handling (configurable)
- [x] 18.3 Implement feature flags: USE_MOCK_LLMS, USE_OLLAMA, ENABLE_FALLBACK, ENABLE_SAFETY_VALIDATION
- [x] 18.4 Implement confidence calibration (temperature scaling if needed)
- [x] 18.5 Verify latency targets: Easy <2s p95, Medium <5s p95, Hard <15s p95
- [x] 18.6 Run tests and fix failures
- [x] 18.7 **CHECKPOINT:** Timeouts, feature flags, calibration, latency targets met

### 19.0 [✅] Testing (Unit, Integration, E2E)
**FR/NFR:** NFR-7, FR-31, FR-32, FR-33, FR-34

- [x] 19.1 Run all unit tests (preprocessing, embedding, router, policy, registry, config) - target ≥80% coverage
- [x] 19.2 Run all integration tests (API endpoints, model loading, pipeline)
- [x] 19.3 Write E2E tests for frontend-backend chat flow: user submits query → sees response with routing info
- [x] 19.4 Run E2E tests: different difficulty queries route to correct models, fallback behavior, error handling
- [x] 19.5 Run router evaluation on test set: Accuracy, Precision, Recall, F1, Macro-F1, Per-class F1, Confusion Matrix
- [x] 19.6 Run E2E evaluation: routing accuracy, response quality, latency, cost, fallback frequency
- [x] 19.7 Run baseline comparison: dynamic routing vs. fixed qwen3:1.7b for all queries
- [x] 19.8 Run dataset bias check: verify router learns difficulty not dataset/source/task (dataset-only baseline, task-only baseline, cross-dataset eval)
- [x] 19.9 Run quality gates: `ruff check .`, `mypy .`, `bandit -r .`, `pytest --cov=80`
- [x] 19.10 **CHECKPOINT:** All tests pass, coverage ≥80%, quality gates pass, evaluation reports generated

### 20.0 [✅] React Frontend
**FR/NFR:** FR-28, FR-29, FR-30, AC-021, AC-022, AC-023

- [x] 20.1 Initialize React app in frontend/ (Vite + TypeScript + Tailwind CSS v4)
- [x] 20.2 Implement professional chat interface: input field, submit button, response display
- [x] 20.3 Implement routing info display: predicted difficulty, confidence, selected model
- [x] 20.4 Implement error handling display (user-friendly)
- [x] 20.4 Implement responsive design with dark mode support
- [x] 20.5 **CHECKPOINT:** Frontend works, displays routing info, handles errors, looks professional

### 21.0 [  ] Docker
**FR/NFR:** NFR-9

- [ ] 21.1 Write Dockerfile (multi-stage: build → runtime)
- [ ] 21.2 Configure for CPU and GPU (separate Dockerfiles or build args)
- [ ] 21.3 Add health check endpoint
- [ ] 21.4 Test Docker build and run locally
- [ ] 21.5 **CHECKPOINT:** Docker image builds, runs on CPU/GPU, health check works

### 22.0 [  ] AWS Deployment
**FR/NFR:** NFR-9

- [ ] 22.1 Create ECS/EKS deployment configuration
- [ ] 22.2 Create ECR repository setup
- [ ] 22.3 Configure load balancer, auto-scaling
- [ ] 22.4 Set up CloudWatch logging and monitoring
- [ ] 22.5 Document deployment steps
- [ ] 22.6 **CHECKPOINT:** Deployment configuration ready

### 23.0 [✅] Final Evaluation & Documentation
**FR/NFR:** FR-31, FR-32, FR-33, FR-34, AC-024, AC-025, AC-026

- [x] 23.1 Compile final router evaluation report (all metrics)
- [x] 23.2 Compile E2E evaluation report with baseline comparison
- [x] 23.3 Compile dataset bias check report
- [x] 23.4 Create comprehensive README: setup, usage, architecture, API docs, evaluation results
- [x] 23.5 Verify all ACs from Section 6 passing
- [x] 23.6 **CHECKPOINT:** All evaluation reports complete, README comprehensive, all ACs passing

---

## Checkpoint Rule
**BUILD → TEST → VERIFY → FIX → CHECKPOINT → CONTINUE**

Do NOT continue to the next task until the current task works.

## Global Dependency Tracking
See `tasks/_index.md` for cross-task dependencies and readiness status.