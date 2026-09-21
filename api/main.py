import logging
import os
import time
import uuid
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from auth.google_auth import GoogleAuthConfig, GoogleAuthError, GoogleVerifier
from auth.session import SessionConfig, SessionError, SessionManager
from db.history_store import HistoryConfig, HistoryStore
from db.user_store import UserConfig, UserStore
from llm.mock_llm import MockLLM, MockLLMConfig
from llm.ollama_client import OllamaClient, OllamaConfig
from router.predictor import PredictorConfig, QueryPredictor
from router.router_policy import PolicyConfig, RouterPolicy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load .env before anything reads os.environ. Real environment variables
# already set take precedence, so deployments can override the file.
load_dotenv()

# Load configuration at module level
config_path = Path("config.yaml")
config = {}
if config_path.exists():
    with open(config_path) as f:
        config = yaml.safe_load(f)

# Initialize configuration objects
predictor_config = PredictorConfig(
    router_model_path=config.get('router', {}).get('model_path', "models/router/best_model.keras"),
    embedding_model_name=config.get('embedding', {}).get('model', "sentence-transformers/all-MiniLM-L6-v2"),
    embedding_device=config.get('embedding', {}).get('device', "cpu"),
    embedding_cache_dir=config.get('embedding', {}).get('cache_dir', "embeddings/cache")
)

policy_config = PolicyConfig(
    confidence_threshold=config.get('router', {}).get('confidence_threshold', 0.70),
    fallback_tier=config.get('router', {}).get('fallback_tier', "medium"),
    model_registry={
        "easy": config.get('ollama', {}).get('models', {}).get('easy', "qwen3:1.7b"),
        "medium": config.get('ollama', {}).get('models', {}).get('medium', "qwen3:1.7b"),
        "hard": config.get('ollama', {}).get('models', {}).get('hard', "llama3.2:3b")
    }
)

# Initialize LLM client
features = config.get('features', {})
use_mock_llms = features.get('use_mock_llms', True)

if use_mock_llms:
    llm_client = MockLLM(MockLLMConfig(
        responses={
            "easy": "This is a simple, direct answer from the Easy model.",
            "medium": "This is a detailed, well-structured answer from the Medium model with examples and explanations.",
            "hard": "This is a comprehensive, in-depth answer from the Hard model with thorough analysis, edge cases, and advanced concepts."
        }
    ))
    logger.info("Using Mock LLM")
else:
    ollama_config = OllamaConfig(
        host=config.get('ollama', {}).get('host', "http://localhost:11434"),
        timeout=config.get('ollama', {}).get('timeout', 180),
        models={
            "easy": config.get('ollama', {}).get('models', {}).get('easy', "qwen3:1.7b"),
            "medium": config.get('ollama', {}).get('models', {}).get('medium', "qwen3:1.7b"),
            "hard": config.get('ollama', {}).get('models', {}).get('hard', "llama3.2:3b")
        }
    )
    llm_client = OllamaClient(ollama_config)
    logger.info("Using Ollama LLM")

# Initialize search history store (SQLite)
# LLM_ROUTER_HISTORY_DB overrides the configured path (tests use ":memory:")
history_config = HistoryConfig(
    db_path=os.environ.get(
        "LLM_ROUTER_HISTORY_DB",
        config.get('database', {}).get('history_path', "data/history.db")
    ),
    max_entries=config.get('database', {}).get('history_max_entries', 500)
)
history_enabled = features.get('enable_history', True)
history_store = HistoryStore(history_config) if history_enabled else None

# Users live in the same SQLite file as the history
user_store = UserStore(UserConfig(db_path=history_config.db_path))

# Google Sign-In verification + our own session tokens.
# Secrets come from the environment; config.yaml holds only non-secret tuning.
auth_config = config.get('auth', {})
google_verifier = GoogleVerifier(GoogleAuthConfig(
    client_id=os.environ.get("GOOGLE_CLIENT_ID", auth_config.get('google_client_id', "")),
    require_verified_email=auth_config.get('require_verified_email', True)
))
session_manager = SessionManager(SessionConfig(
    secret=os.environ.get("SESSION_SECRET", ""),
    ttl_hours=auth_config.get('session_ttl_hours', 24)
))

if not google_verifier.configured:
    logger.warning(
        "GOOGLE_CLIENT_ID is not set - sign-in will fail until it is configured"
    )

# Initialize components at module level (works with TestClient)
predictor = QueryPredictor(predictor_config)
policy = RouterPolicy(policy_config)

logger.info("All components initialized successfully")


# Create FastAPI app
app = FastAPI(
    title="Neural Network Based Dynamic LLM Router",
    description="Dynamic LLM routing system that selects appropriate LLM based on query difficulty",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="User query")


class ChatResponse(BaseModel):
    query: str
    predicted_class: str
    confidence: float
    selected_model: str
    response: str
    fallback_used: bool
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    router_loaded: bool
    embedding_loaded: bool
    models_loaded: list[str]


class ModelInfo(BaseModel):
    tier: str
    name: str
    loaded: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class GoogleLoginRequest(BaseModel):
    credential: str = Field(..., min_length=1, description="Google ID token")


class UserInfo(BaseModel):
    id: int
    email: str
    name: str
    picture: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: UserInfo


class HistoryEntry(BaseModel):
    id: int
    query: str
    response: str
    predicted_class: str
    confidence: float
    selected_model: str
    fallback_used: bool
    latency_ms: int
    created_at: str


class HistoryResponse(BaseModel):
    entries: list[HistoryEntry]
    total: int
    limit: int
    offset: int


# Bearer scheme; auto_error off so we can return our own 401 body
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)
) -> dict:
    """Resolve the signed-in user from the Authorization header.

    Raises 401 when the token is absent, malformed, expired, or points at a
    user who no longer exists.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )

    try:
        claims = session_manager.verify(credentials.credentials)
    except SessionError as exc:
        raise HTTPException(
            status_code=401,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"}
        ) from exc

    user = user_store.get(claims["user_id"])
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="User no longer exists",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return user


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # Log request
    logger.info(f"[{request_id}] {request.method} {request.url.path}")

    response = await call_next(request)

    # Log response
    latency_ms = int((time.time() - start_time) * 1000)
    logger.info(f"[{request_id}] {response.status_code} - {latency_ms}ms")

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-MS"] = str(latency_ms)

    return response


@app.post("/auth/google")
async def google_login(request: GoogleLoginRequest):
    """Exchange a Google ID token for one of our session tokens."""
    try:
        profile = google_verifier.verify(request.credential)
    except GoogleAuthError as exc:
        logger.warning(f"Google sign-in rejected: {exc}")
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    user = user_store.upsert(
        google_sub=profile["google_sub"],
        email=profile["email"],
        name=profile["name"],
        picture=profile["picture"]
    )
    token = session_manager.issue(user["id"], user["email"])
    logger.info(f"Signed in user id={user['id']}")

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": session_manager.ttl_seconds,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "picture": user["picture"]
        }
    }


@app.get("/auth/me")
async def read_current_user(current_user: dict = Depends(get_current_user)):
    """Return the signed-in user's profile."""
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "name": current_user["name"],
        "picture": current_user["picture"]
    }


@app.post("/auth/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Sign out.

    Session tokens are stateless, so the server has nothing to revoke - the
    client discards the token. Kept as an endpoint so the frontend has one
    place to call and we get a log line.
    """
    logger.info(f"Signed out user id={current_user['id']}")
    return {"detail": "Signed out"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if predictor and predictor.router else "degraded",
        "router_loaded": predictor is not None and predictor.router is not None,
        "embedding_loaded": predictor is not None and predictor.embedding_model is not None,
        "models_loaded": list(policy.get_all_models().values()),
        "auth_configured": google_verifier.configured,
        "confidence_threshold": policy.config.confidence_threshold
    }


@app.get("/models")
async def list_models(current_user: dict = Depends(get_current_user)):
    """List available models"""
    model_infos = [
        {"tier": tier, "name": name, "loaded": True}
        for tier, name in policy.get_all_models().items()
    ]
    return {"models": model_infos}


def _validate_query(request: dict) -> str:
    """Pull the query out of a request body and validate it."""
    query = request.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(query) > 4096:
        raise HTTPException(status_code=400, detail="Query too long (max 4096 characters)")
    return query


def _classify(query: str) -> dict:
    """Run the router: preprocess, embed, predict, then apply the policy.

    This is everything before the LLM call, and it is the cheap half - a few
    milliseconds against seconds of generation. Kept separate so /route can
    return the routing decision on its own.
    """
    processed = predictor.preprocessor.preprocess(query)
    embedding = predictor.embedding_model.encode(processed)

    probs = predictor.router.predict(embedding.reshape(1, -1))[0]
    pred_idx = int(probs.argmax())
    confidence = float(probs[pred_idx])
    predicted_class = ["easy", "medium", "hard"][pred_idx]

    policy_result = policy.select_model(predicted_class, confidence)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "selected_tier": policy_result["selected_tier"],
        "selected_model": policy_result["selected_model"],
        "fallback_used": policy_result.get("fallback_used", False),
        "confidence_threshold": policy.config.confidence_threshold
    }


@app.post("/route")
async def route_only(request: dict, current_user: dict = Depends(get_current_user)):
    """Classify a query and return the routing decision without generating.

    Generation takes seconds; routing takes milliseconds. The UI calls this
    first so it can show which model was picked, and why, while the answer
    is still being written.
    """
    query = _validate_query(request)
    start_time = time.time()

    try:
        decision = _classify(query)
    except Exception as e:
        logger.error(f"Error routing query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e

    decision["latency_ms"] = int((time.time() - start_time) * 1000)
    return decision


@app.post("/chat")
async def chat(request: dict, current_user: dict = Depends(get_current_user)):
    """
    Main chat endpoint - routes query to appropriate LLM based on predicted difficulty.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]

    # 1. Validate query
    query = _validate_query(request)

    try:
        # 2-5. Preprocess, embed, predict, apply router policy
        decision = _classify(query)
        predicted_class = decision["predicted_class"]
        confidence = decision["confidence"]

        # 6. Send ORIGINAL query to selected LLM
        llm_result = llm_client.chat(decision["selected_tier"], query)

        # 7. Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Build response
        response = {
            "query": query,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "selected_model": decision["selected_model"],
            "response": llm_result.get("response", ""),
            "fallback_used": decision["fallback_used"],
            "latency_ms": latency_ms,
            "confidence_threshold": decision["confidence_threshold"]
        }

        # 8. Persist to search history (SQL)
        if history_store is not None:
            try:
                entry = history_store.add(
                    user_id=current_user["id"],
                    query=query,
                    response=response["response"],
                    predicted_class=predicted_class,
                    confidence=confidence,
                    selected_model=decision["selected_model"],
                    fallback_used=decision["fallback_used"],
                    latency_ms=latency_ms
                )
                response["history_id"] = entry["id"]
            except Exception as history_error:
                # History is auxiliary - never fail the chat request over it
                logger.warning(f"[{request_id}] Failed to save history: {history_error}")

        # Log (minimal - no raw query)
        logger.info(
            f"[{request_id}] class={predicted_class} "
            f"conf={confidence:.2f} model={decision['selected_model']} "
            f"fallback={decision['fallback_used']} "
            f"latency={latency_ms}ms"
        )

        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def _require_history() -> HistoryStore:
    """Return the history store or raise 503 when the feature is disabled."""
    if history_store is None:
        raise HTTPException(status_code=503, detail="Search history is disabled")
    return history_store


@app.get("/history")
async def list_history(
    limit: int = Query(50, ge=1, le=200, description="Max entries to return"),
    offset: int = Query(0, ge=0, description="Entries to skip"),
    q: str | None = Query(None, max_length=256, description="Filter by query text"),
    current_user: dict = Depends(get_current_user)
):
    """List the signed-in user's past searches, newest first."""
    store = _require_history()
    user_id = current_user["id"]
    search = q.strip() if q else None

    return {
        "entries": store.list(user_id, limit=limit, offset=offset, search=search),
        "total": store.count(user_id, search=search),
        "limit": limit,
        "offset": offset
    }


@app.get("/history/stats")
async def history_stats(current_user: dict = Depends(get_current_user)):
    """Aggregate statistics over the signed-in user's search history."""
    return _require_history().stats(current_user["id"])


@app.get("/history/{entry_id}")
async def get_history_entry(
    entry_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Fetch one of the user's past searches, including its full response."""
    entry = _require_history().get(current_user["id"], entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="History entry not found")
    return entry


@app.delete("/history/{entry_id}")
async def delete_history_entry(
    entry_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Delete one of the user's past searches."""
    if not _require_history().delete(current_user["id"], entry_id):
        raise HTTPException(status_code=404, detail="History entry not found")
    return {"deleted": 1, "id": entry_id}


@app.delete("/history")
async def clear_history(current_user: dict = Depends(get_current_user)):
    """Delete the signed-in user's entire search history."""
    return {"deleted": _require_history().clear(current_user["id"])}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
