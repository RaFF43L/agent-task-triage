import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from actions import (
    TriageRequest,
    TriageResponse,
    process_triage,
    process_triage_stream,
    IngestTextRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    ingest_text_document,
    ingest_uploaded_file,
    search_documents,
)
from actions.evaluation import (
    EvaluationRequest,
    EvaluationResult,
    DetailedEvaluation,
    evaluate_response,
    evaluate_response_detailed,
)
from config.envs import envs
from config.logging import setup_logging, get_logger

setup_logging(level=envs.log_level)
logger = get_logger(__name__)


def _run_migrations() -> None:
    """Applies pending Alembic migrations before starting the application."""
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ini_path = os.path.join(project_root, "alembic.ini")
    if not os.path.exists(ini_path):
        return

    from alembic.config import Config
    from alembic import command

    try:
        alembic_cfg = Config(ini_path)
        # The API may run with cwd in `src/`; we fix the absolute path of the
        # scripts directory to not depend on the working directory.
        alembic_cfg.set_main_option(
            "script_location", os.path.join(project_root, "migrations")
        )
        command.upgrade(alembic_cfg, "head")
        logger.info("Migrations applied (alembic upgrade head).")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to apply migrations: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {envs.app_name} (env={envs.app_env})")
    logger.info(f"Bedrock region={envs.aws_region} model={envs.model}")
    _run_migrations()
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="4dev-lang-graph — Triage Agent",
    description="Triage agent with LangGraph + Amazon Bedrock",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    """Simple healthcheck."""
    return {"status": "ok", "app": envs.app_name, "env": envs.app_env}


@app.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest) -> TriageResponse:
    """HTTP transport: delegates to the `process_triage` action (blocking)."""
    try:
        return await process_triage(request.message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error executing triage")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/triage/stream")
async def triage_stream(request: TriageRequest) -> StreamingResponse:
    """HTTP transport via SSE: emits real-time events during triage.

    The body is the same (`{"message": "..."}`); the response is `text/event-stream`
    with events: `triage`, `attempt`, `status`, `quality`, `token`, `done`, `error`.
    """
    return StreamingResponse(
        process_triage_stream(request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # avoids buffering in proxies (e.g.: nginx)
        },
    )


# --------------------------------------------------------------------------- #
# RAG — ingestion and search (Bedrock embeddings + pgvector)
# --------------------------------------------------------------------------- #
@app.post("/rag/ingest", response_model=IngestResponse)
async def rag_ingest_text(request: IngestTextRequest) -> IngestResponse:
    """Indexes plain text in the vector store."""
    try:
        return await ingest_text_document(
            request.text,
            source=request.source,
            metadata=request.metadata,
            collection=request.collection,
            schema=request.schema_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error ingesting text")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/rag/ingest/file", response_model=IngestResponse)
async def rag_ingest_file(
    file: UploadFile = File(...),
    metadata: str | None = Form(default=None),
    collection: str | None = Form(default=None),
    schema: str | None = Form(default=None),
) -> IngestResponse:
    """Indexes an uploaded file (txt/md/pdf).

    Optional `metadata`: JSON string with extra metadata per document.
    Optional `collection`/`schema`: isolated destination (see get_vectorstore).
    """
    try:
        extra = json.loads(metadata) if metadata else {}
        data = await file.read()
        return await ingest_uploaded_file(
            file.filename or "upload",
            data,
            metadata=extra,
            collection=collection,
            schema=schema,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error ingesting file")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/rag/search", response_model=SearchResponse)
async def rag_search(request: SearchRequest) -> SearchResponse:
    """Searches by similarity for the most relevant chunks for the query."""
    try:
        return await search_documents(
            request.query,
            k=request.k,
            metadata_filter=request.filter,
            collection=request.collection,
            schema=request.schema_name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error in RAG search")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Evaluation — LLM-as-judge endpoints for testing
# --------------------------------------------------------------------------- #
@app.post("/evaluate", response_model=EvaluationResult)
async def evaluate(request: EvaluationRequest) -> EvaluationResult:
    """Evaluate an agent response using LLM-as-judge.
    
    Supports two modes:
    - simple: Fast evaluation with basic pass/fail and feedback
    - detailed: Comprehensive multi-dimensional evaluation with scores
    
    Use this endpoint to test and validate agent responses before deployment
    or for quality monitoring.
    """
    try:
        return evaluate_response(
            message=request.message,
            work_result=request.work_result,
            category=request.category,
            priority=request.priority,
            attempts=request.attempts,
            mode=request.mode,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error evaluating response")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/evaluate/detailed", response_model=DetailedEvaluation)
async def evaluate_detailed(request: EvaluationRequest) -> DetailedEvaluation:
    """Perform detailed multi-dimensional evaluation of an agent response.
    
    Returns comprehensive scoring across multiple criteria:
    - Relevance: Does it address the request?
    - Accuracy: Is the information correct?
    - Completeness: Are all necessary details included?
    - Clarity: Is it well-structured and easy to understand?
    - Tone: Is it professional and empathetic?
    - Conciseness: Is the length appropriate?
    
    Each criterion includes a score (1-5) and justification, plus an overall
    assessment with approval status and improvement suggestions.
    """
    try:
        return evaluate_response_detailed(
            message=request.message,
            work_result=request.work_result,
            category=request.category,
            priority=request.priority,
            attempts=request.attempts,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error evaluating response (detailed)")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
