"""FastAPI server for obsidian-chat."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import config
from .llm import LLMClient, LLMError
from .logging import get_logger
from .rag import ObsidianRAG
from .research import scholar_search, build_research_context, is_research_available, ResearchError
from .utils import __version__, SYSTEM_PROMPT, SUMMARIZE_PROMPT, build_context_prompt

log = get_logger(__name__)


# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    top_k: int = Field(default=10, ge=1, le=50)
    use_rag: bool = True
    use_research: bool = False  # Search academic papers via Semantic Scholar
    research_limit: int = Field(default=3, ge=1, le=10)
    stream: bool = True
    # Conversation history support
    history: list[ChatMessage] = Field(default_factory=list)
    summary: str | None = None  # Compressed summary of older conversation


class ChatResponse(BaseModel):
    response: str
    sources: list[dict]


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=50)


class QueryResult(BaseModel):
    content: str
    source: str
    title: str
    score: float


class QueryResponse(BaseModel):
    results: list[QueryResult]


class IndexRequest(BaseModel):
    force: bool = False


class IndexResponse(BaseModel):
    files_processed: int
    chunks_added: int
    errors: list[str]


class StatsResponse(BaseModel):
    total_chunks: int
    collection_name: str
    vault_path: str


class HealthResponse(BaseModel):
    status: str
    llm_url: str
    vault_path: str
    indexed_chunks: int
    research_available: bool


# Global instances
rag: ObsidianRAG | None = None
llm: LLMClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global rag, llm
    log.info("Starting obsidian-chat server v{}", __version__)
    log.info("Initializing RAG with vault: {}", config.vault_path)
    rag = ObsidianRAG()
    log.info("RAG initialized with {} chunks", rag.collection.count())
    log.info("Initializing LLM client: {}", config.llm_base_url)
    llm = LLMClient()
    log.info("Server ready")
    yield
    log.info("Shutting down server")
    if llm:
        llm.close()


app = FastAPI(
    title="Obsidian Chat API",
    description="Chat with your Obsidian vault using RAG and a local LLM",
    version=__version__,
    lifespan=lifespan,
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve static files
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def serve_ui():
    """Serve the web UI."""
    return FileResponse(STATIC_DIR / "index.html")


class VersionResponse(BaseModel):
    version: str


@app.get("/version", response_model=VersionResponse)
async def get_version():
    """Get the application version."""
    return VersionResponse(version=__version__)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and configuration."""
    return HealthResponse(
        status="ok",
        llm_url=config.llm_base_url,
        vault_path=config.vault_path,
        indexed_chunks=rag.collection.count() if rag else 0,
        research_available=is_research_available(),
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get index statistics."""
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not initialized")
    stats = rag.get_stats()
    return StatsResponse(**stats)


@app.post("/query", response_model=QueryResponse)
async def query_notes(request: QueryRequest):
    """Query indexed notes without LLM (RAG only)."""
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    results = rag.query(request.query, top_k=request.top_k)
    return QueryResponse(
        results=[QueryResult(**r) for r in results]
    )


class ResearchRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=20)


class ResearchPaper(BaseModel):
    title: str | None
    year: int | None
    abstract: str | None
    url: str | None


class ResearchResponse(BaseModel):
    papers: list[ResearchPaper]


@app.post("/research", response_model=ResearchResponse)
async def search_papers(request: ResearchRequest):
    """Search academic papers via Semantic Scholar."""
    if not is_research_available():
        raise HTTPException(
            status_code=503,
            detail="Research not available. Set SEMANTIC_SCHOLAR_API_KEY in .env"
        )

    try:
        papers = scholar_search(request.query, limit=request.limit)
        return ResearchResponse(papers=[ResearchPaper(**p) for p in papers])
    except ResearchError as e:
        log.error("Research error: {}", e.message)
        raise HTTPException(status_code=e.status_code or 502, detail=str(e))


@app.post("/index", response_model=IndexResponse)
async def index_vault(request: IndexRequest):
    """Index or reindex the Obsidian vault."""
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    # Run indexing in thread pool to not block
    loop = asyncio.get_event_loop()
    stats = await loop.run_in_executor(
        None, lambda: rag.index_vault(force_reindex=request.force)
    )
    return IndexResponse(**stats)


def build_messages_with_history(request: ChatRequest, context_prompt: str) -> list[dict]:
    """Build message list including history, summary, and current message."""
    messages = []

    # Add summary of older conversation if provided
    if request.summary:
        messages.append({
            "role": "user",
            "content": f"[Previous conversation summary: {request.summary}]"
        })
        messages.append({
            "role": "assistant",
            "content": "I understand. I'll keep that context in mind."
        })

    # Add recent conversation history
    for msg in request.history:
        messages.append({"role": msg.role, "content": msg.content})

    # Build current message with RAG context
    if context_prompt:
        full_message = f"{context_prompt}\n\nUser question: {request.message}"
    else:
        full_message = request.message

    messages.append({"role": "user", "content": full_message})

    return messages


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the LLM using RAG context (non-streaming)."""
    if not rag or not llm:
        raise HTTPException(status_code=503, detail="Services not initialized")

    log.debug("Chat request: {} (use_rag={}, use_research={}, top_k={})",
              request.message[:50], request.use_rag, request.use_research, request.top_k)

    # Build combined context from RAG and research
    sources = []
    context_parts = []

    # Get RAG context from Obsidian notes
    if request.use_rag:
        contexts = rag.query(request.message, top_k=request.top_k)
        sources = contexts
        rag_context = build_context_prompt(contexts)
        if rag_context:
            context_parts.append(rag_context)
        log.debug("RAG returned {} contexts", len(contexts))

    # Get research papers if enabled
    if request.use_research and is_research_available():
        try:
            papers = scholar_search(request.message, limit=request.research_limit)
            research_context = build_research_context(papers)
            if research_context:
                context_parts.append(research_context)
            log.debug("Research returned {} papers", len(papers))
        except ResearchError as e:
            log.warning("Research search failed: {}", e.message)

    context_prompt = "\n\n".join(context_parts)

    # Build messages with history
    messages = build_messages_with_history(request, context_prompt)

    # Get response (non-streaming)
    try:
        response = llm.chat(messages, system_prompt=SYSTEM_PROMPT, stream=False)
        log.debug("LLM response received ({} chars)", len(response))
    except LLMError as e:
        log.error("LLM error: {}", e.message)
        raise HTTPException(status_code=e.status_code or 502, detail=str(e))

    return ChatResponse(response=response, sources=sources)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Chat with the LLM using RAG context (streaming)."""
    if not rag or not llm:
        raise HTTPException(status_code=503, detail="Services not initialized")

    log.debug("Chat stream request: {} (use_rag={}, use_research={}, top_k={})",
              request.message[:50], request.use_rag, request.use_research, request.top_k)

    # Build combined context from RAG and research
    context_parts = []

    # Get RAG context from Obsidian notes
    if request.use_rag:
        contexts = rag.query(request.message, top_k=request.top_k)
        rag_context = build_context_prompt(contexts)
        if rag_context:
            context_parts.append(rag_context)
        log.debug("RAG returned {} contexts", len(contexts))

    # Get research papers if enabled
    if request.use_research and is_research_available():
        try:
            papers = scholar_search(request.message, limit=request.research_limit)
            research_context = build_research_context(papers)
            if research_context:
                context_parts.append(research_context)
            log.debug("Research returned {} papers", len(papers))
        except ResearchError as e:
            log.warning("Research search failed: {}", e.message)

    context_prompt = "\n\n".join(context_parts)

    # Build messages with history
    messages = build_messages_with_history(request, context_prompt)

    async def generate() -> AsyncGenerator[str, None]:
        try:
            for chunk in llm.chat(messages, system_prompt=SYSTEM_PROMPT, stream=True):
                yield chunk
        except LLMError as e:
            log.error("LLM streaming error: {}", e.message)
            yield f"\n\n[Error: {e.message}]"

    return StreamingResponse(generate(), media_type="text/plain")


class SummarizeRequest(BaseModel):
    messages: list[ChatMessage]


class SummarizeResponse(BaseModel):
    summary: str


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_conversation(request: SummarizeRequest):
    """Summarize a conversation for context compression."""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not initialized")

    # Format conversation for summarization
    conversation_text = "\n".join(
        f"{msg.role.upper()}: {msg.content}" for msg in request.messages
    )

    messages = [{
        "role": "user",
        "content": f"{SUMMARIZE_PROMPT}\n\nConversation:\n{conversation_text}"
    }]

    try:
        summary = llm.chat(messages, stream=False)
    except LLMError as e:
        raise HTTPException(status_code=e.status_code or 502, detail=str(e))

    return SummarizeResponse(summary=summary)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
