from __future__ import annotations

import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cobol_rag.chat import ChatSession
from cobol_rag.config import load_config
from cobol_rag.index import collection_count, open_index
from cobol_rag.loaders import LoaderError, load_path
from cobol_rag.observability import (
    list_feedback,
    list_traces,
    load_trace,
    write_feedback,
)
from cobol_rag.query import QueryError
from cobol_rag.remove import apply_remove_plan, build_remove_plan
from cobol_rag.reset import apply_reset_plan, build_reset_plan
from cobol_rag.retrieve import clear_retrieval_cache, retrieve as retrieve_documents
from cobol_rag.sync import apply_sync_plan, build_sync_plan

app = FastAPI(title="COBOL RAG API")


@app.middleware("http")
async def prevent_stale_ui_assets(request: Request, call_next: Any) -> Any:
    """Keep the local UI shell and bundles fresh during active development."""
    response = await call_next(request)
    if request.url.path in {"/", "/index.html", "/app.js", "/style.css"}:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Mount the static files for the UI
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
UI_DIR = PROJECT_ROOT / "ui"

# Conversation memory must never leak between browsers or users.  The browser
# supplies an opaque session id; command-line and older clients retain a
# backwards-compatible "default" session.
_MAX_CHAT_SESSIONS = 256
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_chat_sessions: "OrderedDict[str, tuple[float, ChatSession]]" = OrderedDict()
_chat_sessions_lock = threading.RLock()


def _normalize_session_id(session_id: str | None) -> str:
    candidate = (session_id or "default").strip()
    return candidate if _SESSION_ID_PATTERN.fullmatch(candidate) else "default"


def _request_session_id(request: Request, body_session_id: str | None = None) -> str:
    return _normalize_session_id(body_session_id or request.headers.get("X-Session-ID"))


def get_chat_session(session_id: str | None = None) -> ChatSession:
    key = _normalize_session_id(session_id)
    with _chat_sessions_lock:
        cached = _chat_sessions.pop(key, None)
        if cached is not None:
            session = cached[1]
            _chat_sessions[key] = (time.monotonic(), session)
            return session
        settings = load_config(CONFIG_PATH)
        session = ChatSession(config=settings)
        _chat_sessions[key] = (time.monotonic(), session)
        while len(_chat_sessions) > _MAX_CHAT_SESSIONS:
            _chat_sessions.popitem(last=False)
        return session

class ChatRequest(BaseModel):
    message: str
    program: Optional[str] = None
    session_id: Optional[str] = None

class SyncRequest(BaseModel):
    paths: Optional[List[str]] = None

class InspectRequest(BaseModel):
    target: str

class RetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = None

class FeedbackRequest(BaseModel):
    trace_id: str
    rating: str
    labels: List[str] = []
    note: str = ""

@app.get("/api/health")
def health() -> Any:
    settings = load_config(CONFIG_PATH)
    return {
        "status": "ok",
        "collection": settings.index.collection,
        "inbox_dir": str(settings.paths.inbox_dir),
        "llm": settings.llm.model,
        "embedding": settings.embedding.model,
    }

@app.post("/api/chat")
def chat(req: ChatRequest, request: Request) -> Any:
    session = get_chat_session(_request_session_id(request, req.session_id))
    try:
        answer = session.ask(req.message, target_program=req.program)
        sources = [
            {
                "source_id": s.metadata.get("source_id", ""),
                "source_path": s.metadata.get("source_path", ""),
                "source_format": s.metadata.get("source_format", ""),
                "source_file": s.metadata.get("source_file", ""),
                "evidence_path": s.metadata.get("evidence_path", ""),
                "chunk_type": s.metadata.get("chunk_type", ""),
                "program": s.metadata.get("program", ""),
                "variable": s.metadata.get("variable", ""),
                "paragraph": s.metadata.get("paragraph", ""),
                "entity_key": s.metadata.get("entity_key", ""),
                "intent_domain": s.metadata.get("intent_domain", ""),
                "context_role": s.metadata.get("context_role", ""),
                "score": float(s.score) if s.score is not None else None,
            }
            for s in answer.sources
        ]
        return {
            "answer": answer.answer,
            "sources": sources,
            "route": answer.route,
            "scope": answer.scope.as_dict(),
            "session_state": session.state.as_dict(),
            "trace_id": answer.trace_id,
            "guard_status": answer.guard_status,
            "plan": answer.plan.as_dict() if answer.plan else {},
            "execution_mode": answer.execution_mode,
            "debug": answer.debug,
        }
    except QueryError as error:
        raise HTTPException(status_code=400, detail=str(error))

@app.post("/api/chat/cancel")
def chat_cancel(request: Request) -> Any:
    """Stop waiting on the current question and keep it out of chat memory.

    The worker thread cannot be interrupted mid-generation, so the request runs
    to completion; what this changes is that its result is discarded instead of
    becoming the context the next question is resolved against.
    """
    session = get_chat_session(_request_session_id(request))
    generation = session.cancel()
    return {
        "status": "ok",
        "generation": generation,
        "message": "Stopped. The answer in progress will be discarded.",
    }


@app.post("/api/chat/reset")
def chat_reset(request: Request) -> Any:
    session = get_chat_session(_request_session_id(request))
    session.cancel()
    session.reset()
    return {"status": "ok", "message": "Chat memory cleared."}

@app.get("/api/config")
def get_config_endpoint() -> Any:
    settings = load_config(CONFIG_PATH)
    return {
        "paths": {
            "chroma_dir": str(settings.paths.chroma_dir),
            "inbox_dir": str(settings.paths.inbox_dir),
            "manifest_dir": str(settings.paths.manifest_dir),
            "trace_dir": str(settings.paths.trace_dir),
            "feedback_dir": str(settings.paths.feedback_dir),
        },
        "llm": {
            "provider": settings.llm.provider,
            "model": settings.llm.model,
            "context_window": settings.llm.context_window,
        },
        "embedding": {
            "provider": settings.embedding.provider,
            "model": settings.embedding.model,
        },
        "index": {
            "collection": settings.index.collection,
        },
        "retrieval": {
            "top_k": settings.retrieval.top_k,
        },
        "answers": {
            "require_citations": settings.answers.require_citations,
        },
        "observability": {"enabled": settings.observability.enabled},
    }

@app.get("/api/index-info")
def get_index_info() -> Any:
    settings = load_config(CONFIG_PATH)
    resources = open_index(settings)
    return {
        "chroma_dir": str(settings.paths.chroma_dir),
        "collection": settings.index.collection,
        "documents": collection_count(resources),
        "llm": settings.llm.model,
        "embedding": settings.embedding.model,
    }

@app.get("/api/inbox")
def get_inbox() -> Any:
    settings = load_config(CONFIG_PATH)
    inbox_dir = settings.paths.inbox_dir
    
    def _build_tree(dir_path: Path) -> dict:
        tree = {"name": dir_path.name, "path": str(dir_path), "is_dir": True, "children": []}
        if dir_path.exists() and dir_path.is_dir():
            for item in dir_path.iterdir():
                if item.is_dir():
                    tree["children"].append(_build_tree(item))
                else:
                    tree["children"].append({
                        "name": item.name,
                        "path": str(item),
                        "is_dir": False
                    })
        # sort directories first, then files
        tree["children"].sort(key=lambda x: (not x["is_dir"], x["name"]))
        return tree

    return _build_tree(inbox_dir)

@app.post("/api/sync")
def sync_inbox(req: SyncRequest) -> Any:
    settings = load_config(CONFIG_PATH)
    plan = build_sync_plan(settings, dry_run=False)
    
    # Filter items if specific paths are provided
    if req.paths is not None:
        selected_paths = {Path(p).resolve() for p in req.paths}
        filtered_items = []
        for item in plan.items:
            item_path = Path(item.source_path).resolve()
            # Include if item path matches a selected path or is under a selected path (directory)
            if any(item_path == sp or sp in item_path.parents for sp in selected_paths):
                filtered_items.append(item)
        plan = replace(plan, items=filtered_items)

    apply_sync_plan(settings, plan)
    clear_retrieval_cache()
    
    return {
        "collection": plan.collection,
        "documents_processed": len(plan.items),
        "added": plan.count("add"),
        "updated": plan.count("update"),
        "skipped": plan.count("skip"),
    }

@app.post("/api/reset")
def reset_collection() -> Any:
    settings = load_config(CONFIG_PATH)
    plan = build_reset_plan(settings, dry_run=False)
    apply_reset_plan(settings, plan)
    clear_retrieval_cache()
    return {"status": "ok", "message": f"Collection {plan.collection} reset."}

@app.post("/api/inspect")
def inspect_target(req: InspectRequest) -> Any:
    settings = load_config(CONFIG_PATH)
    target = Path(req.target)
    try:
        loaded = load_path(target, config=settings)
    except LoaderError as error:
        raise HTTPException(status_code=400, detail=str(error))
    
    docs = []
    for item in loaded:
        document = item.document
        docs.append({
            "loader": item.loader_name,
            "source_id": str(document.metadata.get("source_id", "")),
            "source_path": str(document.metadata.get("source_path", item.source_path)),
            "chars": len(document.text),
            "preview": document.text[:120] + "..." if len(document.text) > 120 else document.text
        })
        
    return {
        "target": req.target,
        "documents": len(loaded),
        "items": docs
    }

@app.post("/api/retrieve")
def retrieve_raw(req: RetrieveRequest) -> Any:
    settings = load_config(CONFIG_PATH)
    results = retrieve_documents(query=req.query, config=settings, top_k=req.top_k)
    
    docs = []
    for r in results:
        docs.append({
            "score": float(r.score) if r.score is not None else None,
            "source_format": str(r.metadata.get("source_format", "")),
            "source_id": str(r.metadata.get("source_id", "")),
            "source_path": str(r.metadata.get("source_path", "")),
            "source_file": str(r.metadata.get("source_file", "")),
            "evidence_path": str(r.metadata.get("evidence_path", "")),
            "chunk_type": str(r.metadata.get("chunk_type", "")),
            "program": str(r.metadata.get("program", "")),
            "variable": str(r.metadata.get("variable", "")),
            "paragraph": str(r.metadata.get("paragraph", "")),
            "preview": r.text[:180] + "..." if len(r.text) > 180 else r.text
        })
        
    return {
        "query": req.query,
        "results_count": len(results),
        "items": docs
    }

@app.get("/api/traces")
def traces(limit: int = 20) -> Any:
    settings = load_config(CONFIG_PATH)
    return {"items": list_traces(settings, limit=limit)}

@app.get("/api/traces/{trace_id}")
def trace_detail(trace_id: str) -> Any:
    settings = load_config(CONFIG_PATH)
    try:
        trace = load_trace(settings, trace_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace

@app.post("/api/feedback")
def feedback(req: FeedbackRequest) -> Any:
    settings = load_config(CONFIG_PATH)
    try:
        return write_feedback(
            settings,
            trace_id=req.trace_id,
            rating=req.rating,
            labels=req.labels,
            note=req.note,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

@app.get("/api/feedback")
def feedback_list(limit: int = 50) -> Any:
    settings = load_config(CONFIG_PATH)
    return {"items": list_feedback(settings, limit=limit)}

app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

def run():
    uvicorn.run("cobol_rag.api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    run()
