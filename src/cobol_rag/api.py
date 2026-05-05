from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cobol_rag.chat import ChatSession
from cobol_rag.config import load_config
from cobol_rag.index import collection_count, open_index
from cobol_rag.loaders import LoaderError, load_path
from cobol_rag.query import QueryError
from cobol_rag.remove import apply_remove_plan, build_remove_plan
from cobol_rag.reset import apply_reset_plan, build_reset_plan
from cobol_rag.retrieve import retrieve as retrieve_documents
from cobol_rag.sync import apply_sync_plan, build_sync_plan

app = FastAPI(title="COBOL RAG API")

# Mount the static files for the UI
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
UI_DIR = PROJECT_ROOT / "ui"

# Global chat session cache (simplistic approach for single user)
_chat_session: Optional[ChatSession] = None

def get_chat_session() -> ChatSession:
    global _chat_session
    if _chat_session is None:
        settings = load_config(CONFIG_PATH)
        _chat_session = ChatSession(config=settings)
    return _chat_session

class ChatRequest(BaseModel):
    message: str

class SyncRequest(BaseModel):
    paths: Optional[List[str]] = None

class InspectRequest(BaseModel):
    target: str

class RetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = None

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
def chat(req: ChatRequest) -> Any:
    session = get_chat_session()
    try:
        answer = session.ask(req.message)
        sources = [
            {
                "source_id": s.metadata.get("source_id", ""),
                "source_path": s.metadata.get("source_path", ""),
                "source_format": s.metadata.get("source_format", ""),
                "score": float(s.score) if s.score is not None else None,
            }
            for s in answer.sources
        ]
        return {"answer": answer.answer, "sources": sources}
    except QueryError as error:
        raise HTTPException(status_code=400, detail=str(error))

@app.post("/api/chat/reset")
def chat_reset() -> Any:
    session = get_chat_session()
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
            "llm_polish_final_scripts": settings.answers.llm_polish_final_scripts,
        }
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
            "preview": r.text[:180] + "..." if len(r.text) > 180 else r.text
        })
        
    return {
        "query": req.query,
        "results_count": len(results),
        "items": docs
    }

app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

def run():
    uvicorn.run("cobol_rag.api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    run()
