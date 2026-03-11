"""
app/server.py — FastAPI backend for the RAG Chatbot web UI.

Endpoints:
  GET  /                        → serves index.html
  GET  /api/languages           → list available languages
  POST /api/activate            → set active language (ingests if needed)
  POST /api/ask                 → streaming SSE response
  GET  /api/status              → current pipeline status
"""
from __future__ import annotations
import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import os
import signal
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Request / Response models ──────────────────────────────────────────────

class ActivateRequest(BaseModel):
    language: str

class AskRequest(BaseModel):
    question: str

# ── App factory ────────────────────────────────────────────────────────────

def create_app(vs_manager, pipeline) -> FastAPI:
    last_ping = {"time": datetime.now()}
    TIMEOUT_SECONDS = 10

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        async def _watchdog():
            await asyncio.sleep(5)
            while True:
                await asyncio.sleep(3)
                if datetime.now() - last_ping["time"] > timedelta(seconds=TIMEOUT_SECONDS):
                    os.kill(os.getpid(), signal.SIGTERM)
        asyncio.create_task(_watchdog())
        yield
        # Shutdown (anything after yield runs on shutdown)

    app = FastAPI(title="RAG Coding Assistant", lifespan=lifespan)

    @app.post("/api/ping")
    async def ping():
        last_ping["time"] = datetime.now()
        return {"ok": True}

    WEB_DIR = Path(__file__).parent / "web"

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/languages")
    async def get_languages():
        from app.document_loader import get_available_languages
        return {"languages": get_available_languages()}

    @app.get("/api/status")
    async def get_status():
        return {
            "language": pipeline.current_language,
            "ready": pipeline.current_language is not None,
        }

    @app.post("/api/activate")
    async def activate_language(req: ActivateRequest):
        language = req.language
        if not language:
            raise HTTPException(400, "language required")

        if vs_manager.collection_exists(language):
            pipeline.set_language(language)
            return {"status": "ready", "language": language}

        # Need to ingest — run in thread pool so we don't block event loop
        loop = asyncio.get_event_loop()
        def _ingest():
            from app.document_loader import load_documents_for_language
            docs = load_documents_for_language(language)
            vs_manager.ingest(language, docs)
            pipeline.set_language(language)
            return len(docs)

        try:
            n = await loop.run_in_executor(None, _ingest)
            return {"status": "ingested", "language": language, "chunks": n}
        except Exception as exc:
            raise HTTPException(500, str(exc))

    @app.post("/api/ask")
    async def ask(req: AskRequest):
        if not req.question.strip():
            raise HTTPException(400, "question required")
        if pipeline.current_language is None:
            raise HTTPException(400, "No language selected")

        async def token_stream() -> AsyncGenerator[str, None]:
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def _run():
                try:
                    for token in pipeline.ask_stream(req.question):
                        loop.call_soon_threadsafe(queue.put_nowait, token)
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, f"\n\nError: {exc}")
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

            threading.Thread(target=_run, daemon=True).start()

            while True:
                token = await queue.get()
                if token is None:
                    break
                # SSE format
                data = json.dumps({"token": token})
                yield f"data: {data}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            token_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def run_server(vs_manager, pipeline, host="127.0.0.1", port=5000):
    app = create_app(vs_manager, pipeline)
    uvicorn.run(app, host=host, port=port, log_level="warning")