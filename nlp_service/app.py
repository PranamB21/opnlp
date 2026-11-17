"""FastAPI NLP microservice for the WhisperLiveKit meeting assistant.

Endpoints
---------
GET  /            extended dashboard (topics / key points / action items panels)
WS   /asr         audio relay + transcript subscription (see bridge.py)
POST /api/end     final extraction, summary and Slack delivery
GET  /api/state   current meeting state snapshot
POST /api/reset   clear the meeting state
GET  /api/export  markdown export of the notes so far
POST /api/ingest  push transcript lines directly (headless / integration testing)
GET  /health      liveness and pipeline info
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from .bridge import handle_dashboard_socket
from .config import Settings
from .extractor import Extractor, ExtractionError
from .models import ActionItem, TranscriptSegment, parse_clock
from .slack import post_to_slack
from .state import SessionHub
from .window import RollingWindow

logging.basicConfig(
    level=os.environ.get("NLP_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_WEB_DIR = Path(__file__).resolve().parent.parent / "WhisperLiveKit" / "whisperlivekit" / "web"


def resolve_web_dir(settings: Settings) -> Path:
    if settings.web_dir:
        return Path(settings.web_dir).expanduser().resolve()
    return DEFAULT_WEB_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings
    app.state.hub = SessionHub()
    app.state.extractor = Extractor(settings)
    app.state.hub.provider = app.state.extractor.provider
    logger.info(
        "NLP service ready: provider=%s asr=%s window=%.0fs",
        app.state.extractor.provider,
        settings.asr_ws_url,
        settings.window_seconds,
    )
    if app.state.extractor.provider == "mock":
        logger.warning(
            "Running with the offline heuristic extractor (no ANTHROPIC_API_KEY / OPENAI_API_KEY). "
            "Set a key for real LLM extraction."
        )
    try:
        yield
    finally:
        await app.state.extractor.aclose()


app = FastAPI(title="WhisperLiveKit NLP service", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/asr")
async def asr_bridge(websocket: WebSocket):
    settings: Settings = websocket.app.state.settings
    await websocket.accept()
    try:
        await handle_dashboard_socket(
            websocket,
            settings,
            websocket.app.state.hub,
            websocket.app.state.extractor,
            raw_query=websocket.url.query,
        )
    except WebSocketDisconnect:
        logger.info("Dashboard disconnected")
    except Exception:
        logger.exception("Dashboard bridge failed")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


@app.get("/api/state")
async def get_state(request: Request):
    hub: SessionHub = request.app.state.hub
    return JSONResponse(hub.state.snapshot())


@app.get("/api/export", response_class=PlainTextResponse)
async def export_notes(request: Request):
    hub: SessionHub = request.app.state.hub
    return PlainTextResponse(hub.state.export_markdown(), media_type="text/markdown")


@app.post("/api/reset")
async def reset_state(request: Request):
    hub: SessionHub = request.app.state.hub
    hub.state.reset()
    hub.new_meeting_pending = False
    await hub.broadcast_state()
    return {"status": "reset"}


@app.post("/api/end")
async def end_meeting(request: Request):
    """Flush the last window, summarize the meeting, and push to Slack."""
    settings: Settings = request.app.state.settings
    hub: SessionHub = request.app.state.hub
    extractor: Extractor = request.app.state.extractor

    pipeline = getattr(hub, "active_pipeline", None)
    if pipeline is not None:
        await pipeline.flush_pending("end_meeting")

    state = hub.state
    if not state.segments:
        raise HTTPException(status_code=409, detail="No transcript yet; start a session first.")

    try:
        result = await extractor.summarize(state.transcript_lines(), state.actions_as_dicts())
    except ExtractionError as exc:
        logger.warning("Summary generation failed: %s", exc)
        result = {"summary": "", "decisions": [], "topics": [], "action_items": []}

    state.summary = result.get("summary") or ""
    state.decisions = result.get("decisions") or []
    for item in result.get("action_items") or []:
        state.merge_action_item(
            ActionItem(task=item["task"], owner=item.get("owner"), deadline=item.get("deadline"))
        )
    state.ended = True
    hub.new_meeting_pending = True

    topics = [t.topic_label for t in state.topics] or list(result.get("topics") or [])
    slack_result = await post_to_slack(settings, state.summary, state.actions_as_dicts(), topics)

    payload = {
        "type": "nlp",
        "event": "final",
        "summary": state.summary,
        "decisions": state.decisions,
        "topics": topics,
        "action_items": state.actions_as_dicts(),
        "markdown": state.export_markdown(),
        "slack": slack_result,
        "state": state.snapshot(),
    }
    await hub.broadcast(payload)
    logger.info("Meeting finalized: %d action items, slack=%s", len(state.action_items), slack_result["status"])
    return payload


@app.post("/api/ingest")
async def ingest_lines(request: Request):
    """Push committed transcript lines straight into the pipeline (no audio).

    Useful for integration tests and for feeding transcripts from a different
    ASR source without touching the bridge.
    """
    body = await request.json()
    hub: SessionHub = request.app.state.hub
    raw_lines = body.get("lines") if isinstance(body, dict) else body
    if not isinstance(raw_lines, list):
        raise HTTPException(status_code=400, detail="Expected a list of lines")

    pipeline = getattr(hub, "active_pipeline", None)
    created = False
    if pipeline is None:
        pipeline = _make_detached_pipeline(request)
        created = True

    segments = []
    for line in raw_lines:
        if not isinstance(line, dict):
            continue
        text = (line.get("text") or "").strip()
        start = parse_clock(line.get("start", line.get("start_time")))
        end = parse_clock(line.get("end", line.get("end_time")))
        if not text or start is None or end is None:
            continue
        speaker = line.get("speaker")
        speaker_label = f"Speaker {speaker}" if isinstance(speaker, int) and speaker > 0 else "Speaker"
        segments.append(TranscriptSegment(start=start, end=end, text=text, speaker=speaker_label))

    new_segments = pipeline.window.ingest(segments)
    hub.state.add_segments(new_segments)
    pipeline.check_boundary()
    return {"status": "ok", "added": len(new_segments), "detached_pipeline": created}


@app.get("/health")
async def health(request: Request):
    hub: SessionHub = request.app.state.hub
    settings: Settings = request.app.state.settings
    return {
        "status": "ok",
        "provider": hub.provider,
        "asr_ws_url": settings.asr_ws_url,
        "clients": hub.client_count,
        "windows": hub.state.windows_processed,
        "segments": len(hub.state.segments),
        "busy": hub.busy_windows > 0,
    }


def _make_detached_pipeline(request: Request):
    """Pipeline for transcripts that arrive without an audio bridge."""
    from .bridge import TranscriptPipeline

    hub: SessionHub = request.app.state.hub
    pipeline = TranscriptPipeline(request.app.state.settings, hub, request.app.state.extractor)
    pipeline.start()
    hub.active_pipeline = pipeline
    return pipeline


# ---------------------------------------------------------------------------
# Dashboard assets (extended WhisperLiveKit UI)
# ---------------------------------------------------------------------------

_WEB_ASSETS = {
    "live_transcription.html": "text/html",
    "live_transcription.css": "text/css",
    "live_transcription.js": "application/javascript",
    "pcm_worklet.js": "application/javascript",
    "recorder_worker.js": "application/javascript",
}

_MEDIA_TYPES = {
    ".svg": "image/svg+xml",
    ".css": "text/css",
    ".js": "application/javascript",
    ".html": "text/html",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


@app.get("/")
async def dashboard(request: Request):
    path = resolve_web_dir(request.app.state.settings) / "live_transcription.html"
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Dashboard assets not found at {path}")
    return FileResponse(path, media_type="text/html")


@app.get("/live_transcription.css")
@app.get("/live_transcription.js")
async def root_asset(request: Request):
    name = request.url.path.lstrip("/")
    path = resolve_web_dir(request.app.state.settings) / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type=_WEB_ASSETS[name])


@app.get("/web/{asset_path:path}")
async def web_asset(asset_path: str, request: Request):
    base = resolve_web_dir(request.app.state.settings)
    if asset_path in _WEB_ASSETS:
        path = base / asset_path
    else:
        path = (base / asset_path).resolve()
        if not str(path).startswith(str(base.resolve())):
            raise HTTPException(status_code=404, detail="Not found")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    media = _WEB_ASSETS.get(asset_path) or _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media)