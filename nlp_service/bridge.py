"""WebSocket bridge between the dashboard and WhisperLiveKit.

The dashboard talks to the NLP service instead of directly to the ASR server.
Audio frames are relayed verbatim; every transcription frame coming back is
inspected, committed lines are buffered into a rolling window, and finished
windows are handed to the LLM extractor. Results are pushed back on the same
socket as `{"type": "nlp", ...}` messages.

Because the bridge only speaks the public WhisperLiveKit WebSocket protocol,
it works unchanged whether the ASR server runs with or without `--diarization`
and with either PCM (AudioWorklet) or WebM (MediaRecorder) client audio.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import websockets
from fastapi import WebSocket
from starlette.websockets import WebSocketState

from .config import Settings
from .extractor import Extractor, ExtractionError
from .models import TranscriptSegment, parse_clock
from .state import SessionClient, SessionHub
from .window import RollingWindow, WindowBatch

logger = logging.getLogger(__name__)


def build_upstream_url(base: str, raw_query: str) -> str:
    """Forward per-session ASR options, forcing full-state mode."""
    parts = urlsplit(base)
    params = dict(parse_qsl(raw_query, keep_blank_values=False))
    params["mode"] = "full"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), ""))


def parse_transcript_lines(payload: dict[str, Any]) -> list[TranscriptSegment]:
    """Convert WhisperLiveKit's `lines` array into transcript segments."""
    segments: list[TranscriptSegment] = []
    for line in payload.get("lines") or []:
        if not isinstance(line, dict):
            continue
        text = (line.get("text") or "").strip()
        if not text:
            continue  # silence markers carry no text
        start = parse_clock(line.get("start"))
        end = parse_clock(line.get("end"))
        if start is None or end is None or end < start:
            continue
        speaker = line.get("speaker")
        if isinstance(speaker, int) and speaker > 0:
            speaker_label = f"Speaker {speaker}"
        else:
            speaker_label = "Speaker"
        segments.append(TranscriptSegment(start=start, end=end, text=text, speaker=speaker_label))
    return segments


class TranscriptPipeline:
    """Rolling-window extraction for one live ASR session."""

    def __init__(self, settings: Settings, hub: SessionHub, extractor: Extractor):
        self.settings = settings
        self.hub = hub
        self.extractor = extractor
        self.window = RollingWindow(
            window_seconds=settings.window_seconds,
            min_window_seconds=settings.min_window_seconds,
            max_window_seconds=settings.max_window_seconds,
            pause_gap_seconds=settings.pause_gap_seconds,
        )
        self.queue: asyncio.Queue[WindowBatch] = asyncio.Queue(maxsize=8)
        self._worker: Optional[asyncio.Task] = None
        self._ticker: Optional[asyncio.Task] = None
        self._closed = False
        self.completed = False
        self._started_ingest = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._worker = asyncio.create_task(self._run_worker(), name="nlp-extractor")
        self._ticker = asyncio.create_task(self._run_ticker(), name="nlp-window-ticker")

    async def close(self, flush: bool = True) -> None:
        self._closed = True
        if flush:
            batch = self.window.take_remaining("session_end")
            if batch is not None:
                self._enqueue(batch)
            # Give in-flight extraction a bounded chance to finish.
            try:
                await asyncio.wait_for(self.queue.join(), timeout=90)
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for final NLP extraction")
        for task in (self._ticker, self._worker):
            if task is not None and not task.done():
                task.cancel()
        for task in (self._ticker, self._worker):
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)

    # -- ingestion ---------------------------------------------------------

    def ingest_frame(self, payload: dict[str, Any]) -> None:
        if payload.get("type") == "ready_to_stop":
            self.completed = True
            return
        segments = parse_transcript_lines(payload)
        if segments:
            self._maybe_start_new_meeting(segments)
            new_segments = self.window.ingest(segments)
            if new_segments:
                self.hub.state.add_segments(new_segments)
        self.check_boundary()

    def _maybe_start_new_meeting(self, segments: list[TranscriptSegment]) -> None:
        """Clear ended-meeting notes when a fresh recording begins.

        A new ASR session restarts its clock near zero. If the previous meeting
        was ended deliberately, the first lines of the new session start a clean
        meeting; a reconnect in the middle of a meeting keeps existing notes.
        """
        if self._started_ingest:
            return
        self._started_ingest = True
        if segments[0].start > 2.0 or not self.hub.new_meeting_pending:
            return
        self.hub.state.reset()
        self.hub.new_meeting_pending = False
        logger.info("Started a new meeting; previous notes cleared")
        asyncio.create_task(self.hub.broadcast_status("New meeting started."))

    def check_boundary(self) -> None:
        """Cut and enqueue the pending window when a boundary rule fires."""
        batch = self.window.take_if_ready()
        if batch is not None:
            self._enqueue(batch)

    async def flush_pending(self, reason: str = "flush") -> None:
        """Force the pending window out and wait for queued extraction to finish."""
        batch = self.window.take_remaining(reason)
        if batch is not None:
            self._enqueue(batch)
        try:
            await asyncio.wait_for(self.queue.join(), timeout=90)
        except asyncio.TimeoutError:
            logger.warning("Timed out flushing NLP windows before %s", reason)

    def _enqueue(self, batch: WindowBatch) -> None:
        if self.queue.full():
            try:
                dropped = self.queue.get_nowait()
                self.queue.task_done()
                logger.warning(
                    "Dropping unprocessed NLP window %s (LLM slower than speech)", dropped.index
                )
            except asyncio.QueueEmpty:
                pass
        self.queue.put_nowait(batch)

    # -- background loops --------------------------------------------------

    async def _run_ticker(self) -> None:
        """Cut windows on a timer too, so trailing speech still gets processed."""
        while not self._closed:
            await asyncio.sleep(2.0)
            try:
                self.check_boundary()
            except Exception:
                logger.exception("Ticker failed to evaluate window boundary")

    async def _run_worker(self) -> None:
        while True:
            batch = await self.queue.get()
            try:
                await self._process(batch)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("NLP extraction failed for window %s", batch.index)
                await self.hub.broadcast_status("NLP extraction failed; continuing.", error=True)
            finally:
                self.queue.task_done()

    async def _process(self, batch: WindowBatch) -> None:
        self.hub.busy_windows += 1
        try:
            await self.hub.broadcast_status(
                f"Analyzing window {batch.index + 1} ({batch.duration():.0f}s, {batch.reason})"
            )
            extraction = await self.extractor.extract(
                batch.segments,
                batch.start,
                batch.end,
                prior_topics=self.hub.state.topics[-self.settings.history_topics:],
            )
            self.hub.state.merge(extraction)
        finally:
            self.hub.busy_windows = max(0, self.hub.busy_windows - 1)
        await self.hub.broadcast_state()
        await self.hub.broadcast_status(
            f"Window {batch.index + 1} done: "
            f"{len(extraction.topic_segments)} topic(s), "
            f"{sum(len(g.points) for g in extraction.key_points)} point(s), "
            f"{len(extraction.action_items)} action(s)"
        )


async def _send_json_safe(websocket: WebSocket, payload: dict) -> None:
    if websocket.application_state != WebSocketState.CONNECTED:
        raise ConnectionError("dashboard socket closed")
    await websocket.send_json(payload)


async def handle_dashboard_socket(
    browser: WebSocket,
    settings: Settings,
    hub: SessionHub,
    extractor: Extractor,
    raw_query: str = "",
) -> None:
    """Relay one dashboard session to WhisperLiveKit and stream NLP updates back."""
    upstream_url = build_upstream_url(settings.asr_ws_url, raw_query)
    client = SessionClient(send=lambda payload: _send_json_safe(browser, payload))
    hub.add_client(client)
    pipeline: Optional[TranscriptPipeline] = None
    try:
        async with websockets.connect(upstream_url, max_size=None, open_timeout=15) as upstream:
            await hub.broadcast_status("NLP service connected to ASR.")
            await client.send({"type": "nlp", "event": "state", "state": hub.state.snapshot()})
            pipeline = TranscriptPipeline(settings, hub, extractor)
            pipeline.start()
            hub.active_pipeline = pipeline

            async def browser_to_asr() -> None:
                while True:
                    message = await browser.receive()
                    if message.get("bytes") is not None:
                        await upstream.send(message["bytes"])
                    elif message.get("text") is not None:
                        # The native ASR endpoint only accepts audio frames; ignore
                        # control text instead of breaking the upstream session.
                        continue
                    else:
                        await upstream.send(b"")

            async def asr_to_browser() -> None:
                async for raw in upstream:
                    if isinstance(raw, bytes):
                        if browser.application_state == WebSocketState.CONNECTED:
                            await browser.send_bytes(raw)
                        continue
                    if browser.application_state == WebSocketState.CONNECTED:
                        await browser.send_text(raw)
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        pipeline.ingest_frame(payload)

            tasks = [
                asyncio.create_task(browser_to_asr(), name="browser-to-asr"),
                asyncio.create_task(asr_to_browser(), name="asr-to-browser"),
            ]
            try:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    exc = task.exception()
                    if exc and not isinstance(exc, (ConnectionError, websockets.ConnectionClosed)):
                        logger.warning("Bridge task ended with error: %r", exc)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            if not pipeline.completed:
                # WhisperLiveKit ends idle or failed sessions on its own. Tell the
                # dashboard instead of leaving a silently frozen transcript.
                logger.warning("ASR session ended before the client finished audio")
                try:
                    await client.send(
                        {
                            "type": "nlp",
                            "event": "error",
                            "message": "ASR session closed unexpectedly (idle session or server error).",
                        }
                    )
                    if browser.application_state == WebSocketState.CONNECTED:
                        await browser.close(code=1011, reason="asr session closed")
                except Exception:
                    pass
    except (OSError, websockets.WebSocketException) as exc:
        logger.warning("Could not reach ASR server at %s: %s", upstream_url, exc)
        try:
            await client.send(
                {
                    "type": "nlp",
                    "event": "error",
                    "message": f"NLP service cannot reach the ASR server at {upstream_url}",
                }
            )
        except Exception:
            pass
    finally:
        hub.remove_client(client)
        if pipeline is not None:
            if hub.active_pipeline is pipeline:
                hub.active_pipeline = None
            if pipeline.completed:
                hub.new_meeting_pending = True
            # Finish the tail of the meeting without blocking the socket teardown.
            asyncio.create_task(pipeline.close(flush=True))