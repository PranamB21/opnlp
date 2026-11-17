"""Replay an audio file through the NLP bridge and print what the pipeline produces.

This is the headless equivalent of clicking "record" in the dashboard, and it is
also how the end-to-end smoke test runs:

    # ASR server must run with --pcm-input for raw audio frames
    wlk serve --model base --language en --diarization --pcm-input
    python nlp_service/app_entry.py  # or uvicorn
    python demo_audio/feed_demo.py demo_audio/meeting.wav --end
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys

import websockets

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2


def load_pcm(path: str) -> bytes:
    cmd = [
        "ffmpeg", "-i", path, "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-loglevel", "error", "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:300]}")
    return result.stdout


def _format_line(line: dict) -> str:
    return (
        f"[{line.get('start')} - {line.get('end')}] "
        f"Speaker {line.get('speaker')}: {(line.get('text') or '').strip()}"
    )


async def replay(
    audio_path: str,
    url: str,
    speed: float,
    chunk_seconds: float,
    verbose: bool,
    end: bool,
    http_base: str,
) -> int:
    pcm = load_pcm(audio_path)
    duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    chunk_bytes = int(chunk_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)
    print(f"Replaying {audio_path}: {duration:.1f}s → {url} (speed={speed})")

    printed: set[str] = set()
    last_line: dict = {}
    stop = asyncio.Event()

    async with websockets.connect(url, max_size=None) as ws:
        async def send_audio() -> None:
            offset = 0
            while offset < len(pcm):
                await ws.send(pcm[offset : offset + chunk_bytes])
                offset += chunk_bytes
                if speed > 0:
                    await asyncio.sleep(chunk_seconds / speed)
            await ws.send(b"")
            print("Audio finished; waiting for final results...")

        async def receive() -> None:
            nonlocal last_line
            async for raw in ws:
                payload = json.loads(raw)
                kind = payload.get("type")
                if kind == "nlp":
                    event = payload.get("event")
                    if event in {"state", "update", "final"}:
                        state = payload.get("state") or {}
                        stats = state.get("stats") or {}
                        print(
                            f"\n[NLP] window {stats.get('windows', 0)} · "
                            f"topics={len(state.get('topics') or [])} "
                            f"points={sum(len(g.get('points') or []) for g in state.get('key_points') or [])} "
                            f"actions={len(state.get('action_items') or [])}"
                        )
                        if verbose:
                            for topic in state.get("topics") or []:
                                print(f"   topic {topic['start_label']}-{topic['end_label']}: {topic['label']}")
                            for group in state.get("key_points") or []:
                                for point in group.get("points") or []:
                                    print(f"   point [{group['label']}] {point}")
                            for item in state.get("action_items") or []:
                                print(
                                    f"   action: {item['task']} (owner={item.get('owner')}, "
                                    f"deadline={item.get('deadline')})"
                                )
                    elif event == "status":
                        print(f"[NLP status] {payload.get('message')}")
                    elif event == "error":
                        print(f"[NLP error] {payload.get('message')}")
                    continue
                if kind == "config":
                    print(f"Server config: {payload}")
                    continue
                if kind == "ready_to_stop":
                    if last_line:
                        print(_format_line(last_line))
                    print("Server finished processing audio.")
                    stop.set()
                    return
                # WhisperLiveKit resends the full transcript and grows only the
                # last line in place. Print every line once it is settled (no
                # longer last) so the log reads like a transcript.
                lines = [
                    line for line in payload.get("lines") or []
                    if (line.get("text") or "").strip()
                ]
                for line in lines[:-1]:
                    key = str(line.get("start"))
                    if key in printed:
                        continue
                    printed.add(key)
                    print(_format_line(line))
                if lines:
                    last_line = lines[-1]
            stop.set()

        send_task = asyncio.create_task(send_audio())
        recv_task = asyncio.create_task(receive())
        try:
            await asyncio.wait_for(asyncio.gather(send_task, recv_task), timeout=duration / max(speed, 0.1) + 120)
        except asyncio.TimeoutError:
            print("Timed out waiting for the pipeline to finish.", file=sys.stderr)
            for task in (send_task, recv_task):
                task.cancel()
            await asyncio.gather(send_task, recv_task, return_exceptions=True)

    if end:
        await finalize(http_base)
    return 0


async def finalize(http_base: str) -> None:
    import httpx

    print("\nPOST /api/end ...")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{http_base}/api/end")
        if response.status_code != 200:
            print(f"end meeting failed: {response.status_code} {response.text[:300]}", file=sys.stderr)
            return
        payload = response.json()
        print("\n=== Meeting summary ===")
        print(payload.get("summary") or "(none)")
        print("\n=== Action items ===")
        for item in payload.get("action_items") or []:
            print(f"- {item['task']} (owner={item.get('owner')}, deadline={item.get('deadline')})")
        slack = payload.get("slack") or {}
        print(f"\nSlack: {slack.get('status')} — {slack.get('message')}")
        if slack.get("preview"):
            print("--- Slack preview ---")
            print(slack["preview"])
    except Exception as exc:
        print(f"end meeting request failed: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Audio file (any format ffmpeg decodes)")
    parser.add_argument("--url", default="ws://localhost:8001/asr", help="NLP bridge WebSocket URL")
    parser.add_argument("--http", default="http://localhost:8001", help="NLP service HTTP base URL")
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 = real time, 4 = faster")
    parser.add_argument("--chunk-seconds", type=float, default=0.5)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--end", action="store_true", help="Call /api/end afterwards (summary + Slack)")
    args = parser.parse_args()
    return asyncio.run(
        replay(args.audio, args.url, args.speed, args.chunk_seconds, args.verbose, args.end, args.http)
    )


if __name__ == "__main__":
    raise SystemExit(main())