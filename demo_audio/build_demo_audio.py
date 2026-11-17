"""Build a two-speaker demo meeting WAV with edge-tts.

Synthesizes a scripted project standup with two distinct neural voices, so the
pipeline can be demoed (and regression-tested) without a live microphone.
Requires `edge-tts` and network access on first run; output is cached.

Usage:
    python demo_audio/build_demo_audio.py
    python demo_audio/build_demo_audio.py --out demo_audio/meeting.wav
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT: list[tuple[int, str]] = [
    (1, "Okay team, let's start the weekly project standup. First item is the speech recognition backend."),
    (2, "The backend is running, but the diarization model is slower than we expected on the laptop GPU."),
    (1, "How much slower are we talking about?"),
    (2, "About two seconds of lag per chunk. I can profile the sortformer model this afternoon."),
    (1, "Good. Please write the numbers into the performance document by Thursday."),
    (2, "Will do. One more thing, the dashboard panels still need the key points and the action items."),
    (1, "Right, that is the second topic, the dashboard. Sara, can you finish the timeline panel before the demo?"),
    (2, "Yes, I will finish the topic timeline by Friday, and I will test the mobile layout too."),
    (1, "Great. Third topic is the NLP service. We need to pick the extraction model and finalize the prompt."),
    (2, "I think we should use the Claude API with a strict JSON schema, and fall back to a local heuristic if the key is missing."),
    (1, "Agreed. I will write the README section about our design choices tonight."),
    (2, "Should we also add the Slack webhook?"),
    (1, "Yes, but only the incoming webhook for the end meeting button. We can skip a full bot for now."),
    (2, "Understood. I will add the webhook and post a test message tomorrow morning."),
    (1, "Perfect. Let's wrap up. The deadline for the whole pipeline is Monday, so the demo runs on Tuesday."),
    (2, "Got it. I will send the updated checklist by the end of the week."),
]

VOICES = {
    1: "en-US-GuyNeural",
    2: "en-US-JennyNeural",
}

GAP_SECONDS = 0.55
SAMPLE_RATE = 16000


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{result.stderr[-800:]}")


async def _synth(text: str, voice: str, dest: Path) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(dest))


def synthesize_turn(text: str, voice: str, dest: Path) -> None:
    for attempt in range(3):
        try:
            asyncio.run(_synth(text, voice, dest))
            if dest.stat().st_size > 0:
                return
        except Exception as exc:  # network hiccups are common on first run
            if attempt == 2:
                raise RuntimeError(f"edge-tts failed for {voice}: {exc}") from exc
            asyncio.run(asyncio.sleep(1.0 + attempt))


def to_wav(src: Path, dest: Path) -> None:
    run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(dest),
        ]
    )


def build(out: Path, force: bool = False) -> Path:
    if out.exists() and not force:
        print(f"{out} already exists; use --force to rebuild.")
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        silence = tmpdir / "silence.wav"
        run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
                "-t", str(GAP_SECONDS), "-c:a", "pcm_s16le", str(silence),
            ]
        )

        concat_list = tmpdir / "concat.txt"
        entries: list[str] = []
        for index, (speaker, text) in enumerate(SCRIPT):
            mp3 = tmpdir / f"turn_{index:02d}.mp3"
            wav = tmpdir / f"turn_{index:02d}.wav"
            print(f"  [{index + 1}/{len(SCRIPT)}] speaker {speaker}: {text[:60]}...")
            synthesize_turn(text, VOICES[speaker], mp3)
            to_wav(mp3, wav)
            entries.append(f"file '{wav}'")
            entries.append(f"file '{silence}'")

        concat_list.write_text("\n".join(entries) + "\n", encoding="utf-8")
        run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                "-i", str(concat_list), "-af", "loudnorm=I=-18:TP=-2",
                "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(out),
            ]
        )

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
        capture_output=True,
        text=True,
    )
    print(f"Wrote {out} ({probe.stdout.strip()} s, {len(SCRIPT)} turns, 2 speakers)")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "meeting.wav"),
        help="Output WAV path",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if the file exists")
    args = parser.parse_args()
    try:
        build(Path(args.out), force=args.force)
    except Exception as exc:
        print(f"Failed to build demo audio: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())