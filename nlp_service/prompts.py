"""Prompt templates for structured meeting extraction.

All prompts keep the LLM on rails: a single JSON object, no prose, explicit
nulls for unknown fields. The schema is restated in the user turn as well,
because small/cheap models follow the nearest instruction most reliably.
"""

from __future__ import annotations

import json
from typing import Sequence

from .models import TopicSegment, TranscriptSegment, format_clock

EXTRACTION_SYSTEM_PROMPT = """You are a meeting/lecture analysis engine. You read a short window of \
diarized speech-to-text and return structured notes about it.

OUTPUT CONTRACT
- Reply with ONE JSON object and nothing else. No prose, no markdown fences, no comments.
- Top-level keys, always present, always arrays: "topic_segments", "key_points", "action_items".
- Schema:
  {
    "topic_segments": [
      {"start_time": <float seconds>, "end_time": <float seconds>, "topic_label": "<3-6 word label>"}
    ],
    "key_points": [
      {"topic_label": "<label of the matching topic segment>", "points": ["<short bullet>", "..."]}
    ],
    "action_items": [
      {"task": "<imperative, concrete task>", "owner": "<name or null>", "deadline": "<as stated or null>"}
    ]
  }

RULES
1. Times are absolute seconds from the start of the recording, copied or interpolated from the
   timestamps given in the transcript lines. Never emit times outside the window.
2. topic_segments must be ordered and non-overlapping. Use 1-3 segments per window. If the whole
   window is one topic, return a single segment spanning the window.
3. key_points: 2-4 bullets per topic segment, each under 20 words, paraphrased, no filler
   ("um", "you know") and no speaker attribution unless it matters.
4. action_items: only commitments, requests, or follow-ups that someone must actually do. Do not
   invent tasks. If nothing is actionable, return [].
5. owner: fill only when the transcript names or clearly identifies a person (e.g. "I'll send it"
   by Speaker 2 -> owner "Speaker 2" only if that is the only identification available). Otherwise null.
   Never guess a real name that was not spoken.
6. deadline: copy the time reference as spoken ("Friday", "end of week", "March 3"). If no time is
   mentioned, null. Do not convert to calendar dates.
7. If the window is small talk, noise, or unintelligible, return
   {"topic_segments": [], "key_points": [], "action_items": []}.
8. Output must be valid JSON: double quotes, no trailing commas, no NaN."""


def _prior_topic_block(prior_topics: Sequence[TopicSegment]) -> str:
    if not prior_topics:
        return "PREVIOUS TOPICS: none (this is the first window)"
    lines = [
        f"- {format_clock(t.start_time)}-{format_clock(t.end_time)}: {t.topic_label}"
        for t in prior_topics[-8:]
    ]
    return "PREVIOUS TOPICS (for continuity, do not repeat them as new segments):\n" + "\n".join(lines)


def build_extraction_prompt(
    segments: Sequence[TranscriptSegment],
    window_start: float,
    window_end: float,
    prior_topics: Sequence[TopicSegment] = (),
) -> str:
    """Build the user turn for one rolling-window extraction call."""
    transcript = "\n".join(seg.as_line() for seg in segments)
    return f"""WINDOW: {format_clock(window_start)} - {format_clock(window_end)} \
(absolute seconds {window_start:.1f} to {window_end:.1f})

{_prior_topic_block(prior_topics)}

TRANSCRIPT
{transcript}

Return the JSON object for this window only."""


SUMMARY_SYSTEM_PROMPT = """You are a meeting summarizer. You receive the full diarized transcript of a \
meeting or lecture plus the action items already extracted from it.

Reply with ONE JSON object and nothing else:
{
  "summary": "<5-8 sentence summary covering the agenda, decisions and conclusions>",
  "decisions": ["<decision>", "..."],
  "action_items": [{"task": "<task>", "owner": "<name or null>", "deadline": "<as stated or null>"}],
  "topics": ["<short topic label>", "..."]
}

Rules: no prose outside the JSON; keep owners and deadlines exactly as stated or null; do not invent
attendees, decisions, or tasks; if the meeting is empty, return empty arrays and an empty summary."""


def build_summary_prompt(
    transcript_lines: Sequence[str],
    action_items: Sequence[dict],
    max_chars: int = 24000,
) -> str:
    """Assemble the full-meeting prompt, trimming the transcript from the middle if huge."""
    body = "\n".join(transcript_lines)
    if len(body) > max_chars:
        head = body[: max_chars // 3]
        tail = body[-(max_chars - len(head) - 40):]
        body = f"{head}\n... [transcript trimmed] ...\n{tail}"
    actions = json.dumps(list(action_items), ensure_ascii=False, indent=2) if action_items else "[]"
    return f"""TRANSCRIPT
{body}

ACTION ITEMS ALREADY EXTRACTED
{actions}

Return the JSON object described in the system prompt."""