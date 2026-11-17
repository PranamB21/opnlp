"""LLM-backed structured extraction.

Provider selection is deliberately thin: one `_call_*` per API flavour, a shared
JSON-repair step, and an offline heuristic (`mock`) so the pipeline still runs
end to end when no API key is configured.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional, Sequence

import httpx

from .config import Settings
from .models import (
    ActionItem,
    Extraction,
    KeyPointGroup,
    TopicSegment,
    TranscriptSegment,
    normalize_extraction,
)
from .prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    build_extraction_prompt,
    build_summary_prompt,
)

logger = logging.getLogger(__name__)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class ExtractionError(RuntimeError):
    pass


def parse_json_object(raw: str) -> dict:
    """Extract the first JSON object from a model response.

    Handles markdown fences, leading prose, and trailing commentary, which are
    the three failure modes actually seen with JSON-mode prompts.
    """
    if raw is None:
        raise ExtractionError("empty model response")
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return {"topic_segments": value}
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ExtractionError(f"no JSON object in model response: {text[:200]!r}")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ExtractionError(f"malformed JSON object: {exc}") from exc
    raise ExtractionError("unterminated JSON object in model response")


class Extractor:
    def __init__(self, settings: Settings, transport: Optional[httpx.AsyncBaseTransport] = None):
        self.settings = settings
        self.provider = settings.resolved_provider()
        self._client = httpx.AsyncClient(timeout=settings.llm_timeout_seconds, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- public API --------------------------------------------------------

    async def extract(
        self,
        segments: Sequence[TranscriptSegment],
        window_start: float,
        window_end: float,
        prior_topics: Sequence[TopicSegment] = (),
    ) -> Extraction:
        prompt = build_extraction_prompt(segments, window_start, window_end, prior_topics)
        payload = await self._complete_json(EXTRACTION_SYSTEM_PROMPT, prompt)
        extraction = normalize_extraction(payload, window_start, window_end)
        if self.provider == "mock":
            extraction = _merge_heuristic_tail(extraction, segments, window_start, window_end)
        return extraction

    async def summarize(
        self,
        transcript_lines: Sequence[str],
        action_items: Sequence[dict],
    ) -> dict:
        prompt = build_summary_prompt(transcript_lines, action_items)
        payload = await self._complete_json(SUMMARY_SYSTEM_PROMPT, prompt)
        if not isinstance(payload, dict):
            payload = {}
        summary = str(payload.get("summary") or "").strip()
        decisions = [str(d).strip() for d in payload.get("decisions") or [] if str(d).strip()]
        topics = [str(t).strip() for t in payload.get("topics") or [] if str(t).strip()]
        items = []
        for item in payload.get("action_items") or []:
            if isinstance(item, str):
                item = {"task": item}
            if isinstance(item, dict) and str(item.get("task") or "").strip():
                items.append(
                    {
                        "task": str(item["task"]).strip(),
                        "owner": item.get("owner") or None,
                        "deadline": item.get("deadline") or None,
                    }
                )
        if self.provider == "mock":
            summary = summary or _mock_summary(transcript_lines, action_items)
        return {"summary": summary, "decisions": decisions, "topics": topics, "action_items": items}

    # -- providers ---------------------------------------------------------

    async def _complete_json(self, system_prompt: str, user_prompt: str) -> Any:
        if self.provider == "anthropic":
            raw = await self._call_anthropic(system_prompt, user_prompt)
        elif self.provider == "openai":
            raw = await self._call_openai(system_prompt, user_prompt)
        else:
            return _mock_payload(user_prompt)
        return parse_json_object(raw)

    async def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        if not self.settings.anthropic_api_key:
            raise ExtractionError("ANTHROPIC_API_KEY is not set")
        body = {
            "model": self.settings.anthropic_model,
            "max_tokens": self.settings.llm_max_output_tokens,
            "temperature": 0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        headers = {
            "x-api-key": self.settings.anthropic_api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        data = await self._post_with_retry(ANTHROPIC_URL, body, headers)
        blocks = data.get("content") or []
        texts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
        if not texts:
            raise ExtractionError("anthropic response had no text block")
        return "".join(texts)

    async def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        if not self.settings.openai_api_key:
            raise ExtractionError("OPENAI_API_KEY is not set")
        body = {
            "model": self.settings.openai_model,
            "temperature": 0,
            "max_tokens": self.settings.llm_max_output_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "content-type": "application/json",
        }
        url = self.settings.openai_base_url.rstrip("/") + "/chat/completions"
        data = await self._post_with_retry(url, body, headers)
        choices = data.get("choices") or []
        if not choices:
            raise ExtractionError("openai response had no choices")
        return (choices[0].get("message") or {}).get("content") or ""

    async def _post_with_retry(self, url: str, body: dict, headers: dict, attempts: int = 2) -> dict:
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                response = await self._client.post(url, json=body, headers=headers)
                if response.status_code >= 500 or response.status_code == 429:
                    raise ExtractionError(f"provider returned {response.status_code}: {response.text[:200]}")
                if response.status_code >= 400:
                    raise ExtractionError(f"provider rejected request ({response.status_code}): {response.text[:300]}")
                return response.json()
            except (httpx.HTTPError, ExtractionError) as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    await asyncio.sleep(1.5 * (attempt + 1))
        raise ExtractionError(f"LLM call failed: {last_error}")


# ---------------------------------------------------------------------------
# Offline heuristic fallback
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_ACTION_PATTERNS = (
    re.compile(r"\b(?:i will|i'll|we will|we'll|we need to|you need to|you should|let's|let us|make sure|"
               r"I need to|we should|please|can you|could you|follow up|send|schedule|prepare|review|"
               r"submit|share|update|check)\b", re.IGNORECASE),
    re.compile(r"\b(?:todo|action item|deadline)\b", re.IGNORECASE),
)
_DEADLINE = re.compile(
    r"\b(?:by|before|until|on|due)\s+((?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"tomorrow|today|tonight|end of (?:day|week|month)|eod|eow|next week|next month|[A-Z][a-z]+ \d{1,2}))",
    re.IGNORECASE,
)


def _mock_payload(prompt: str) -> dict:
    """Offline stand-in: the pipeline shape without a model call."""
    return {"topic_segments": [], "key_points": [], "action_items": []}


def _top_terms(text: str, count: int = 4) -> str:
    stop = {
        "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with", "is", "are", "was",
        "were", "be", "been", "it", "this", "that", "we", "i", "you", "they", "he", "she", "so", "if",
        "at", "as", "by", "from", "our", "your", "have", "has", "had", "do", "does", "did", "not", "can",
        "will", "would", "should", "about", "into", "like", "just", "then", "there", "what", "when", "how",
    }
    words = [re.sub(r"[^A-Za-z0-9\-]", "", w) for w in text.split()]
    picked = [w for w in words if w and w.lower() not in stop and len(w) > 3][:count]
    if not picked:
        picked = words[:count]
    label = " ".join(picked).strip()
    return label[:60] or "Untitled topic"


def _merge_heuristic_tail(
    extraction: Extraction,
    segments: Sequence[TranscriptSegment],
    window_start: float,
    window_end: float,
) -> Extraction:
    """Fill the gaps left by the offline stub using simple per-speaker grouping."""
    if segments and not extraction.topic_segments:
        groups: list[list[TranscriptSegment]] = []
        current: list[TranscriptSegment] = []
        for seg in segments:
            if current and (seg.speaker != current[-1].speaker or seg.start - current[-1].end >= 4.0):
                groups.append(current)
                current = []
            current.append(seg)
        if current:
            groups.append(current)

        # Merge tiny groups into the previous one so labels are not one-word topics.
        merged: list[list[TranscriptSegment]] = []
        for group in groups:
            span = group[-1].end - group[0].start
            if merged and span < 8.0:
                merged[-1].extend(group)
            else:
                merged.append(group)

        for group in merged:
            text = " ".join(g.text for g in group)
            label = _top_terms(text)
            start, end = group[0].start, group[-1].end
            extraction.topic_segments.append(TopicSegment(start_time=start, end_time=end, topic_label=label))
            sentences = [
                s.strip() for s in _SENTENCE_SPLIT.split(text) if len(s.strip()) > 12
            ]
            sentences.sort(key=len, reverse=True)
            points = [s[:160] for s in sentences[:3]]
            if points:
                extraction.key_points.append(
                    KeyPointGroup(topic_label=label, start_time=start, end_time=end, points=points)
                )

    if not extraction.action_items:
        filler = {
            "we'll do", "we will do", "got it", "understood", "sure", "okay", "ok",
            "let's wrap up", "let us wrap up", "will do", "sounds good",
        }
        for seg in segments:
            sentences = [s.strip() for s in _SENTENCE_SPLIT.split(seg.text) if s.strip()]
            for sentence in sentences:
                if not any(p.search(sentence) for p in _ACTION_PATTERNS):
                    continue
                if len(sentence) < 12 or sentence.endswith("?"):
                    continue
                if sentence.lower().strip(" .!") in filler:
                    continue
                deadline_match = _DEADLINE.search(sentence)
                owner = seg.speaker if re.search(r"\b(i will|i'll|i need to)\b", sentence, re.I) else None
                extraction.action_items.append(
                    ActionItem(
                        task=sentence[:200],
                        owner=owner,
                        deadline=deadline_match.group(1).title() if deadline_match else None,
                    )
                )
                break
    return extraction


def _mock_summary(transcript_lines: Sequence[str], action_items: Sequence[dict]) -> str:
    words = sum(len(line.split()) for line in transcript_lines)
    task_text = "; ".join(item.get("task", "") for item in action_items[:5] if isinstance(item, dict))
    summary = f"Transcript covers roughly {words} words across {len(transcript_lines)} speaker turns."
    if task_text:
        summary += f" Open follow-ups: {task_text}."
    return summary