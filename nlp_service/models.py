"""Data models for transcript segments and LLM extractions.

The LLM is instructed to emit a strict JSON shape, but real responses drift:
fields get renamed, key points arrive as bare strings, action items come back as
plain sentences. `normalize_extraction` is the tolerant boundary between the raw
model output and the rest of the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str

    def as_line(self) -> str:
        return f"[{format_clock(self.start)} - {format_clock(self.end)}] {self.speaker}: {self.text}"


def format_clock(seconds: float) -> str:
    """Render seconds as M:SS (H:MM:SS past an hour), matching dashboard labels."""
    total = int(round(max(0.0, float(seconds or 0.0))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def parse_clock(value: Any) -> Optional[float]:
    """Parse WhisperLiveKit's 'H:MM:SS.cc' timestamps (also accepts floats)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    return None


@dataclass
class TopicSegment:
    start_time: float
    end_time: float
    topic_label: str


@dataclass
class KeyPointGroup:
    topic_label: str
    start_time: float
    end_time: float
    points: list[str] = field(default_factory=list)


@dataclass
class ActionItem:
    task: str
    owner: Optional[str] = None
    deadline: Optional[str] = None


@dataclass
class Extraction:
    topic_segments: list[TopicSegment] = field(default_factory=list)
    key_points: list[KeyPointGroup] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)


_TOPIC_LABEL_KEYS = ("topic_label", "topic", "label", "name", "title")
_START_KEYS = ("start_time", "start", "from", "segment_start")
_END_KEYS = ("end_time", "end", "to", "segment_end")


def _first(mapping: dict, keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _clean_text(value: Any, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = text.lstrip("-•* ").strip()
    return text[:limit]


def _as_float(value: Any, fallback: float) -> float:
    parsed = parse_clock(value)
    return fallback if parsed is None else parsed


def _points_from(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    points = []
    for item in value:
        if isinstance(item, dict):
            text = item.get("point") or item.get("text") or item.get("bullet") or item.get("key_point")
        else:
            text = item
        cleaned = _clean_text(text)
        if cleaned:
            points.append(cleaned)
    return points


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = _clean_text(value, limit=80)
    if not text or text.lower() in {"null", "none", "n/a", "na", "unknown", "not mentioned", "unassigned"}:
        return None
    return text


def normalize_extraction(payload: Any, window_start: float, window_end: float) -> Extraction:
    """Coerce a loosely-shaped LLM payload into `Extraction`.

    Missing or malformed fields degrade to empty lists / null values instead of
    raising, so one bad LLM response cannot take the pipeline down mid-demo.
    """
    if not isinstance(payload, dict):
        payload = {}

    extraction = Extraction()

    raw_topics = payload.get("topic_segments") or payload.get("topics") or []
    for item in raw_topics if isinstance(raw_topics, list) else []:
        if isinstance(item, str):
            item = {"topic_label": item}
        if not isinstance(item, dict):
            continue
        label = _clean_text(_first(item, _TOPIC_LABEL_KEYS), limit=60)
        if not label:
            continue
        start = _as_float(_first(item, _START_KEYS), window_start)
        end = _as_float(_first(item, _END_KEYS), window_end)
        start, end = min(start, end), max(start, end)
        extraction.topic_segments.append(TopicSegment(start_time=start, end_time=end, topic_label=label))

    raw_points = payload.get("key_points") or payload.get("keypoints") or []
    if isinstance(raw_points, dict):
        raw_points = [raw_points]
    for item in raw_points if isinstance(raw_points, list) else []:
        if isinstance(item, str):
            # Flat bullet list: attach to the nearest group, else derive one from
            # the first topic segment so nothing is dropped.
            group = extraction.key_points[-1] if extraction.key_points else None
            if group is None:
                label = (
                    extraction.topic_segments[0].topic_label if extraction.topic_segments else "Key points"
                )
                start = extraction.topic_segments[0].start_time if extraction.topic_segments else window_start
                end = extraction.topic_segments[0].end_time if extraction.topic_segments else window_end
                group = KeyPointGroup(topic_label=label, start_time=start, end_time=end)
                extraction.key_points.append(group)
            cleaned = _clean_text(item)
            if cleaned:
                group.points.append(cleaned)
            continue
        if not isinstance(item, dict):
            continue
        label = _clean_text(_first(item, _TOPIC_LABEL_KEYS), limit=60) or (
            extraction.topic_segments[0].topic_label if extraction.topic_segments else "Key points"
        )
        group = KeyPointGroup(
            topic_label=label,
            start_time=_as_float(_first(item, _START_KEYS), window_start),
            end_time=_as_float(_first(item, _END_KEYS), window_end),
        )
        group.points = _points_from(item.get("points") or item.get("key_points") or item.get("bullets"))
        if group.points:
            extraction.key_points.append(group)

    raw_actions = payload.get("action_items") or payload.get("actions") or []
    for item in raw_actions:
        if isinstance(item, str):
            item = {"task": item}
        if not isinstance(item, dict):
            continue
        task = _clean_text(item.get("task") or item.get("action") or item.get("item"), limit=200)
        if not task:
            continue
        extraction.action_items.append(
            ActionItem(
                task=task,
                owner=_optional_str(item.get("owner")),
                deadline=_optional_str(item.get("deadline")),
            )
        )

    return extraction