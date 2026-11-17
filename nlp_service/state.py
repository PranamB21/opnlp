"""In-memory meeting state: merged topics, key points, action items, transcript log.

One meeting is active at a time. Dashboards are passive viewers: they receive a
snapshot on connect and full-state updates after every window, so a browser
refresh mid-lecture restores all three panels.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .models import (
    ActionItem,
    Extraction,
    KeyPointGroup,
    TopicSegment,
    TranscriptSegment,
    format_clock,
)


def _norm_task(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())[:100]


def _norm_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@dataclass
class MeetingState:
    topics: list[TopicSegment] = field(default_factory=list)
    key_points: list[KeyPointGroup] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    segments: list[TranscriptSegment] = field(default_factory=list)
    windows_processed: int = 0
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    summary: str = ""
    decisions: list[str] = field(default_factory=list)
    ended: bool = False

    # -- mutation ----------------------------------------------------------

    def add_segments(self, segments: list[TranscriptSegment]) -> None:
        self.segments.extend(segments)

    def merge(self, extraction: Extraction) -> None:
        self.windows_processed += 1
        self.updated_at = time.time()

        for topic in extraction.topic_segments:
            self._merge_topic(topic)

        for group in extraction.key_points:
            self._merge_key_points(group)

        for item in extraction.action_items:
            self.merge_action_item(item)

    def _merge_topic(self, topic: TopicSegment) -> None:
        label = _norm_label(topic.topic_label)
        for existing in self.topics:
            if _norm_label(existing.topic_label) == label and topic.start_time <= existing.end_time + 2.0:
                existing.end_time = max(existing.end_time, topic.end_time)
                existing.start_time = min(existing.start_time, topic.start_time)
                return
        # Trim a sliver of overlap so the timeline reads as consecutive blocks.
        if self.topics:
            last = self.topics[-1]
            if topic.start_time < last.end_time and last.end_time - topic.start_time <= 3.0:
                topic.start_time = last.end_time
        if topic.end_time - topic.start_time < 0.5:
            return
        self.topics.append(topic)
        self.topics.sort(key=lambda t: t.start_time)

    def _merge_key_points(self, group: KeyPointGroup) -> None:
        for existing in self.key_points:
            if _norm_label(existing.topic_label) == _norm_label(group.topic_label):
                for point in group.points:
                    if _norm_label(point) not in {_norm_label(p) for p in existing.points}:
                        existing.points.append(point)
                existing.end_time = max(existing.end_time, group.end_time)
                return
        self.key_points.append(group)

    def merge_action_item(self, item: ActionItem) -> None:
        key = _norm_task(item.task)
        for existing in self.action_items:
            existing_key = _norm_task(existing.task)
            same = existing_key == key
            # Streaming revisions of a live line arrive as growing prefixes
            # ("Send the deck" -> "Send the deck to Priya"), so treat a prefix
            # relationship as the same task and keep the longer wording.
            related = (
                not same
                and len(key) >= 8
                and len(existing_key) >= 8
                and (key.startswith(existing_key) or existing_key.startswith(key))
            )
            if not (same or related):
                continue
            if related and len(item.task) > len(existing.task):
                existing.task = item.task
            if existing.owner is None and item.owner:
                existing.owner = item.owner
            if existing.deadline is None and item.deadline:
                existing.deadline = item.deadline
            return
        self.action_items.append(item)

    def reset(self) -> None:
        self.topics.clear()
        self.key_points.clear()
        self.action_items.clear()
        self.segments.clear()
        self.windows_processed = 0
        self.started_at = time.time()
        self.updated_at = time.time()
        self.summary = ""
        self.decisions = []
        self.ended = False

    # -- views -------------------------------------------------------------

    def transcript_lines(self) -> list[str]:
        return [seg.as_line() for seg in self.segments]

    def actions_as_dicts(self) -> list[dict]:
        return [
            {"task": item.task, "owner": item.owner, "deadline": item.deadline}
            for item in self.action_items
        ]

    def snapshot(self) -> dict[str, Any]:
        return {
            "topics": [
                {
                    "start": round(t.start_time, 2),
                    "end": round(t.end_time, 2),
                    "start_label": format_clock(t.start_time),
                    "end_label": format_clock(t.end_time),
                    "label": t.topic_label,
                }
                for t in self.topics
            ],
            "key_points": [
                {
                    "label": g.topic_label,
                    "start_label": format_clock(g.start_time),
                    "points": list(g.points),
                }
                for g in self.key_points
            ],
            "action_items": [
                {"task": i.task, "owner": i.owner, "deadline": i.deadline} for i in self.action_items
            ],
            "stats": {
                "windows": self.windows_processed,
                "segments": len(self.segments),
                "duration": format_clock(self.segments[-1].end if self.segments else 0.0),
                "updated_at": self.updated_at,
                "ended": self.ended,
            },
            "summary": self.summary,
            "decisions": list(self.decisions),
        }

    def export_markdown(self) -> str:
        lines = [f"# Meeting notes", ""]
        if self.summary:
            lines += [self.summary, ""]
        if self.topics:
            lines += ["## Topics", ""]
            for topic in self.topics:
                lines.append(
                    f"- **{format_clock(topic.start_time)}-{format_clock(topic.end_time)}** {topic.topic_label}"
                )
            lines.append("")
        if self.key_points:
            lines += ["## Key points", ""]
            for group in self.key_points:
                lines.append(f"**{group.topic_label}**")
                lines += [f"- {point}" for point in group.points]
                lines.append("")
        if self.action_items:
            lines += ["## Action items", ""]
            for item in self.action_items:
                owner = item.owner or "unassigned"
                deadline = f" — due {item.deadline}" if item.deadline else ""
                lines.append(f"- [ ] {item.task} ({owner}){deadline}")
        return "\n".join(lines).strip() + "\n"


@dataclass(eq=False)
class SessionClient:
    """A dashboard connection receiving pushed NLP updates."""

    send: Callable[[dict], Awaitable[None]]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionHub:
    """Registry of dashboard clients plus the shared meeting state."""

    def __init__(self, state: Optional[MeetingState] = None):
        self.state = state or MeetingState()
        self._clients: set[SessionClient] = set()
        self._state_lock = asyncio.Lock()
        self.provider = "unknown"
        self.busy_windows = 0
        # Set by the bridge so /api/end can flush the live session.
        self.active_pipeline: Optional[Any] = None
        # True once a meeting has been ended deliberately, so the next recording
        # starts from clean panels instead of appending to the old meeting.
        self.new_meeting_pending: bool = False

    def add_client(self, client: SessionClient) -> None:
        self._clients.add(client)

    def remove_client(self, client: SessionClient) -> None:
        self._clients.discard(client)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def broadcast(self, payload: dict) -> None:
        for client in list(self._clients):
            try:
                async with client.lock:
                    await client.send(payload)
            except Exception:
                self.remove_client(client)

    async def broadcast_state(self, event: str = "update") -> None:
        async with self._state_lock:
            snapshot = self.state.snapshot()
        await self.broadcast({"type": "nlp", "event": event, "state": snapshot})

    async def broadcast_status(self, message: str, **extra: Any) -> None:
        payload = {
            "type": "nlp",
            "event": "status",
            "provider": self.provider,
            "busy": self.busy_windows > 0,
            "message": message,
        }
        payload.update(extra)
        await self.broadcast(payload)