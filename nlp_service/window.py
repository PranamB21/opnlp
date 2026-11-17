"""Rolling-window buffer over the live transcript.

The NLP service never processes the whole transcript on every update. Committed
segments are appended to a pending window which is cut when it gets long enough,
when the speaker changes, or when a pause shows the thought has ended.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .models import TranscriptSegment


@dataclass
class WindowBatch:
    segments: list[TranscriptSegment]
    start: float
    end: float
    reason: str
    index: int

    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class RollingWindow:
    window_seconds: float = 40.0
    min_window_seconds: float = 20.0
    max_window_seconds: float = 90.0
    pause_gap_seconds: float = 2.5

    _pending: list[TranscriptSegment] = field(default_factory=list)
    # Canonical segment objects, keyed by (speaker, start). The same objects are
    # appended to the meeting transcript log, so revisions update both views.
    _seen: dict[tuple[str, int], TranscriptSegment] = field(default_factory=dict)
    _finalized: set[tuple[str, int]] = field(default_factory=set)
    _index: int = 0

    # -- ingestion ---------------------------------------------------------

    @staticmethod
    def segment_key(segment: TranscriptSegment) -> tuple[str, int]:
        return (segment.speaker, int(round(segment.start * 100)))

    def ingest(self, segments: Iterable[TranscriptSegment]) -> list[TranscriptSegment]:
        """Merge a transcription frame into the window; return brand-new segments.

        WhisperLiveKit resends the full committed transcript every frame and a
        line grows in place: same start, later end, longer text. Identity is
        therefore (speaker, start). Growth is applied to the existing object
        until that line has been cut into an analyzed window; after that the
        revision is ignored so finished analysis is never rewritten.
        """
        added: list[TranscriptSegment] = []
        for seg in segments:
            if not seg.text or not seg.text.strip():
                continue
            key = self.segment_key(seg)
            existing = self._seen.get(key)
            if existing is not None:
                if key in self._finalized:
                    continue
                if seg.end >= existing.end:
                    existing.end = seg.end
                existing.text = seg.text
                continue
            self._seen[key] = seg
            self._pending.append(seg)
            added.append(seg)
        return added

    # -- cutting -----------------------------------------------------------

    @property
    def pending(self) -> list[TranscriptSegment]:
        return list(self._pending)

    def pending_duration(self) -> float:
        if not self._pending:
            return 0.0
        return self._pending[-1].end - self._pending[0].start

    def boundary_reason(self) -> Optional[str]:
        """Why the pending window should be cut now, or None."""
        if not self._pending:
            return None
        duration = self.pending_duration()
        if duration >= self.max_window_seconds:
            return "max_duration"
        if duration >= self.window_seconds:
            return "duration"
        if len(self._pending) >= 2 and duration >= self.min_window_seconds:
            previous, current = self._pending[-2], self._pending[-1]
            if current.start - previous.end >= self.pause_gap_seconds:
                return "pause"
            if current.speaker != previous.speaker:
                return "speaker_change"
        return None

    def take(self, reason: str = "manual") -> Optional[WindowBatch]:
        """Cut the pending window, returning None when there is nothing to send."""
        if not self._pending:
            return None
        batch = WindowBatch(
            segments=list(self._pending),
            start=self._pending[0].start,
            end=self._pending[-1].end,
            reason=reason,
            index=self._index,
        )
        self._index += 1
        for segment in batch.segments:
            self._finalized.add(self.segment_key(segment))
        self._pending.clear()
        return batch

    def take_if_ready(self) -> Optional[WindowBatch]:
        reason = self.boundary_reason()
        if reason is None:
            return None
        return self.take(reason)

    def take_remaining(self, reason: str = "flush") -> Optional[WindowBatch]:
        return self.take(reason)