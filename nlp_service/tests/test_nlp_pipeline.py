"""Unit and API tests for the NLP service (no GPU, no network, no API key)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from nlp_service.app import app
from nlp_service.bridge import build_upstream_url, parse_transcript_lines
from nlp_service.config import Settings
from nlp_service.extractor import Extractor, parse_json_object
from nlp_service.models import (
    ActionItem,
    Extraction,
    KeyPointGroup,
    TopicSegment,
    TranscriptSegment,
    format_clock,
    parse_clock,
    normalize_extraction,
)
from nlp_service.state import MeetingState
from nlp_service.window import RollingWindow


def seg(start: float, end: float, text: str, speaker: int = 1) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text, speaker=f"Speaker {speaker}")


# ---------------------------------------------------------------------------
# Timestamps and payload parsing
# ---------------------------------------------------------------------------


def test_clock_roundtrip():
    assert parse_clock("0:01:23.45") == pytest.approx(83.45)
    assert parse_clock("1:00:00.00") == pytest.approx(3600.0)
    assert parse_clock(12.5) == pytest.approx(12.5)
    assert parse_clock("") is None
    assert format_clock(83) == "1:23"
    assert format_clock(3725) == "1:02:05"


def test_parse_transcript_lines_skips_silence_and_bad_rows():
    payload = {
        "lines": [
            {"speaker": 1, "text": "Hello there", "start": "0:00:01.00", "end": "0:00:03.00"},
            {"speaker": -2, "text": "", "start": "0:00:03.00", "end": "0:00:05.00"},
            {"speaker": 2, "text": "  ", "start": "0:00:05.00", "end": "0:00:06.00"},
            {"speaker": 2, "text": "Second speaker", "start": "0:00:06.00", "end": "0:00:09.00"},
            {"speaker": 2, "text": "Broken span", "start": "0:00:12.00", "end": "0:00:10.00"},
            "not-a-line",
        ]
    }
    segments = parse_transcript_lines(payload)
    assert [s.text for s in segments] == ["Hello there", "Second speaker"]
    assert segments[1].speaker == "Speaker 2"


def test_build_upstream_url_forces_full_mode():
    url = build_upstream_url("ws://localhost:8000/asr", "language=en&mode=diff&context=WLK")
    assert "mode=full" in url and "mode=diff" not in url
    assert "language=en" in url


# ---------------------------------------------------------------------------
# Rolling window
# ---------------------------------------------------------------------------


def test_window_dedupes_repeated_frames():
    window = RollingWindow()
    first = window.ingest([seg(0, 3, "one"), seg(3, 6, "two")])
    again = window.ingest([seg(0, 3, "one"), seg(3, 6, "two")])
    assert len(first) == 2
    assert again == []
    assert window.pending_duration() == pytest.approx(6.0)


def test_window_updates_revised_text_in_place():
    window = RollingWindow()
    window.ingest([seg(0, 3, "helo world")])
    window.ingest([seg(0, 3, "hello world")])
    assert [s.text for s in window.pending] == ["hello world"]


def test_window_grows_line_until_cut_then_ignores_revisions():
    window = RollingWindow(window_seconds=10, min_window_seconds=2, pause_gap_seconds=9)
    assert [s.text for s in window.ingest([seg(0, 1, "Okay")])] == ["Okay"]
    # WhisperLiveKit resends the line as it grows: same start, later end.
    assert window.ingest([seg(0, 4, "Okay team, let's start")]) == []
    assert window.pending[0].text == "Okay team, let's start"
    assert window.pending[0].end == pytest.approx(4)
    window.ingest([seg(4, 12, "the standup")])
    batch = window.take("duration")
    assert [s.text for s in batch.segments] == ["Okay team, let's start", "the standup"]
    # After the window is analyzed, a late growth frame must not rewrite it.
    window.ingest([seg(0, 8, "Okay team, let's start the standup")])
    assert batch.segments[0].text == "Okay team, let's start"


def test_pipeline_treats_growing_line_as_one_segment():
    from nlp_service.bridge import TranscriptPipeline
    from nlp_service.state import SessionHub

    settings = Settings(
        llm_provider="mock", window_seconds=1000, min_window_seconds=1000, max_window_seconds=1000
    )
    hub = SessionHub()
    extractor = Extractor(settings)
    pipeline = TranscriptPipeline(settings, hub, extractor)

    def frame(text: str, end_seconds: int) -> dict:
        return {
            "lines": [
                {
                    "speaker": 1,
                    "text": text,
                    "start": "0:00:00.00",
                    "end": f"0:00:{end_seconds:02d}.00",
                }
            ]
        }

    pipeline.ingest_frame(frame("Okay", 1))
    pipeline.ingest_frame(frame("Okay team, let's start", 4))
    pipeline.ingest_frame(frame("Okay team, let's start the standup", 7))
    assert len(hub.state.segments) == 1
    assert hub.state.segments[0].text == "Okay team, let's start the standup"
    assert hub.state.segments[0].end == pytest.approx(7)
    assert pipeline.completed is False
    pipeline.ingest_frame({"type": "ready_to_stop"})
    assert pipeline.completed is True
    __import__("asyncio").run(extractor.aclose())


def test_window_cuts_on_speaker_change_after_minimum():
    window = RollingWindow(window_seconds=40, min_window_seconds=5, pause_gap_seconds=99)
    window.ingest([seg(0, 6, "a", speaker=1), seg(6, 9, "b", speaker=1)])
    assert window.boundary_reason() is None
    window.ingest([seg(9, 12, "c", speaker=2)])
    assert window.boundary_reason() == "speaker_change"
    batch = window.take_if_ready()
    assert batch is not None and batch.reason == "speaker_change"
    assert [s.text for s in batch.segments] == ["a", "b", "c"]


def test_window_cuts_on_pause_and_max_duration():
    pause_window = RollingWindow(window_seconds=40, min_window_seconds=3, pause_gap_seconds=2.0)
    pause_window.ingest([seg(0, 4, "a")])
    pause_window.ingest([seg(9, 11, "b")])
    assert pause_window.boundary_reason() == "pause"

    long_window = RollingWindow(window_seconds=100, min_window_seconds=1, max_window_seconds=20)
    long_window.ingest([seg(0, 25, "a")])
    assert long_window.boundary_reason() == "max_duration"


def test_window_duration_rule():
    window = RollingWindow(window_seconds=10, min_window_seconds=2)
    window.ingest([seg(0, 5, "a")])
    assert window.boundary_reason() is None
    window.ingest([seg(5, 11, "b")])
    assert window.boundary_reason() == "duration"
    assert window.take_if_ready().reason == "duration"
    assert window.take_if_ready() is None


# ---------------------------------------------------------------------------
# LLM output handling
# ---------------------------------------------------------------------------


def test_parse_json_object_handles_fences_and_prose():
    assert parse_json_object('```json\n{"topic_segments": []}\n```') == {"topic_segments": []}
    assert parse_json_object('Here you go: {"key_points": []} hope that helps') == {"key_points": []}
    nested = parse_json_object('prefix {"a": {"b": "}"}} suffix')
    assert nested == {"a": {"b": "}"}}
    with pytest.raises(Exception):
        parse_json_object("no json here")


def test_normalize_extraction_fills_gaps():
    payload = {
        "topics": [{"topic": "Budget review", "start": 12, "end": 30}],
        "key_points": [{"label": "Budget review", "points": ["Spend is up"]}, "Loose bullet"],
        "actions": [
            {"action": "Send the deck", "owner": "null", "deadline": "Friday"},
            "Book the room",
        ],
    }
    extraction = normalize_extraction(payload, window_start=10.0, window_end=40.0)
    assert extraction.topic_segments[0].topic_label == "Budget review"
    assert extraction.topic_segments[0].start_time == pytest.approx(12)
    assert extraction.key_points[0].points == ["Spend is up", "Loose bullet"]
    assert extraction.action_items[0].task == "Send the deck"
    assert extraction.action_items[0].owner is None  # "null" string degrades to None
    assert extraction.action_items[0].deadline == "Friday"
    assert extraction.action_items[1].owner is None


def test_normalize_extraction_survives_junk():
    assert normalize_extraction(None, 0, 10).topic_segments == []
    assert normalize_extraction({"topic_segments": "nope"}, 0, 10).topic_segments == []
    extraction = normalize_extraction({"topic_segments": [{"topic_label": ""}]}, 0, 10)
    assert extraction.topic_segments == []


# ---------------------------------------------------------------------------
# Meeting state
# ---------------------------------------------------------------------------


def test_state_merges_topics_and_dedupes_actions():
    state = MeetingState()
    state.merge(
        Extraction(
            topic_segments=[TopicSegment(0, 20, "Kickoff")],
            key_points=[KeyPointGroup("Kickoff", 0, 20, ["Agenda agreed"])],
            action_items=[ActionItem("Send the deck", owner=None, deadline=None)],
        )
    )
    state.merge(
        Extraction(
            topic_segments=[TopicSegment(20, 40, "Kickoff")],
            key_points=[KeyPointGroup("Kickoff", 20, 40, ["Agenda agreed", "Demo on Tuesday"])],
            action_items=[ActionItem("Send the deck", owner="Speaker 1", deadline="Friday")],
        )
    )
    assert len(state.topics) == 1
    assert state.topics[0].end_time == pytest.approx(40)
    assert state.key_points[0].points == ["Agenda agreed", "Demo on Tuesday"]
    assert len(state.action_items) == 1
    assert state.action_items[0].owner == "Speaker 1"
    assert state.action_items[0].deadline == "Friday"
    snapshot = state.snapshot()
    assert snapshot["stats"]["windows"] == 2
    assert "Send the deck" in state.export_markdown()


def test_state_merges_growing_prefix_action_items():
    state = MeetingState()
    state.merge(Extraction(action_items=[ActionItem("I will finish the topic")]))
    state.merge(Extraction(action_items=[ActionItem("I will finish the topic timeline by Friday")]))
    state.merge(Extraction(action_items=[ActionItem("Book the room", owner="Speaker 2")]))
    assert [item.task for item in state.action_items] == [
        "I will finish the topic timeline by Friday",
        "Book the room",
    ]


# ---------------------------------------------------------------------------
# Offline extractor (mock provider)
# ---------------------------------------------------------------------------


def test_mock_extractor_produces_topics_and_actions():
    settings = Settings(llm_provider="mock")

    async def run():
        extractor = Extractor(settings)
        try:
            return await extractor.extract(
                [
                    seg(0, 6, "Okay team, let's start the standup. First item is the backend."),
                    seg(6, 12, "The backend is running. I will profile the model this afternoon.", speaker=2),
                ],
                window_start=0.0,
                window_end=12.0,
            )
        finally:
            await extractor.aclose()

    extraction = __import__("asyncio").run(run())
    assert extraction.topic_segments, "mock extractor should label at least one topic"
    assert all(t.topic_label for t in extraction.topic_segments)
    assert extraction.action_items, "mock extractor should find the follow-up sentence"


def test_mock_summary_mentions_action_items():
    settings = Settings(llm_provider="mock")

    async def run():
        extractor = Extractor(settings)
        try:
            return await extractor.summarize(
                ["[0:00 - 0:05] Speaker 1: Hello team", "[0:05 - 0:09] Speaker 2: I will send the notes"],
                [{"task": "Send the notes", "owner": "Speaker 2", "deadline": None}],
            )
        finally:
            await extractor.aclose()

    result = __import__("asyncio").run(run())
    assert "Send the notes" in result["summary"]


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("NLP_LLM_PROVIDER", "mock")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("NLP_WINDOW_SECONDS", "20")
    monkeypatch.setenv("NLP_MIN_WINDOW_SECONDS", "4")
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_state_endpoints(client):
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["provider"] == "mock"
    state = client.get("/api/state").json()
    assert state["topics"] == []


def test_ingest_triggers_extraction_and_updates_state(client):
    lines = [
        {"speaker": 1, "start": "0:00:00.00", "end": "0:00:09.00",
         "text": "Let's start the standup. First topic is the speech recognition backend."},
        {"speaker": 2, "start": "0:00:09.00", "end": "0:00:19.00",
         "text": "The backend is running. I will profile the model this afternoon."},
        {"speaker": 1, "start": "0:00:19.00", "end": "0:00:26.00",
         "text": "Good. Please write the numbers in the document by Thursday."},
    ]
    response = client.post("/api/ingest", json={"lines": lines})
    assert response.status_code == 200
    assert response.json()["added"] == 3

    deadline = time.time() + 8
    state = {}
    while time.time() < deadline:
        state = client.get("/api/state").json()
        if state["stats"]["windows"] > 0 and state["topics"]:
            break
        time.sleep(0.1)

    assert state["stats"]["windows"] >= 1
    assert state["stats"]["segments"] == 3
    assert state["topics"], "expected at least one topic from the mock extractor"
    assert state["topics"][0]["label"]
    assert state["topics"][0]["start_label"]


def test_end_meeting_without_transcript_returns_conflict(client):
    client.post("/api/reset")
    response = client.post("/api/end")
    assert response.status_code == 409


def test_end_meeting_returns_summary_and_slack_preview(client):
    client.post(
        "/api/ingest",
        json={
            "lines": [
                {"speaker": 1, "start": "0:00:00.00", "end": "0:00:10.00",
                 "text": "We reviewed the pipeline. I will write the README tonight."},
            ]
        },
    )
    response = client.post("/api/end")
    assert response.status_code == 200
    payload = response.json()
    assert payload["event"] == "final"
    assert payload["slack"]["status"] == "skipped"
    assert "Action items" in payload["slack"]["preview"]
    assert payload["markdown"].startswith("# Meeting notes")

# ---------------------------------------------------------------------------
# Slack delivery
# ---------------------------------------------------------------------------


class _FakeSlackResponse:
    def __init__(self, status_code: int, text: str = "ok"):
        self.status_code = status_code
        self.text = text


def _fake_http_client(monkeypatch, status_code: int, captured: dict):
    from nlp_service import slack as slack_module

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json):
            captured["url"] = url
            captured["payload"] = json
            return _FakeSlackResponse(status_code)

    monkeypatch.setattr(slack_module.httpx, "AsyncClient", FakeClient)


def test_slack_posts_action_items_when_webhook_configured(monkeypatch):
    from nlp_service import slack as slack_module

    captured: dict = {}
    _fake_http_client(monkeypatch, 200, captured)
    settings = Settings(slack_webhook_url="https://hooks.slack.com/services/T000/B000/XXXX")
    result = __import__("asyncio").run(
        slack_module.post_to_slack(
            settings,
            "The team agreed on the pipeline.",
            [{"task": "Send the deck", "owner": "Speaker 2", "deadline": "Friday"}],
            ["Kickoff"],
        )
    )
    assert result["status"] == "sent"
    assert captured["url"].startswith("https://hooks.slack.com/")
    blocks = captured["payload"]["blocks"]
    assert any("Send the deck" in block.get("text", {}).get("text", "") for block in blocks)
    assert any(block["type"] == "header" for block in blocks)


def test_slack_reports_http_errors(monkeypatch):
    from nlp_service import slack as slack_module

    captured: dict = {}
    _fake_http_client(monkeypatch, 400, captured)
    settings = Settings(slack_webhook_url="https://hooks.slack.com/services/T000/B000/XXXX")
    result = __import__("asyncio").run(slack_module.post_to_slack(settings, "Summary", [], []))
    assert result["status"] == "error"
    assert "400" in result["message"]


# ---------------------------------------------------------------------------
# Cloud provider request shaping (httpx MockTransport, no network)
# ---------------------------------------------------------------------------


def test_anthropic_request_shape_and_parsing():
    import asyncio
    import json

    import httpx

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "text",
                        "text": '{"topic_segments":[{"start_time":1,"end_time":9,"topic_label":"Intro"}],'
                        '"key_points":[{"topic_label":"Intro","points":["Agenda agreed"]}],'
                        '"action_items":[{"task":"Send the deck","owner":null,"deadline":null}]}',
                    }
                ]
            },
        )

    settings = Settings(llm_provider="anthropic", anthropic_api_key="test-key")

    async def run():
        extractor = Extractor(settings, transport=httpx.MockTransport(handler))
        try:
            return await extractor.extract([seg(0, 9, "Hello team")], 0.0, 9.0)
        finally:
            await extractor.aclose()

    extraction = asyncio.run(run())
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["body"]["system"].startswith("You are a meeting/lecture analysis engine")
    assert "Hello team" in captured["body"]["messages"][0]["content"]
    assert extraction.topic_segments[0].topic_label == "Intro"
    assert extraction.action_items[0].owner is None


def test_openai_request_shape_and_parsing():
    import asyncio
    import json

    import httpx

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"topic_segments":[],"key_points":[],'
                            '"action_items":[{"task":"Book the room","owner":"Speaker 1","deadline":"Friday"}]}'
                        }
                    }
                ]
            },
        )

    settings = Settings(llm_provider="openai", openai_api_key="sk-test")

    async def run():
        extractor = Extractor(settings, transport=httpx.MockTransport(handler))
        try:
            return await extractor.extract([seg(0, 9, "Book the room by Friday")], 0.0, 9.0)
        finally:
            await extractor.aclose()

    extraction = asyncio.run(run())
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["authorization"] == "Bearer sk-test"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert extraction.action_items[0].owner == "Speaker 1"
    assert extraction.action_items[0].deadline == "Friday"


def test_provider_errors_surface_as_extraction_error():
    import asyncio

    import httpx

    from nlp_service.extractor import ExtractionError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    settings = Settings(llm_provider="anthropic", anthropic_api_key="nope")

    async def run():
        extractor = Extractor(settings, transport=httpx.MockTransport(handler))
        try:
            return await extractor.extract([seg(0, 5, "hello")], 0.0, 5.0)
        finally:
            await extractor.aclose()

    with pytest.raises(ExtractionError):
        asyncio.run(run())


def test_new_recording_clears_ended_meeting_but_reconnect_keeps_notes():
    import asyncio

    from nlp_service.bridge import TranscriptPipeline
    from nlp_service.state import SessionHub

    settings = Settings(
        llm_provider="mock", window_seconds=1000, min_window_seconds=1000, max_window_seconds=1000
    )
    extractor = Extractor(settings)

    def frame(text: str) -> dict:
        return {
            "lines": [
                {"speaker": 1, "text": text, "start": "0:00:00.00", "end": "0:00:05.00"},
            ]
        }

    async def scenario():
        hub = SessionHub()
        first = TranscriptPipeline(settings, hub, extractor)
        first.ingest_frame(frame("First meeting line"))
        assert len(hub.state.segments) == 1

        # Reconnect mid-meeting (never completed): notes must survive.
        reconnect = TranscriptPipeline(settings, hub, extractor)
        reconnect.ingest_frame(frame("Line after reconnect"))
        assert len(hub.state.segments) == 2

        # Deliberate end + new recording at t=0: clean meeting.
        hub.new_meeting_pending = True
        second = TranscriptPipeline(settings, hub, extractor)
        second.ingest_frame(frame("Second meeting line"))
        await asyncio.sleep(0)
        assert [s.text for s in hub.state.segments] == ["Second meeting line"]
        assert hub.new_meeting_pending is False

    asyncio.run(scenario())
    asyncio.run(extractor.aclose())
