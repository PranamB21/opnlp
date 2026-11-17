"""Slack incoming-webhook delivery for the end-of-meeting summary."""

from __future__ import annotations

import logging
from typing import Sequence

import httpx

from .config import Settings

logger = logging.getLogger(__name__)

_MAX_BLOCK_TEXT = 2900  # Slack section blocks cap text at 3000 chars


def build_slack_blocks(
    summary: str,
    action_items: Sequence[dict],
    topics: Sequence[str] = (),
) -> list[dict]:
    """Build Block Kit payload: summary, topics, then action items."""
    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text", "text": "Meeting summary"}},
    ]
    if summary:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": summary[:_MAX_BLOCK_TEXT]}})
    else:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "_No summary was produced._"}})

    if topics:
        topic_text = " *Topics:* " + " · ".join(topics[:12])
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": topic_text[:2900]}]})

    blocks.append({"type": "divider"})

    if action_items:
        lines = []
        for item in action_items[:20]:
            owner = item.get("owner") or "unassigned"
            deadline = f" — due {item['deadline']}" if item.get("deadline") else ""
            lines.append(f"• {item.get('task', '').strip()} _({owner}{deadline})_")
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Action items*\n" + "\n".join(lines)[:2900]},
            }
        )
    else:
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": "*Action items*\n_None detected._"}}
        )
    return blocks


def render_slack_preview(summary: str, action_items: Sequence[dict], topics: Sequence[str] = ()) -> str:
    """Plain-text fallback used when Slack is not configured."""
    parts = ["Meeting summary", "", summary or "(no summary)"]
    if topics:
        parts += ["", "Topics: " + " · ".join(topics)]
    parts += ["", "Action items:"]
    if action_items:
        for item in action_items:
            owner = item.get("owner") or "unassigned"
            deadline = f" — due {item['deadline']}" if item.get("deadline") else ""
            parts.append(f"- {item.get('task', '')} ({owner}{deadline})")
    else:
        parts.append("- none")
    return "\n".join(parts)


async def post_to_slack(
    settings: Settings,
    summary: str,
    action_items: Sequence[dict],
    topics: Sequence[str] = (),
) -> dict:
    """Post the summary to Slack, or return a preview when no webhook is set."""
    preview = render_slack_preview(summary, action_items, topics)
    if not settings.slack_webhook_url:
        return {
            "status": "skipped",
            "message": "SLACK_WEBHOOK_URL is not configured; returning preview only.",
            "preview": preview,
        }

    payload = {
        "text": summary or "Meeting summary",
        "blocks": build_slack_blocks(summary, action_items, topics),
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(settings.slack_webhook_url, json=payload)
        if response.status_code >= 400:
            logger.warning("Slack webhook rejected payload: %s %s", response.status_code, response.text[:200])
            return {
                "status": "error",
                "message": f"Slack returned {response.status_code}: {response.text[:200]}",
                "preview": preview,
            }
    except httpx.HTTPError as exc:
        logger.warning("Slack webhook call failed: %s", exc)
        return {"status": "error", "message": f"Slack request failed: {exc}", "preview": preview}

    return {"status": "sent", "message": "Meeting summary posted to Slack.", "preview": preview}