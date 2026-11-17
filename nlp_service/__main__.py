"""Entry point: `python -m nlp_service` starts the FastAPI service with uvicorn."""

from __future__ import annotations

import os

import uvicorn

from .app import app


def main() -> None:
    host = os.environ.get("NLP_HOST", "0.0.0.0")
    port = int(os.environ.get("NLP_PORT", "8001"))
    uvicorn.run(app, host=host, port=port, log_level=os.environ.get("NLP_LOG_LEVEL", "info").lower())


if __name__ == "__main__":
    main()