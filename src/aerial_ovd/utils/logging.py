"""Structured logging: human-readable console output plus one JSON object per line in a file."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", None, None)))


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Anything passed via `extra=` becomes a structured field.
        for key, value in vars(record).items():
            if key not in _RESERVED and key not in payload and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that prints to stderr and, if `log_file` is given, appends JSON lines to it."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(getattr(h, "_aovd_console", False) for h in logger.handlers):
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        console._aovd_console = True  # type: ignore[attr-defined]
        logger.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file).resolve()
        already = any(getattr(h, "baseFilename", None) == str(log_path) for h in logger.handlers)
        if not already:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(JsonLineFormatter())
            logger.addHandler(fh)
    return logger
