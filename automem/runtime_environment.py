from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# Everything the logging module itself puts on a record. Derived from a real
# LogRecord rather than hard-coded so it stays correct across Python versions
# (3.12 added `taskName`, for instance).
_RESERVED_RECORD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {
    "asctime",
    "message",
    "taskName",
}


class StructuredFormatter(logging.Formatter):
    """Human-readable line, plus any ``extra={...}`` fields as trailing JSON.

    Standard logging silently discards ``extra`` unless the format string names
    every key. AutoMem emits structured events this way — ``recall_complete``
    carries query, latency, vector match counts, filter flags — so with a plain
    formatter the entire payload was computed on every request and thrown away,
    leaving a bare event name in the log. Appending the non-standard record
    attributes keeps ``docker logs`` readable while making the fields parseable
    downstream, e.g. in Loki:

        {container="memory-flask-api-1"} |= "recall_complete"
          | regexp "recall_complete \\| (?P<extra>\\{.*\\})$"
          | line_format "{{.extra}}" | json
    """

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            key: value for key, value in vars(record).items() if key not in _RESERVED_RECORD_ATTRS
        }
        if not extras:
            return base
        try:
            payload = json.dumps(extras, default=str, separators=(",", ":"), sort_keys=True)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return base
        return f"{base} | {payload}"


def configure_logging(*, level: int = logging.INFO) -> Any:
    root_handler = logging.StreamHandler(sys.stdout)
    root_handler.setFormatter(StructuredFormatter(_LOG_FORMAT))
    logging.basicConfig(level=level, handlers=[root_handler])
    logger = logging.getLogger("automem.api")

    for logger_name in ["werkzeug", "flask.app"]:
        framework_logger = logging.getLogger(logger_name)
        framework_logger.handlers.clear()
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(StructuredFormatter(_LOG_FORMAT))
        framework_logger.addHandler(stdout_handler)
        framework_logger.setLevel(level)
        # Without this, records also propagate to the root handler above and
        # every werkzeug/flask line is written to stdout twice.
        framework_logger.propagate = False

    return logger


def ensure_local_package_importable(*, file_path: str) -> None:
    try:
        import automem  # type: ignore  # noqa: F401
    except Exception:
        root = Path(file_path).resolve().parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
