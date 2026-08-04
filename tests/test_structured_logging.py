"""Structured ``extra={...}`` fields must survive to the log output.

Python's logging silently drops ``extra`` unless the format string names every
key. AutoMem emits its telemetry that way — ``recall_complete`` carries query,
latency, vector match counts and filter flags — so with a plain formatter the
whole payload was computed on every single request and discarded, leaving only
the bare event name in the log. The instrumentation existed and was unusable.

That failure mode is silent by construction: nothing errors, the log line just
says less than you think it does. Hence a test.
"""

from __future__ import annotations

import json
import logging

from automem.runtime_environment import StructuredFormatter


def _render(msg, **extra):
    record = logging.LogRecord(
        name="automem.api",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return StructuredFormatter("%(levelname)s | %(name)s | %(message)s").format(record)


def _payload(line):
    return json.loads(line.split(" | ", 3)[-1])


def test_extra_fields_are_emitted() -> None:
    line = _render("recall_complete", query="BWS_PROJECT_ID", latency_ms=141.2, keyword_results=2)

    assert line.startswith("INFO | automem.api | recall_complete | ")
    assert _payload(line) == {
        "keyword_results": 2,
        "latency_ms": 141.2,
        "query": "BWS_PROJECT_ID",
    }


def test_records_without_extras_are_unchanged() -> None:
    """Ordinary log lines must not grow a trailing separator or empty object."""
    assert _render("plain message") == "INFO | automem.api | plain message"


def test_reserved_attributes_are_not_leaked() -> None:
    """Only caller-supplied fields appear — not lineno, pathname, msg, etc."""
    payload = _payload(_render("recall_complete", query="x"))

    assert set(payload) == {"query"}


def test_unserializable_values_do_not_break_logging() -> None:
    """A bad value must degrade to a plain line, never raise from the log call."""
    line = _render("recall_complete", weird=object())

    assert line.startswith("INFO | automem.api | recall_complete")
    assert "weird" in line  # coerced via default=str rather than dropped


def test_payload_is_single_line() -> None:
    """Loki splits on newlines; a multi-line payload would fragment the event."""
    line = _render("recall_complete", query="line one\nline two")

    assert "\n" not in line
