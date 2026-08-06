"""Ollama evicts idle models; the embedding provider must be able to say otherwise.

Ollama unloads a model 5 minutes after its last request and reloads it on the
next one. For a chat model that reads as a warm-up. For an *embedding* model on
the recall path it is a per-request cliff: measured against a 4GB
qwen3-embedding:0.6b, a cold call costs ~3.3s versus ~0.10s warm.

Recall traffic that is bursty and hours apart therefore pays the reload on
nearly every query — the first recall of every session eats 3+ seconds. Sending
``keep_alive`` on the request keeps that decision in AutoMem rather than
requiring OLLAMA_KEEP_ALIVE on the Ollama host, which would change behavior for
every other consumer sharing that GPU.

Omitting the field must stay byte-identical to the previous behavior, since the
value trades VRAM residency for latency and that is a deployment choice.
"""

from __future__ import annotations

import pytest

from automem.embedding.ollama import OllamaEmbeddingProvider


class _RecordingSession:
    def __init__(self, dimension=4):
        self.payloads = []
        self._dimension = dimension

    def post(self, url, json=None, timeout=None):  # noqa: A002 - mirrors requests' signature
        self.payloads.append(json)
        return _Response({"embedding": [0.1] * self._dimension})


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _provider(**kwargs):
    provider = OllamaEmbeddingProvider(
        base_url="http://ollama:11434/", model="qwen3-embedding:0.6b", dimension=4, **kwargs
    )
    provider.session = _RecordingSession()
    return provider


def test_keep_alive_is_sent_when_configured() -> None:
    provider = _provider(keep_alive="1h")

    provider.generate_embedding("BWS_PROJECT_ID")

    assert provider.session.payloads[0]["keep_alive"] == "1h"


def test_keep_alive_absent_by_default() -> None:
    """No configuration must mean no field — Ollama's own default applies."""
    provider = _provider()

    provider.generate_embedding("BWS_PROJECT_ID")

    assert "keep_alive" not in provider.session.payloads[0]


@pytest.mark.parametrize("blank", ["", None])
def test_blank_keep_alive_is_treated_as_unset(blank) -> None:
    """An unset env var arrives as "" — that must not send keep_alive: ""."""
    provider = _provider(keep_alive=blank)

    provider.generate_embedding("BWS_PROJECT_ID")

    assert "keep_alive" not in provider.session.payloads[0]


def test_indefinite_residency_is_expressible() -> None:
    provider = _provider(keep_alive="-1")

    provider.generate_embedding("BWS_PROJECT_ID")

    assert provider.session.payloads[0]["keep_alive"] == "-1"


def test_payload_still_carries_model_and_prompt() -> None:
    """The addition must not disturb the existing request shape."""
    provider = _provider(keep_alive="30m")

    provider.generate_embedding("hello")

    payload = provider.session.payloads[0]
    assert payload["model"] == "qwen3-embedding:0.6b"
    assert payload["prompt"] == "hello"


def test_batch_embeddings_also_keep_the_model_resident() -> None:
    """Batch paths matter most: a re-embed run should not let the model evict."""
    provider = _provider(keep_alive="1h")

    provider.generate_embeddings_batch(["one", "two", "three"])

    assert len(provider.session.payloads) == 3
    assert all(p["keep_alive"] == "1h" for p in provider.session.payloads)
