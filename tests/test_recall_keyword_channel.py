"""Content keyword search must be a candidate channel, not a leftover-slot filler.

Reproduces the production miss: an exact identifier (error code, env var,
filename) that lives verbatim in exactly one memory is unrecallable, because
``graph_keyword_search`` only ran when vector search left slots empty --
``remaining_slots = max(0, per_query_limit - len(local_results))``. Once vector
over-fetch landed (4x the requested page), a corpus that fills a Qdrant page
makes that difference categorically 0, so the keyword channel never fires and a
trivial ``CONTAINS`` match never reaches the blended re-ranker.

These tests assert at the injection seam (``handle_recall`` receives
``graph_keyword_search`` by injection) because ``FakeGraph`` does not simulate
the ``UNWIND $keywords`` Cypher; that Cypher is covered by
``tests/test_keyword_score_normalization.py`` and by the live eval.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import app
import automem.api.recall as recall_module
from automem.utils.text import _extract_keywords
from tests.support.fake_graph import FakeGraph


def _result(memory_id, score, *, match_type="vector", content=None, tags=None):
    return {
        "id": memory_id,
        "score": score,
        "match_score": score,
        "match_type": match_type,
        "source": "qdrant" if match_type == "vector" else "graph",
        "memory": {
            "id": memory_id,
            "content": content or f"content for {memory_id}",
            "tags": tags or [],
            "target_score": score,
            "importance": 0.5,
            "confidence": 0.9,
            "type": "Context",
            "enriched": True,
            "timestamp": "2026-05-18T00:00:00+00:00",
        },
        "relations": [],
    }


class _KeywordSpy:
    """Stands in for ``_graph_keyword_search`` and records every invocation."""

    def __init__(self, results=None):
        self.calls: list[dict] = []
        self._results = list(results or [])

    def __call__(self, _graph, query_text, limit, seen_ids, **kwargs):
        self.calls.append(
            {
                "query": query_text,
                "limit": limit,
                "seen_ids": set(seen_ids),
                "kwargs": dict(kwargs),
            }
        )
        return [dict(res) for res in self._results if res["id"] not in seen_ids]

    @property
    def called(self) -> bool:
        return bool(self.calls)


# 20 decoys saturate the pool exactly the way production does: a limit-5 request
# over-fetches 4x, Qdrant fills the page, and the legacy gate computes
# max(0, 5 - 20) == 0.
_SATURATING_DECOYS = [_result(f"decoy-{i}", 0.95 - i * 0.01) for i in range(20)]

# The keyword-only candidate: vector never returns it, but its content holds the
# queried token verbatim. High score so re-ranking, not luck, decides placement.
_KEYWORD_HIT = _result(
    "keyword-hit",
    1.50,
    match_type="keyword",
    content="deploy notes: SUBSYS_E124105B replaced the old controller",
)


def _run_recall(
    *,
    keyword_search,
    vector_results=_SATURATING_DECOYS,
    url="/recall?query=SUBSYS_E124105B&limit=5&current_only=false",
):
    """Drive ``handle_recall`` with an injected keyword-search spy."""

    def _vector_search(_qdrant, _graph, _query, _embedding, limit, seen_ids, *_args):
        matches = [dict(res) for res in vector_results[:limit]]
        for match in matches:
            seen_ids.add(match["id"])
        return matches

    with app.app.test_request_context(url):
        response = recall_module.handle_recall(
            get_memory_graph=lambda: FakeGraph(),
            get_qdrant_client=lambda: object(),
            normalize_tag_list=lambda value: value if isinstance(value, list) else [],
            normalize_timestamp=lambda value: value,
            parse_time_expression=lambda _value: (None, None),
            extract_keywords=_extract_keywords,
            compute_metadata_score=lambda result, _query, _tokens, _context: (
                float((result.get("memory") or {}).get("target_score") or 0.0),
                {},
            ),
            result_passes_filters=lambda *_args, **_kwargs: True,
            graph_keyword_search=keyword_search,
            vector_search=_vector_search,
            vector_filter_only_tag_search=lambda *_args, **_kwargs: [],
            metadata_keyword_search=lambda *_args, **_kwargs: [],
            recall_max_limit=50,
            logger=SimpleNamespace(
                debug=lambda *_args, **_kwargs: None,
                info=lambda *_args, **_kwargs: None,
                exception=lambda *_args, **_kwargs: None,
            ),
        )

    return [result["id"] for result in response.get_json()["results"]]


def test_keyword_channel_runs_when_vector_saturates() -> None:
    """The bug: a full vector pool must not suppress the keyword channel."""
    spy = _KeywordSpy([_KEYWORD_HIT])

    ids = _run_recall(keyword_search=spy)

    assert spy.called, "keyword search never ran despite an exact-token query"
    assert "keyword-hit" in ids
    # Deduped against the vector pool, as before.
    assert spy.calls[0]["seen_ids"] == {res["id"] for res in _SATURATING_DECOYS}


def test_keyword_results_reranked_not_appended() -> None:
    """Keyword candidates go through _rank_local_results, not onto the tail."""
    spy = _KeywordSpy([_KEYWORD_HIT])

    ids = _run_recall(keyword_search=spy)

    assert ids[0] == "keyword-hit", "keyword candidate was not re-ranked with the pool"
    assert len(ids) == 5, "response size must still honor the requested limit"


def test_keyword_overfetch_off_switch_restores_legacy(monkeypatch) -> None:
    """RECALL_KEYWORD_OVERFETCH=0 reproduces the old fill-only semantics."""
    monkeypatch.setattr(recall_module, "RECALL_KEYWORD_OVERFETCH", 0, raising=False)

    saturated_spy = _KeywordSpy([_KEYWORD_HIT])
    ids = _run_recall(keyword_search=saturated_spy)
    assert not saturated_spy.called, "off-switch must not run the keyword channel"
    assert "keyword-hit" not in ids

    # Legacy behavior is fill-only, not off: with an underdelivering vector pool
    # the leftover slots are still handed to keyword search.
    underfilled_spy = _KeywordSpy([_KEYWORD_HIT])
    ids = _run_recall(keyword_search=underfilled_spy, vector_results=_SATURATING_DECOYS[:2])
    assert underfilled_spy.called
    assert underfilled_spy.calls[0]["limit"] == 3  # 5 requested - 2 vector hits
    assert "keyword-hit" in ids


@pytest.mark.parametrize("trending_query", ["", "*"])
def test_trending_queries_stay_on_the_legacy_fill_path(trending_query) -> None:
    """Empty/"*" queries route to trending, which must not be force-appended.

    graph_keyword_search turns an empty or "*" query into trending results
    rather than a content match. Those only ever belonged in slots vector left
    empty -- promoting them to an unconditional channel would staple trending
    memories onto every already-full tags-only page.
    """
    url = f"/recall?query={trending_query}&tags=cursor&limit=5&current_only=false"

    saturated_spy = _KeywordSpy([_KEYWORD_HIT])
    _run_recall(keyword_search=saturated_spy, url=url)
    assert not saturated_spy.called, "trending must not run against a full page"

    underfilled_spy = _KeywordSpy([_KEYWORD_HIT])
    _run_recall(keyword_search=underfilled_spy, vector_results=_SATURATING_DECOYS[:2], url=url)
    assert underfilled_spy.called, "trending fill is preserved when slots are free"
    assert underfilled_spy.calls[0]["limit"] == 3


@pytest.mark.parametrize(
    ("overfetch", "fetch_cap", "expected_limit"),
    [
        (2, 200, 10),  # default: 5 * 2, well under the cap
        (8, 200, 40),  # multiplier scales the budget
        (2, 3, 5),  # cap never reduces the budget below the requested page
    ],
)
def test_keyword_budget_capped(monkeypatch, overfetch, fetch_cap, expected_limit) -> None:
    monkeypatch.setattr(recall_module, "RECALL_KEYWORD_OVERFETCH", overfetch, raising=False)
    monkeypatch.setattr(recall_module, "RECALL_VECTOR_FETCH_CAP", fetch_cap)

    spy = _KeywordSpy([_KEYWORD_HIT])
    _run_recall(keyword_search=spy)

    assert spy.called
    assert spy.calls[0]["limit"] == expected_limit
