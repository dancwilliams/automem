"""Keyword credit must describe the evidence, not the delivery channel.

``_graph_keyword_search`` normalizes its raw Cypher score by
``max_raw_score = 3 * len(keywords) + 3``, which budgets a tag hit for every
keyword plus a whole-phrase tag hit. A content-only match — the ordinary case
for an identifier, which rarely appears in tags — can therefore never exceed
4/6 = 0.667 no matter how exact it is.

Meanwhile a result the vector channel happened to surface is scored by content
token overlap, which reaches 1.0 for the same match. So one memory scored 1.0
or 0.667 for identical evidence, decided purely by which channel delivered it.

Observed in production: querying ``ntkrnlmp.exe`` ranked a *vector* result
scoring 0.500 on a partial token match above the memory containing both query
tokens verbatim, which was pinned at 0.667 because the keyword channel found it.

The fix takes ``max(channel_score, content_overlap)``, so the channel score is a
floor and never a ceiling.
"""

from __future__ import annotations

import pytest

from automem.utils.scoring import _compute_metadata_score

CONTENT = "bugcheck traced to ntkrnlmp.exe in the minidump"
TOKENS = ["ntkrnlmp", "exe"]


def _result(match_type, match_score, *, content=CONTENT):
    return {
        "id": "mem-1",
        "match_type": match_type,
        "match_score": match_score,
        "memory": {
            "id": "mem-1",
            "content": content,
            "tags": [],
            "importance": 0.5,
            "confidence": 0.9,
            "timestamp": "2026-06-11T00:00:00+00:00",
        },
    }


def _components(match_type, match_score, **kwargs):
    _final, components = _compute_metadata_score(
        _result(match_type, match_score, **kwargs), "ntkrnlmp.exe", TOKENS
    )
    return components


def test_keyword_channel_exact_match_is_not_capped_at_two_thirds() -> None:
    """The production case: raw 4/6 normalization must not cap a perfect match."""
    assert _components("keyword", 4 / 6)["keyword"] == pytest.approx(1.0)


def test_same_evidence_scores_the_same_through_either_channel() -> None:
    """The invariant. Identical memory, identical query, different delivery."""
    via_keyword = _components("keyword", 4 / 6)["keyword"]
    via_vector = _components("vector", 0.42)["keyword"]
    assert via_keyword == pytest.approx(via_vector)


def test_channel_score_is_a_floor_not_a_ceiling() -> None:
    """A tag-only keyword hit keeps its channel score when content has nothing.

    The Cypher also scores tag matches, so a result can legitimately arrive with
    a positive channel score and zero content overlap. That score must survive.
    """
    components = _components("keyword", 0.5, content="an unrelated note on networking")
    assert components["keyword"] == pytest.approx(0.5)


def test_partial_content_match_still_beaten_by_higher_channel_score() -> None:
    # 1 of 2 tokens in content = 0.5 overlap; channel reports a stronger 0.8.
    components = _components("keyword", 0.8, content="minidump mentions ntkrnlmp only")
    assert components["keyword"] == pytest.approx(0.8)


def test_keyword_component_never_exceeds_one() -> None:
    """Issue #190's contract holds: no channel may exceed the 0-1 range."""
    assert _components("keyword", 11.0)["keyword"] == pytest.approx(1.0)


def test_trending_results_are_unaffected() -> None:
    """Trending fires on empty/'*' queries, so there are no tokens to overlap."""
    result = _result("trending", 0.9)
    _final, components = _compute_metadata_score(result, "", [])
    assert components["keyword"] == pytest.approx(0.9)
