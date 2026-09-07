"""The context bonus must be weighted like every other scoring channel.

Regression for: a query containing an incidental style word ("lint") flipped
recall into style focus, and the resulting +0.70 bonus was added to the total
unweighted while SEARCH_WEIGHT_VECTOR caps at 0.35 - so style memories
outranked a perfect semantic match. See automem/api/recall.py
CONTEXT_STYLE_KEYWORDS and _build_context_profile.
"""

from automem.config import SEARCH_WEIGHT_CONTEXT, SEARCH_WEIGHT_VECTOR
from automem.utils.scoring import _compute_metadata_score

STYLE_PROFILE = {
    "priority_tags": {"style", "preferences"},
    "priority_types": {"Style", "Preference"},
    "priority_ids": set(),
    "priority_id_order": [],
    "priority_keywords": set(),
    "weights": {"tag": 0.45, "type": 0.25, "keyword": 0.2, "anchor": 0.9},
    # Style focus inferred from the word "lint" in the query; caller asked for nothing.
    "inferred_style_only": True,
}

EXPLICIT_PROFILE = {**STYLE_PROFILE, "inferred_style_only": False}


def _score(memory, result, profile=None):
    """The memory rides inside the result; returns (final, components)."""
    final, _ = _compute_metadata_score(
        {**result, "memory": memory},
        "lint contradiction corpus",
        ["lint", "contradiction", "corpus"],
        context_profile=profile,
    )
    return final


def test_context_bonus_is_weighted_not_raw():
    """A tag+type hit must contribute weight * 0.70, not a raw 0.70."""
    styled = {"content": "unrelated prose", "type": "Preference", "tags": ["style"]}
    result = {"match_type": "vector", "match_score": 0.0, "id": "s1"}

    with_ctx = _score(styled, result, STYLE_PROFILE)
    without_ctx = _score(styled, result, None)
    delta = with_ctx - without_ctx

    expected = SEARCH_WEIGHT_CONTEXT * 0.70
    assert abs(delta - expected) < 1e-6, f"context contributed {delta}, expected {expected}"
    assert delta < 0.70, "context bonus is still being added unweighted"


def test_style_bonus_cannot_outrank_a_better_semantic_match():
    """The bug: an off-topic style memory beating a strong on-topic one."""
    styled = {"content": "unrelated prose about naming", "type": "Preference", "tags": ["style"]}
    on_topic = {
        "content": "lint contradiction corpus sweep of the memory graph",
        "type": "Decision",
        "tags": ["automem"],
    }

    style_score = _score(styled, {"match_type": "vector", "match_score": 0.58, "id": "s1"},
                         STYLE_PROFILE)
    topic_score = _score(on_topic, {"match_type": "vector", "match_score": 0.71, "id": "t1"},
                         STYLE_PROFILE)

    assert topic_score > style_score, (
        f"style memory ({style_score:.4f}) still outranks the better semantic "
        f"match ({topic_score:.4f})"
    )


def test_weight_is_in_the_same_range_as_its_siblings():
    """It is a tiebreaker, not an override: a full bonus stays under a full vector match."""
    assert 0.0 <= SEARCH_WEIGHT_CONTEXT <= 1.0
    assert SEARCH_WEIGHT_CONTEXT * 0.70 < SEARCH_WEIGHT_VECTOR


def test_explicit_caller_context_keeps_full_strength():
    """context_tags/types the caller actually passed are not damped."""
    styled = {"content": "unrelated prose", "type": "Preference", "tags": ["style"]}
    result = {"match_type": "vector", "match_score": 0.0, "id": "s1"}

    delta = _score(styled, result, EXPLICIT_PROFILE) - _score(styled, result, None)
    assert abs(delta - 0.70) < 1e-6, f"explicit context was damped to {delta}, expected 0.70"
