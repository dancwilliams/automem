"""The content tokenizer must agree with the query tokenizer on identifiers.

``_extract_keywords`` builds query tokens with ``[A-Za-z0-9_\\-]+``, so it keeps
underscores and hyphens together: ``CONSOLIDATION_BASE_DECAY_RATE`` stays one
token. ``_compute_metadata_score`` then re-tokenized the memory's *content* with
``\\b[a-z0-9]+\\b``. The two sets could never intersect on an identifier, so a
memory containing the queried identifier verbatim scored
``keyword_component = 0.0`` — no credit at all for an exact match.

Underscores are the worse half. ``_`` is a word character, so there is no ``\\b``
boundary anywhere inside ``consolidation_base_decay_rate`` — the old pattern
matched *nothing* there, neither the joined identifier nor its parts. Querying
"consolidation" could not match that content either. Hyphens at least split.

This is not theoretical. Recall over-fetches vector candidates 4x, so on a real
corpus the identifier's memory is usually already *in* the vector pool (measured:
rank 187 of 200). The seen-set dedupe then correctly withholds it from the
keyword channel, and this scoring path is the only place its exact match could
have counted. It counted zero, which held exact-identifier R@5 at 0.50.

Sub-token matching is preserved: querying "consolidation" must still match
content holding ``consolidation_base_decay_rate``. The content token set is a
superset, never a replacement.
"""

from __future__ import annotations

import pytest

from automem.utils.scoring import _compute_metadata_score
from automem.utils.text import _extract_keywords


def _vector_result(content, *, memory_id="mem-1"):
    """A vector-channel result: keyword credit comes only from content tokens."""
    return {
        "id": memory_id,
        "match_type": "vector",
        "match_score": 0.5,
        "memory": {
            "id": memory_id,
            "content": content,
            "tags": [],
            "importance": 0.5,
            "timestamp": "2026-06-11T00:00:00+00:00",
        },
    }


def _keyword_component(query, content):
    tokens = _extract_keywords(query.lower())
    assert tokens, "fixture query must extract at least one token"
    _final, components = _compute_metadata_score(_vector_result(content), query, tokens)
    return components["keyword"]


@pytest.mark.parametrize(
    ("query", "content"),
    [
        # Underscored identifiers — env vars, config keys, Python attributes.
        ("CONSOLIDATION_BASE_DECAY_RATE", "set CONSOLIDATION_BASE_DECAY_RATE to 0.01 in compose"),
        ("BWS_PROJECT_ID", "the BWS_PROJECT_ID lives in the bitwarden secrets project"),
        ("max_chars", "truncated at max_chars before the summary ran"),
        ("audio_transcription", "the audio_transcription pipeline stalled overnight"),
        # Hyphenated identifiers — tailwind classes, image tags, slugs.
        ("bg-green-800", "the badge uses bg-green-800 for the healthy state"),
        ("intel-arc-a380", "the intel-arc-a380 is shared between jellyfin and frigate"),
    ],
)
def test_identifier_in_content_earns_keyword_credit(query, content) -> None:
    """A verbatim identifier match must not score zero."""
    assert _keyword_component(query, content) == pytest.approx(1.0)


def test_absent_identifier_still_scores_zero() -> None:
    """The fix must not hand out credit for identifiers that are not there."""
    assert _keyword_component("BWS_PROJECT_ID", "an unrelated note about docker networking") == 0.0


@pytest.mark.parametrize(
    ("query", "content"),
    [
        # Hyphens were already a word boundary, so these matched before and must
        # keep matching: the content token set gains the joined form, it never
        # loses the split forms.
        ("green badge", "the badge uses bg-green-800 for the healthy state"),
        # Dotted names were never broken either: both tokenizers split on '.'.
        ("ntkrnlmp.exe", "bugcheck traced to ntkrnlmp.exe in the minidump"),
    ],
)
def test_existing_sub_token_matching_is_preserved(query, content) -> None:
    assert _keyword_component(query, content) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("query", "content"),
    [
        ("consolidation decay", "set consolidation_base_decay_rate to 0.01"),
        ("transcription", "the audio_transcription pipeline stalled overnight"),
    ],
)
def test_underscore_sub_tokens_become_matchable(query, content) -> None:
    """Newly gained, not merely preserved.

    ``_`` is a word character, so ``\\b[a-z0-9]+\\b`` found no boundary inside an
    underscored identifier and contributed no tokens whatsoever. Querying a word
    that appears only inside such an identifier used to score zero.
    """
    assert _keyword_component(query, content) == pytest.approx(1.0)


def test_partial_query_coverage_scores_proportionally() -> None:
    """Unmatched query tokens must still dilute the score."""
    # 'max_chars' hits, 'kangaroo' does not -> 1 of 2 tokens.
    assert _keyword_component("max_chars kangaroo", "truncated at max_chars") == pytest.approx(0.5)
