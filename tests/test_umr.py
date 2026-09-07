import math

import pytest

from denomark.baselines.umr.model import calculate_z_score, context_seed, score_umr_tokens


class FakeBitmap:
    def __init__(self, values):
        self.values = values

    def get_bit(self, x, y):
        return self.values.get((x, y), False)


def test_calculate_z_score_matches_umr_bernoulli_null():
    assert calculate_z_score(60, 100) == pytest.approx(2.0)
    assert calculate_z_score(50, 100) == pytest.approx(0.0)
    assert calculate_z_score(0, 0) == 0.0
    expected = (7 - 10 * 0.25) / math.sqrt(10 * 0.25 * 0.75)
    assert calculate_z_score(7, 10, gamma=0.25) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("hits", "total", "gamma"),
    [(-1, 10, 0.5), (11, 10, 0.5), (1, -1, 0.5), (1, 10, 0.0), (1, 10, 1.0)],
)
def test_calculate_z_score_rejects_invalid_inputs(hits, total, gamma):
    with pytest.raises(ValueError):
        calculate_z_score(hits, total, gamma)


def test_context_seed_matches_upstream_hash_construction():
    assert context_seed(123, 42) == context_seed(123, 42)
    assert context_seed(123, 42) != context_seed(124, 42)


def test_score_umr_tokens_counts_bitmap_hits_and_recovers_votes():
    previous = 7
    ids = [8, 9, 10]
    targets = []
    prev = previous
    for current in ids:
        targets.append((prev, current, context_seed(prev, 42) % 4))
        prev = current
    values = {}
    for prev, current, bit_index in targets:
        target = "1001"[bit_index]
        values[(prev, current)] = target == "1"

    result = score_umr_tokens(ids, FakeBitmap(values), previous_token=previous)

    assert result["total_gen_tokens"] == 3
    assert result["token_hit_count"] == sum(values.values())
    assert result["z_score"] == pytest.approx(
        calculate_z_score(result["token_hit_count"], result["total_gen_tokens"])
    )
    assert len(result["votes"]) == 4
    assert result["recovered_str"] == "".join(
        "?" if sum(votes) == 0 else ("1" if votes[1] > votes[0] else "0")
        for votes in result["votes"]
    )


def test_score_without_previous_token_uses_first_id_as_context():
    result = score_umr_tokens([1, 2], FakeBitmap({(1, 2): True}))
    assert result["total_gen_tokens"] == 1
    assert result["token_hit_count"] == 1
