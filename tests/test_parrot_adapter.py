"""Check merged Parrot configuration and tracing without model weights."""

from types import SimpleNamespace

import pytest

from denomark.attacks.parrot import build_attacker


@pytest.mark.parametrize("count", [1, 4, 7, 10])
def test_parrot_configuration(count):
    class FakeParrot:
        def augment(self, *args, **kwargs):
            return ["candidate"] * kwargs["max_return_phrases"]

    au = SimpleNamespace(
        SParrot=FakeParrot,
        ParrotParaphraseConfig=SimpleNamespace,
        ParrotParaphrase=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    attacker = build_attacker(au, count)
    assert attacker.cfg.num_beams == count
    assert attacker.cfg.use_bigram_filter == (count > 1)
    if count > 1:
        assert len(attacker.parrot.augment(max_return_phrases=count)) == count
        assert attacker.candidate_counts == [count]
    attacker.reset_trace()
    assert attacker.candidate_counts == []
