import torch

from denomark.baselines.umr.generate import DreamNativeRegretController, DreamUMRApplyCompat


class UnusedUpstream:
    pass


def test_clean_dream_native_controller_never_changes_tokens_or_eos():
    controller = DreamNativeRegretController(
        UnusedUpstream(),
        watermark=None,
        apply_watermark=None,
        mask_id=9,
        prompt_len=2,
        gen_length=3,
        steps=3,
    )
    initial = torch.tensor([[1, 2, 9, 9, 9]])
    assert torch.equal(controller(None, initial.clone(), None), initial)

    # Treat token 4 as an EOS stand-in. The clean/native controller must leave
    # it untouched; minimum length is enforced only after generation.
    transferred = torch.tensor([[1, 2, 4, 9, 9]])
    assert torch.equal(
        controller(0, transferred.clone(), torch.zeros((1, 5, 10))),
        transferred,
    )


def test_dream_umr_preserves_native_eos_and_cannot_introduce_it():
    calls = []

    def fake_apply(*, x, x0, logits, batch_index, index, watermark, mask_id):
        calls.append(True)
        x0[batch_index, index] = logits[batch_index, index].argmax()

    compat = DreamUMRApplyCompat(
        fake_apply,
        convert_gumbel_scores=False,
        preserve_token_ids={4},
        gumbel_temperature=0.0,
    )
    x = torch.tensor([[1, 9]])
    logits = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 10.0], [0.0] * 5]])

    native_eos = torch.tensor([[4, 9]])
    compat(x, native_eos, logits, 0, 0, object(), 9)
    assert native_eos[0, 0].item() == 4
    assert not calls

    native_regular = torch.tensor([[3, 9]])
    compat(x, native_regular, logits, 0, 0, object(), 9)
    assert native_regular[0, 0].item() == 3
    assert calls
    assert logits[0, 0, 4].item() == 10.0
