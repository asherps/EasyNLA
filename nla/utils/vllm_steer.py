"""Karvonen activation injection for the vLLM (vllm-lens) path — shared by the
RL rollout (train_rl_vllm.rollout_batch_vllm) and the offline AV decoder used by
the counterfactual_evidence eval, so both inject IDENTICALLY.

The injection is a `vllm_lens.SteeringVector` applied at `injection_layer` (=1)
with `norm_match=True`: the activation is renormalized to the running residual
norm and ADDed at the marker position — the exact Karvonen formula
`h'_p = h_p + ||h_p|| · v/||v||` the AV was trained with. Requires the
vllm-lens norm-match patch (utils/patch_vllm_lens.py).
"""

from __future__ import annotations


def find_marker_pos(prompt_ids, inj_id, left_id=None, right_id=None):
    """The single injection-marker token position in the AV prompt.

    When left_id/right_id are given, a position only counts if its neighbors
    match the canonical template neighbors — the SAME validity test the HF
    injection path applies (nla.injection). Without it the two paths are
    asymmetric: a stray marker glyph with wrong neighbors (e.g. leaked into the
    source text) would be REJECTED by the HF hook but silently STEERED by vLLM.
    Asserts exactly one (valid) marker either way."""
    n = len(prompt_ids)
    if left_id is None and right_id is None:
        positions = [i for i, t in enumerate(prompt_ids) if t == inj_id]
    else:
        positions = [
            i for i, t in enumerate(prompt_ids)
            if t == inj_id and 0 < i < n - 1
            and (left_id is None or prompt_ids[i - 1] == left_id)
            and (right_id is None or prompt_ids[i + 1] == right_id)
        ]
    assert len(positions) == 1, (
        f"expected exactly 1 valid marker token (inj_id={inj_id}, "
        f"left={left_id}, right={right_id}) in prompt, got {len(positions)}"
    )
    return positions[0]


def read_reset_steer_count(model):
    """vLLM `apply_model` helper (runs IN each worker): read+reset the steering-apply
    counter the vllm-lens patch maintains, so the trainer can verify injection
    actually happened during a rollout (a distribution-invariant check vs. inferring
    it from CJK output). `model` is unused. Returns -1 if the counter isn't present
    (patch not applied) so the trainer degrades gracefully to the CJK/marker checks.

    MUST live in an importable module (not the `python -m ...`-run trainer, which is
    `__main__`): vLLM's collective_rpc pickles this by qualified name to ship it to
    the workers, and a `__main__` function can't be pickled by reference.
    """
    try:
        import vllm_lens._worker_ext as _wx
        fn = getattr(_wx, "get_and_reset_steer_count", None)
        return int(fn()) if fn is not None else -1
    except Exception:
        return -1


def build_steering_vector(activation, marker_pos, injection_layer=1):
    """One activation -> a vllm-lens SteeringVector injected at `marker_pos`.

    SHAPE MATTERS: activations must be 3-D [n_layers, n_positions, d]. vllm-lens
    only honors position_indices for 3-D tensors; a 2-D [n_layers, d] tensor
    takes the BROADCAST branch in _worker_ext.py (_apply_steering) and gets ADDed
    at EVERY token, silently ignoring position_indices.
    """
    from vllm_lens import SteeringVector

    return SteeringVector(
        activations=activation.view(1, 1, -1).cpu().float(),  # [1, 1, d]
        layer_indices=[injection_layer],
        scale=1.0,
        norm_match=True,
        position_indices=[marker_pos],
    )
