"""Pure injection-hook logic — extracted for testability.

The most correctness-critical path in NLA: if injection fails or hits the wrong
position, the model sees the literal ㊗ character and outputs Chinese. This
function is the one place that must be right, so it's pure and unit-testable.
"""

import torch


def inject_at_marked_positions(
    input_ids: torch.Tensor,
    embeddings: torch.Tensor,
    vectors: torch.Tensor,
    inj_id: int,
    left_id: int,
    right_id: int,
    seq_slice: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Overwrite embedding rows at injection-marker positions with activation vectors.

    input_ids: [B, S] — or [1, T_packed] for thd layout. The FULL token stream
        (broadcast across TP ranks — identical everywhere).
    embeddings: [B, S, d] (unsharded) or [B, S_local, d] (seq_slice set). The
        embedding layer output. Cloned; original unchanged.
    vectors: [N, d] — activation vectors in microbatch order. N = number of
        injection sites expected GLOBALLY. Must equal the count of valid matches
        found in the FULL input_ids (regardless of seq_slice).
    inj_id, left_id, right_id: the injection token + its canonical neighbors.
    seq_slice: (start, end) if embeddings holds only positions [start:end) of
        the sequence dim. For Megatron with --sequence-parallel: each TP rank's
        LanguageModelEmbedding output covers [tp_rank * S/TP : (tp_rank+1) * S/TP).
        The scan still runs over FULL input_ids (count + vec_idx are global),
        writes skip positions outside the slice.

    A match is valid iff input_ids[b, p] == inj_id AND input_ids[b, p-1] == left_id
    AND input_ids[b, p+1] == right_id. The neighbor check rejects false positives
    from ㊗ appearing in response text (user pasted it, multi-turn context).

    Raises:
        AssertionError if GLOBAL count of valid matches != vectors.shape[0] —
        means prompt template drift, tokenizer version mismatch, or data corruption.
        Fires identically on every TP rank (scan is over full input_ids).
    """
    seq_len = input_ids.shape[-1]
    if seq_slice is None:
        start, end = 0, seq_len
        assert input_ids.shape == embeddings.shape[:-1], (
            f"input_ids {tuple(input_ids.shape)} and embeddings "
            f"{tuple(embeddings.shape[:-1])} batch dims must match"
        )
    else:
        start, end = seq_slice
        assert input_ids.shape[0] == embeddings.shape[0], (
            f"batch dim mismatch: input_ids {input_ids.shape[0]}, "
            f"embeddings {embeddings.shape[0]}"
        )
        assert embeddings.shape[1] == end - start, (
            f"seq_slice={seq_slice} spans {end - start} positions but "
            f"embeddings seq dim is {embeddings.shape[1]}. SP shard layout "
            f"mismatch — check tp_rank/tp_size computation."
        )
    assert vectors.ndim == 2 and vectors.shape[1] == embeddings.shape[-1], (
        f"vectors must be [N, d_model], got {tuple(vectors.shape)}, "
        f"d_model={embeddings.shape[-1]}"
    )
    out = embeddings.clone()
    vectors = vectors.to(out.device, out.dtype)
    matches = (input_ids == inj_id).nonzero()  # [M, 2] — (batch_idx, seq_idx), row-major sorted
    vec_idx = 0
    for b, p in matches.tolist():
        if p == 0 or p == seq_len - 1:
            continue
        if input_ids[b, p - 1] != left_id or input_ids[b, p + 1] != right_id:
            continue
        if start <= p < end:
            out[b, p - start] = vectors[vec_idx]
        vec_idx += 1
    expected = vectors.shape[0]
    if vec_idx != expected:
        msg = (
            f"found {vec_idx} injection sites with correct neighbors, expected {expected}. "
            f"Check prompt template drift, tokenizer version, cp accidentally >1, "
            f"or (RL) rollout samples with multimodal_train_inputs=None skipped in concat."
        )
        # Under PP, this hook only runs on stage 0. Bare assert leaves stage 1
        # hanging on P2P recv → 10min NCCL timeout with no error. Abort the
        # whole world so the real error surfaces.
        if torch.distributed.is_initialized():
            print(f"[inject_at_marked_positions] FATAL: {msg}", flush=True)
            torch.distributed.destroy_process_group()
        raise RuntimeError(msg)
    return out


def marker_well_formed(prompt_ids, inj_id: int, left_id: int, right_id: int) -> bool:
    """Per-rollout precondition for a correct Karvonen injection (HF path).

    True iff `prompt_ids` (a list/sequence of token ids) contains EXACTLY ONE
    `inj_id` marker and that marker has its canonical left/right neighbors — i.e.
    the same validity test `inject_at_marked_positions` / `karvonen_inject_in_residual`
    apply per position. If this holds, the hook overwrites the marker's residual at
    the right place; if it doesn't, injection lands wrong (or the hook's count assert
    fires).

    This is a MECHANISM check, not an output-text proxy like CJK fraction: it can't
    be eroded by RL shifting the model's output distribution (the failure mode where
    a failed injection stops producing CJK once the policy has learned to avoid it).
    Cheap (pure token scan, no forward), so it's run per rollout to mask out any
    rollout whose marker drifted (tokenizer/template mismatch, marker echoed into the
    response and re-tokenized, etc.) before it can corrupt the AV/AR updates.
    """
    n = len(prompt_ids)
    positions = [i for i, t in enumerate(prompt_ids) if t == inj_id]
    if len(positions) != 1:
        return False
    p = positions[0]
    if p == 0 or p == n - 1:
        return False
    return prompt_ids[p - 1] == left_id and prompt_ids[p + 1] == right_id


def karvonen_inject_in_residual(
    input_ids: torch.Tensor,
    resid: torch.Tensor,
    vectors: torch.Tensor,
    inj_id: int,
    left_id: int,
    right_id: int,
) -> torch.Tensor:
    """ADD-norm-matched injection per Karvonen et al. 2025 (Activation Oracles, eq. 1).

    For each marker position p: h'_p = h_p + ||h_p|| * v / ||v||.

    Caller responsibility: register this hook on the OUTPUT of the second
    transformer layer (i.e. `model.model.layers[1].register_forward_hook(...)`),
    so the residual entering layer 2 is the modified one. Vectors should be
    RAW (no injection_scale normalization) — this function does its own norm
    match against the current residual.
    """
    seq_len = input_ids.shape[-1]
    assert input_ids.shape == resid.shape[:-1], (
        f"input_ids {tuple(input_ids.shape)} and resid {tuple(resid.shape[:-1])} batch dims must match"
    )
    assert vectors.ndim == 2 and vectors.shape[1] == resid.shape[-1], (
        f"vectors must be [N, d_model], got {tuple(vectors.shape)}, d_model={resid.shape[-1]}"
    )
    out = resid.clone()
    vectors = vectors.to(out.device, out.dtype)
    matches = (input_ids == inj_id).nonzero()  # [M, 2] (batch, seq), row-major sorted
    vec_idx = 0
    for b, p in matches.tolist():
        if p == 0 or p == seq_len - 1:
            continue
        if input_ids[b, p - 1] != left_id or input_ids[b, p + 1] != right_id:
            continue
        # Clone the slice before reading — otherwise out[b, p] is a VIEW into
        # `out`'s storage and the in-place write below modifies the same memory
        # the autograd graph references → "modified by inplace op" RuntimeError
        # at backward time.
        h_p = out[b, p].clone()
        v_unit = vectors[vec_idx] / (vectors[vec_idx].norm() + 1e-9)
        out[b, p] = h_p + h_p.norm() * v_unit
        vec_idx += 1
    expected = vectors.shape[0]
    if vec_idx != expected:
        msg = (
            f"Karvonen inject: found {vec_idx} marker sites with correct neighbors, "
            f"expected {expected}. Same diagnosis path as inject_at_marked_positions."
        )
        if torch.distributed.is_initialized():
            print(f"[karvonen_inject_in_residual] FATAL: {msg}", flush=True)
            torch.distributed.destroy_process_group()
        raise RuntimeError(msg)
    return out
