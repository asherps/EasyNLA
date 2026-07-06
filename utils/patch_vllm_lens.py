"""Patch vllm-lens 1.1.0: (1) norm-match steering against the FULL residual
stream, (2) O(1) steering lookup on the offline path, and (3) a steering-apply
counter so the trainer can explicitly verify injection happened during a rollout
(see the "Explicit injection check (3)" hunks near the bottom).

Perf fix (2), added 2026-06-25: `_find_steering_configs` runs inside the
per-layer forward hook — once per running request, on every decoder layer,
every decode step — and unconditionally scans the entire `_steering_data` dict
(one entry per rollout) with a string `startswith`. On the offline LLM.generate
path (NLA RL rollout) every request has an exact `_steering_id`, so that scan
never matches and is pure waste, but it makes rollout throughput collapse
super-linearly with batch (measured ~400 tok/s @ 1024 rollouts → ~168 @ 2048 at
identical concurrency). Fix takes the O(1) `_steering_id` lookup first and skips
the scan. See the OLD_FIND/NEW_FIND hunk.

Original fix (1) — norm-match steering against the FULL residual stream:

Bug (vllm_lens/_worker_ext.py): vLLM's Qwen/Llama decoder layers return
``(hidden_states, residual)`` with the TRUE residual stream materialized as
``hidden_states + residual`` by the next layer's fused add-RMSNorm. The
steering hook wrote and — critically — **norm-matched** against ``output[0]``
(the partial delta) instead of the full stream. The capture path in the same
file already sums the tuple (``capture_src[0] + capture_src[1]``); steering
didn't.

Effect on NLA RL: the Karvonen injection ``h' = h + ‖h‖·v/‖v‖`` was applied
with ``‖delta‖`` instead of ``‖h_full‖`` — injecting the activation far too
weakly during vLLM rollouts (unit test showed ~50× in a synthetic case), while
the HF training forward injected at full magnitude. Same weights, different
policy ⇒ GRPO clip-frac ~40% from step 0, FVE stuck ≤ ~10%.

Fix: ``_apply_steering`` gains a ``norm_ref`` argument anchored to
``output[0] + output[1]`` for tuple layer outputs (write still goes to
``output[0]``, which lands in the stream). Also skips clone/sum work on layers
no steering config targets.

Idempotent — safe to run repeatedly (e.g. after rebuilding the venv):

    <vllm-lens-venv>/bin/python utils/patch_vllm_lens.py
"""

import importlib.util
import shutil
import sys
from pathlib import Path

OLD_APPLY_SIG = """def _apply_steering(
    configs: list[SteeringVector],
    layer_idx: int,
    target: torch.Tensor,
    start: int,
    end: int,
    abs_start: int,
) -> None:
    \"\"\"Apply all matching steering vectors to a token slice *in-place*.

    ``target`` is the (already-cloned) output tensor.  ``start``/``end``
    are batch-relative indices, ``abs_start`` is the absolute sequence
    position of the first token in ``target[start:end]``.
    \"\"\"
    n_tokens = end - start"""

NEW_APPLY_SIG = """def _apply_steering(
    configs: list[SteeringVector],
    layer_idx: int,
    target: torch.Tensor,
    start: int,
    end: int,
    abs_start: int,
    norm_ref: torch.Tensor | None = None,
) -> None:
    \"\"\"Apply all matching steering vectors to a token slice *in-place*.

    ``target`` is the (already-cloned) output tensor.  ``start``/``end``
    are batch-relative indices, ``abs_start`` is the absolute sequence
    position of the first token in ``target[start:end]``.

    ``norm_ref`` is the tensor whose per-position L2 norm anchors
    ``norm_match``. For models whose decoder layers return
    ``(hidden_states, residual)`` tuples (Qwen/Llama in vLLM), the TRUE
    residual stream is ``hidden_states + residual`` — norm-matching against
    ``hidden_states`` alone mis-scales the steering vector. Defaults to
    ``target`` for plain (non-tuple) layer outputs.
    \"\"\"
    if norm_ref is None:
        norm_ref = target
    n_tokens = end - start"""

OLD_BCAST = """            if cfg.norm_match:
                v = norm_match(target[start:end], v)
            target[start:end] = target[start:end] + v * cfg.scale"""

NEW_BCAST = """            if cfg.norm_match:
                v = norm_match(norm_ref[start:end], v)
            target[start:end] = target[start:end] + v * cfg.scale"""

OLD_POS = """                if cfg.norm_match:
                    v = norm_match(target[rel], v)
                target[rel] = target[rel] + v * cfg.scale"""

NEW_POS = """                if cfg.norm_match:
                    v = norm_match(norm_ref[rel], v)
                target[rel] = target[rel] + v * cfg.scale"""

OLD_HOOK = """    # --- Phase 2: apply steering ------------------------------------
    modified_output: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    if needs_steering:
        if isinstance(output, tuple):
            modified_output = (output[0].clone(), output[1])
            target = modified_output[0]
        else:
            modified_output = output.clone()
            target = modified_output"""

NEW_HOOK = """    # --- Phase 2: apply steering ------------------------------------
    modified_output: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    # Only do the clone/full-stream work when some config actually targets
    # THIS layer (the hook fires on every layer; configs usually target one).
    if needs_steering:
        needs_steering = any(
            layer_idx in cfg.layer_index_map
            for cfgs in per_req_steering
            for cfg in cfgs
        )
    if needs_steering:
        if isinstance(output, tuple):
            modified_output = (output[0].clone(), output[1])
            target = modified_output[0]
            # Decoder layers that return (hidden_states, residual) defer the
            # residual add to the next layer's fused add-RMSNorm: the TRUE
            # stream is output[0] + output[1]. Writing the steering vector
            # into output[0] lands it in the stream, but norm_match must be
            # anchored to the FULL stream — same convention as the capture
            # path below (capture_src[0] + capture_src[1]). Norm-matching
            # against output[0] alone mis-scales the injection (observed as
            # ~40% GRPO clip-frac at identical weights vs an HF forward).
            norm_ref = (
                output[0] + output[1] if output[1] is not None else output[0]
            )
        else:
            modified_output = output.clone()
            target = modified_output
            norm_ref = target"""

OLD_CALL = """            _apply_steering(
                per_req_steering[i], layer_idx, target, start, end, abs_start
            )"""

NEW_CALL = """            _apply_steering(
                per_req_steering[i], layer_idx, target, start, end, abs_start,
                norm_ref,
            )"""

# --- Perf fix: O(1) steering lookup on the offline path -------------------
# `_find_steering_configs` runs inside the per-layer forward hook, once per
# RUNNING request, on every one of the ~36 decoder layers, every decode step.
# It unconditionally scans the ENTIRE `_steering_data` dict (one entry per
# rollout in the batch) doing a string `startswith` — i.e. O(num_running ×
# num_total) Python ops per layer per step. On the offline LLM.generate path
# (NLA RL rollout) every request carries an exact `_steering_id`, so that scan
# NEVER matches and is pure waste — but it makes rollout throughput collapse
# super-linearly with batch (measured: ~400 tok/s at 1024 rollouts → ~168 at
# 2048, at identical concurrency). Fix: take the O(1) `_steering_id` dict
# lookup FIRST and return; only fall back to the prefix scan for the async
# path (no `_steering_id`). Behavior-preserving for both paths.
OLD_FIND = """    results: list[SteeringVector] = []
    for external_id, configs in extension._steering_data.items():
        if internal_req_id.startswith(f"{external_id}-"):
            results.extend(configs)
    # Offline path stores a lightweight string key in extra_args
    if extra_args:
        steering_id = extra_args.get("_steering_id")
        if steering_id and steering_id in extension._steering_data:
            results.extend(extension._steering_data[steering_id])
    return results"""

NEW_FIND = """    # Offline path (NLA RL rollout): exact O(1) lookup — taken FIRST so the
    # hot per-layer hook does NOT scan the whole _steering_data dict per
    # request (that scan is O(num_running x num_total) per layer per decode
    # step and collapses rollout throughput at large batch).
    if extra_args:
        steering_id = extra_args.get("_steering_id")
        if steering_id is not None:
            return list(extension._steering_data.get(steering_id, []))
    # Async path (no _steering_id): match by external-id prefix.
    results: list[SteeringVector] = []
    for external_id, configs in extension._steering_data.items():
        if internal_req_id.startswith(f"{external_id}-"):
            results.extend(configs)
    return results"""

# --- Explicit injection check (3): count steering-vector applications ----------
# added 2026-06-30. The NLA trainer needs an explicit, distribution-invariant
# way to verify the steering vector was actually written during a rollout —
# instead of inferring it from CJK garbage in the output (a symptom that RL
# erodes: once the policy learns to avoid CJK, a *failed* injection no longer
# shows up as CJK, so the heuristic silently false-negatives). We expose a
# per-worker counter that increments on every marker position-write; the trainer
# resets it before a rollout `generate` and reads it after (via LLM.apply_model)
# and compares to the number of rollouts. A module-level 1-element list lets the
# hot `_apply_steering` hook increment WITHOUT a `global` declaration.
#
# These two hunks transform the ALREADY-PATCHED state (their OLD = the NEW text
# of the hunks above), so they apply cleanly on top of a file patched before
# this fix existed — same idempotency contract as the rest of the file.
OLD_COUNT_GLOBAL = "def _find_steering_configs("
NEW_COUNT_GLOBAL = '''# NLA explicit injection check: number of steering-vector marker writes since the
# last reset. 1-elem list so the hot _apply_steering hook mutates it without a
# `global`. Read+reset per rollout from the trainer via LLM.apply_model.
_NLA_STEER_APPLY_COUNT = [0]


def get_and_reset_steer_count() -> int:
    """Return marker position-writes since the last call, then reset to 0."""
    c = _NLA_STEER_APPLY_COUNT[0]
    _NLA_STEER_APPLY_COUNT[0] = 0
    return c


def _find_steering_configs('''

# Anchor on the position-write line ALONE — it's byte-identical in the fresh and
# the fix-(1)-patched file (fix-(1) only changed the `norm_match(...)` line ABOVE
# it), and it's unique (the broadcast path writes `target[start:end]`). So this
# hunk is independent of the others and `main()`'s against-original-src check
# works whether or not fix-(1) is applied yet.
OLD_COUNT_INCR = "                target[rel] = target[rel] + v * cfg.scale"
NEW_COUNT_INCR = (
    "                target[rel] = target[rel] + v * cfg.scale\n"
    "                _NLA_STEER_APPLY_COUNT[0] += 1  # NLA: count this marker write"
)

HUNKS = [
    (OLD_APPLY_SIG, NEW_APPLY_SIG),
    (OLD_BCAST, NEW_BCAST),
    (OLD_POS, NEW_POS),
    (OLD_HOOK, NEW_HOOK),
    (OLD_CALL, NEW_CALL),
    (OLD_FIND, NEW_FIND),
    (OLD_COUNT_GLOBAL, NEW_COUNT_GLOBAL),
    (OLD_COUNT_INCR, NEW_COUNT_INCR),
]


def main() -> int:
    spec = importlib.util.find_spec("vllm_lens._worker_ext")
    assert spec and spec.origin, "vllm_lens not importable from this python"
    path = Path(spec.origin)
    src = path.read_text()

    # Per-hunk idempotency: a hunk whose NEW text is already present is skipped
    # (lets us add new hunks to an already-partially-patched file). A hunk whose
    # OLD text is missing AND whose NEW text is absent = version drift -> refuse.
    to_apply = []
    for i, (old, new) in enumerate(HUNKS):
        if new in src:
            continue
        if old not in src:
            print(f"[patch_vllm_lens] hunk {i} not found (neither OLD nor NEW) "
                  f"— vllm_lens version drift? Refusing to patch {path}")
            return 1
        to_apply.append((old, new))

    if not to_apply:
        print(f"[patch_vllm_lens] already patched (all {len(HUNKS)} hunks): {path}")
        return 0

    _orig = path.with_suffix(".py.orig")
    if not _orig.exists():   # keep the PRISTINE original across incremental patches
        shutil.copy2(path, _orig)
    for old, new in to_apply:
        src = src.replace(old, new, 1)
    path.write_text(src)
    pycache = path.parent / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache)
    print(f"[patch_vllm_lens] applied {len(to_apply)} hunk(s) to {path} "
          f"(backup: {path.name}.orig)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
