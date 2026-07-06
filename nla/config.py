"""Load NLA runtime config from a sidecar YAML.

The sidecar pins tokenizer-dependent constants (injection token ID,
neighbor IDs, prompt templates) that were fixed at dataset/model generation time.
Loading them here and asserting against the live tokenizer catches drift before
training starts, not after output goes to Chinese.

Schema + shared types/helpers: nla/schema.py
"""

import math
from dataclasses import dataclass
from typing import Any

import yaml

from nla.schema import SCALE_SQRT_D, compute_canonical_neighbors, resolve_target_scale, sidecar_path_for



@dataclass(frozen=True)
class NLAConfig:
    d_model: int
    injection_char: str
    injection_token_id: int
    injection_left_neighbor_id: int
    injection_right_neighbor_id: int
    actor_prompt_template: str
    critic_prompt_template: str | None
    critic_num_layers: int | None

    # Critic extraction: position = last token of the prompt. The prompt
    # template ends with a fixed suffix (e.g. "</text> <summary>") so the
    # last real token is always the extraction point — no scanning, just
    # index -1 after padding is stripped.
    #
    # critic_suffix_ids is for a ONE-TIME sanity check at dataset load:
    # assert prompt_tokens[-len(suffix):] == suffix. It's tokenize(suffix)[1:]
    # because the first token merges with the explanation's last char
    # (e.g. "detail." + "</" → ".</", so first suffix token is unstable).
    critic_suffix_ids: list[int] | None = None

    # The transformer layer the dataset's activations were extracted at
    # (sidecar `extraction.layer_index`). Consumers use it to default/validate
    # depth choices — e.g. AR-SFT's --ar-num-layers should be layer_index+1;
    # a mismatch silently trains a wrong-depth critic.
    extraction_layer_index: int | None = None

    # Normalization controls. Both are resolved from sidecar's raw values
    # (None | "sqrt_d_model" | float) into concrete float | None at load time.
    #
    # injection_scale: what L2-norm the vector is scaled to before injection
    #   into the actor's embedding. None = inject raw (preserve magnitude —
    #   layer-depth signal?). sqrt(d_model) = match token-embedding scale.
    #
    # mse_scale: what L2-norm BOTH pred and gold are scaled to before MSE.
    #   None = MSE on raw magnitudes (critic must learn scale too).
    #   sqrt(d_model) = direction-only MSE (critic learns direction only).
    #
    # These are INDEPENDENT — you can inject raw but still train the critic
    # with direction-only MSE, or normalize injection but have the critic
    # predict raw magnitudes.
    injection_scale: float | None = None
    mse_scale: float | None = None

    @property
    def sqrt_d(self) -> float:
        return math.sqrt(self.d_model)


def load_nla_config(sidecar_source: str, tokenizer) -> NLAConfig:
    """Load sidecar and verify against live tokenizer.

    `sidecar_source` may be a checkpoint dir (reads {dir}/nla_meta.yaml) or a
    parquet path (reads {path}.nla_meta.yaml). Slice syntax stripped.

    Asserts:
      - injection char tokenizes to expected ID (tokenizer version drift)
      - injection char is not UNK
      - canonical actor prompt produces exactly one injection token
      - neighbor IDs at inj_pos ± 1 match the sidecar
    """
    meta_path = sidecar_path_for(sidecar_source)
    meta = yaml.safe_load(meta_path.read_text())

    kind = meta["kind"]
    assert kind in ("nla_model", "nla_dataset"), f"unknown sidecar kind: {kind!r}"

    if kind == "nla_dataset":
        extraction = meta["extraction"]
        d_model = extraction["d_model"]
    else:
        d_model = meta["d_model"]
        extraction = meta.get("extraction", {})

    # Sidecar may write: null / "sqrt_d_model" / float. Resolve to concrete
    # float | None here so downstream never branches on string sentinels.
    #
    # injection_scale: NO DEFAULT from absent key — it's a training hyperparameter
    # that must be chosen explicitly. Absent → None → train_actor.py asserts.
    # Sidecar value (if present) is a default for RESUMING from a trained checkpoint.
    #
    # mse_scale: defaults to sqrt_d_model. It's loss-numerical-stability,
    # not a tuning knob — the default is almost always right.
    injection_scale = resolve_target_scale(extraction.get("injection_scale"), d_model)
    mse_scale = resolve_target_scale(extraction.get("mse_scale", SCALE_SQRT_D), d_model)

    t = meta["tokens"]
    templates = meta.get("prompt_templates", {})
    critic_meta = meta.get("critic") or {}
    # schema v1 wrote "num_hidden_layers" (clashed with HF's config.json key).
    # v2 writes "extraction_layer_index". Read both for back-compat.
    critic_k = critic_meta.get("extraction_layer_index", critic_meta.get("num_hidden_layers"))

    cfg = NLAConfig(
        d_model=d_model,
        injection_char=t["injection_char"],
        injection_token_id=t["injection_token_id"],
        injection_left_neighbor_id=t["injection_left_neighbor_id"],
        injection_right_neighbor_id=t["injection_right_neighbor_id"],
        actor_prompt_template=templates.get("av") or templates["actor"],
        critic_prompt_template=templates.get("ar") or templates.get("critic"),
        critic_num_layers=critic_k,
        extraction_layer_index=extraction.get("layer_index"),
        critic_suffix_ids=t.get("critic_suffix_ids"),
        injection_scale=injection_scale,
        mse_scale=mse_scale,
    )

    # encode(), not convert_tokens_to_ids(): byte-level BPE tokenizers (Qwen,
    # GPT-2) store the byte-string representation as the token key, not the
    # unicode char. convert_tokens_to_ids('㈎') → None; encode('㈎') → [149705].
    live_inj_ids = tokenizer.encode(cfg.injection_char, add_special_tokens=False)
    assert live_inj_ids == [cfg.injection_token_id], (
        f"tokenizer drift: {cfg.injection_char!r} → {live_inj_ids}, "
        f"sidecar says [{cfg.injection_token_id}]. "
        f"Multi-token means the char split — wrong tokenizer or vocab changed."
    )
    assert live_inj_ids[0] != tokenizer.unk_token_id, (
        f"{cfg.injection_char!r} maps to UNK — pick a different marker"
    )

    live_left, live_right = compute_canonical_neighbors(
        tokenizer, cfg.actor_prompt_template, cfg.injection_char, cfg.injection_token_id
    )
    assert live_left == cfg.injection_left_neighbor_id, (
        f"left neighbor drift: tokenizer gives {live_left}, "
        f"sidecar says {cfg.injection_left_neighbor_id}"
    )
    assert live_right == cfg.injection_right_neighbor_id, (
        f"right neighbor drift: tokenizer gives {live_right}, "
        f"sidecar says {cfg.injection_right_neighbor_id}"
    )

    return cfg


def verify_critic_suffix(tokens: list[int], suffix_ids: list[int], context: str = "") -> None:
    """Assert the tokenized critic prompt ends with the expected suffix.

    One-time check — do at dataset load (or first few samples), not per-forward.
    The suffix is tokenize(template_suffix)[1:] — first token dropped because
    it BPE-merges with the explanation's last char and is unstable. The tail
    (e.g. the IDs for 'text> <summary>') is stable.
    """
    n = len(suffix_ids)
    actual = tokens[-n:]
    assert actual == suffix_ids, (
        f"critic prompt suffix mismatch{' (' + context + ')' if context else ''}: "
        f"expected tokens[-{n}:] == {suffix_ids}, got {actual}. "
        f"Template drift or tokenizer version changed the suffix encoding."
    )


