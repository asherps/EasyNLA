"""AR (reconstructor) forward → prediction vector. Shared by SFT + both RL trainers."""

import torch

from nla.schema import normalize_activation


def critic_predict(critic, input_ids, attention_mask, mse_scale_f):
    """pred = value_head(normalize(backbone_last_hidden, mse_scale)) at the suffix anchor.

    Normalising the backbone-last-hidden BEFORE the value_head bounds the head's
    input norm to a fixed scale (mse_scale), so a tiny weight update can't blow
    up the output norm by 100× — the path that NaN'd AR-SFT before. At identity
    init this agrees with the paper's direct value_head(backbone_last) after the
    loss's final normalize, so it's backward-compatible with AR-SFT checkpoints.

    Extracts at the LAST real token (suffix-anchored). Returns [B, d_model] fp32.
    Caller manages grad/no_grad — this does not toggle.
    """
    cout = critic(input_ids=input_ids, attention_mask=attention_mask)
    backbone_last = cout.backbone_last_hidden  # [B, T, D]
    if attention_mask is not None:
        last_idx = attention_mask.sum(dim=1) - 1
    else:
        last_idx = torch.full(
            (input_ids.shape[0],), input_ids.shape[1] - 1, device=input_ids.device,
        )
    bs = input_ids.shape[0]
    last_h = backbone_last[torch.arange(bs, device=input_ids.device), last_idx].float()
    last_h_norm = normalize_activation(last_h, mse_scale_f)
    # The value_head is deliberately fp32 (see NLACriticModel.from_pretrained);
    # keep its matmul fp32 even when the caller runs the backbone under a bf16
    # autocast (fp32 full-FT mode) — autocast would silently demote it.
    with torch.autocast(device_type=last_h_norm.device.type, enabled=False):
        return critic.value_head(last_h_norm.to(critic.value_head.weight.dtype)).float()
