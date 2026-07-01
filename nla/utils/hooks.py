"""Karvonen activation-injection hook for the AV actor. Shared by SFT/RL/evals.

Registers (1) an embedding forward-hook that stashes the current input_ids and
(2) a forward-hook on transformer block `layer_idx` that norm-match-injects the
activation in `vectors_ref[0]` at the marker token (see
nla.injection.karvonen_inject_in_residual). No-op when seq_len < 2 (the
autoregressive cache steps after a rollout's prefill) or when no marker is
present. Device-aligned so it also works under device_map="auto".
"""

from nla.injection import karvonen_inject_in_residual


def register_karvonen_hook(model, vectors_ref, inj_id, left_id, right_id, layer_idx=1):
    state = {"input_ids": None}

    def embed_hook(module, args, kwargs, output):
        ids = kwargs.get("input") if kwargs else None
        if ids is None and args:
            ids = args[0]
        state["input_ids"] = ids
        return output

    def layer_hook(module, args, output):
        if isinstance(output, tuple):
            resid, *rest = output
        else:
            resid, rest = output, None
        input_ids = state["input_ids"]
        if input_ids is None or resid.shape[1] < 2:
            return output
        v = vectors_ref[0]
        if v is None or v.shape[0] == 0:
            return output
        # device_map="auto": this layer may live on a different GPU than where
        # the caller staged input_ids / the vector. Align to the residual.
        ids = input_ids.to(resid.device)
        if (ids == inj_id).sum().item() == 0:
            return output
        injected = karvonen_inject_in_residual(
            ids, resid, v.to(resid.device), inj_id, left_id, right_id,
        )
        if rest is None:
            return injected
        return (injected, *rest)

    model.get_input_embeddings().register_forward_hook(embed_hook, with_kwargs=True)
    # PEFT-aware: unwrap to the raw CausalLM first, then let arch_adapters find
    # the decoder list — handles multimodal wrappers (Gemma-3 language_model)
    # and the GPT-2/Falcon `.transformer.h` shape, where the old
    # `while hasattr(.model)` walk crashed with AttributeError('layers').
    from nla.utils.arch_adapters import resolve_decoder_layers
    target = model.get_base_model() if hasattr(model, "peft_config") else model
    resolve_decoder_layers(target)[layer_idx].register_forward_hook(layer_hook)
