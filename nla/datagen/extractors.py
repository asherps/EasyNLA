"""Activation extraction backends.

Stage 0 forwards a base model over a corpus and grabs hidden states at a
specified layer. `ActivationExtractor` is the pluggable interface — stage 0
code calls `extract()` with a list of texts and gets back per-text hidden
states + token IDs. GPU placement, batching, model parallelism, and choice
of inference engine are all the extractor's problem.

Swap via `--extractor-cls my.module.MyExtractor` at stage0 invocation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from nla.utils.arch_adapters import resolve_decoder_layers, resolve_text_config
from nla.datagen._common import load_tokenizer


class _CaptureComplete(Exception):
    """Sentinel raised inside the capture hook to abort the forward pass once the
    target layer's hidden state is in hand. Skips the remaining decoder layers and
    — crucially — the lm_head, whose [B, seq, vocab] logits tensor is what OOM'd at
    large batch (512·2048·151936·2 B ≈ 300 GiB for Qwen3). We never use logits."""


@dataclass
class ExtractionResult:
    hidden_states: torch.Tensor  # [seq_len, d_model], float32, CPU, unpadded
    token_ids: list[int]


class ActivationExtractor(ABC):
    """Submit a batch of texts, get back layer-K hidden states + token IDs.

    Subclasses own all batching/device/parallelism decisions. Callers pass
    the full task chunk and wait for results.

    Constructor contract: stage0 always passes `model_name` as a kwarg (set
    from `--base-model`). Custom extractors MUST accept `model_name` in
    __init__ — this is the provenance key written to the sidecar. Everything
    else comes via `--extractor-kwargs`.
    """

    d_model: int
    tokenizer: Any

    @abstractmethod
    def extract(self, texts: list[str], layer_index: int) -> list[ExtractionResult]: ...


class HFExtractor(ActivationExtractor):
    """Default extractor: HuggingFace transformers with a forward hook on the target layer.

    `device_map="auto"` handles multi-GPU model parallelism via accelerate —
    layers get sharded across GPUs transparently, no FSDP needed for
    inference-only.

    A forward hook on just the target layer avoids `output_hidden_states=True`,
    which stores every layer's activations and bloats memory by num_layers×.
    The hook captures the target layer's output for each sub-batch; we then
    slice out padding and return per-text unpadded CPU tensors.

    Assumes Llama-family architecture (`model.model.layers[K]` module path).
    Works for Qwen, Llama, Mistral, Gemma. GPT-2/NeoX/Falcon use different
    paths and will AttributeError loudly at hook registration.

    `layer_index=K` returns the output of the K-th decoder block (post-MLP,
    post-residual-add — the residual stream entering layer K+1). This matches
    HF's `hidden_states[K+1]` when `output_hidden_states=True` (their index 0
    is the embedding output).
    """

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_length: int = 2048,
        batch_size: int = 16,
        attn_implementation: str = "sdpa",
    ):
        self.tokenizer = load_tokenizer(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # LlamaTokenizerFast, GemmaTokenizerFast etc. default to left-padding
        # for generation. We slice [:seq_len] below — MUST be right-padded or
        # we silently return pad-position activations. Same for truncation_side
        # — left-truncation would mean token_ids[0] is NOT the doc start.
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        # sdpa (memory-efficient attention) avoids materializing the
        # [B, heads, seq, seq] eager-attention score tensor — at batch 512 that
        # alloc hit ~300 GiB and OOM'd. sdpa keeps attention O(B·seq) in memory
        # so large no-grad extraction batches fit on an H200.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        ).eval()
        self.d_model = resolve_text_config(self.model.config).hidden_size
        self.max_length = max_length
        self.batch_size = batch_size
        self._captured: torch.Tensor | None = None

    def _register_hook(self, layer_index: int) -> torch.utils.hooks.RemovableHandle:
        layers = resolve_decoder_layers(self.model)
        assert 0 <= layer_index < len(layers), (
            f"layer_index={layer_index} out of range for model with {len(layers)} layers"
        )

        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            # Transformer blocks return tuples; first element is the hidden state.
            # .clone() because .detach() alone shares storage — under torch.compile
            # the buffer could be reused before we .cpu() it post-forward.
            h = output[0] if isinstance(output, tuple) else output
            self._captured = h.detach().clone()
            raise _CaptureComplete

        return layers[layer_index].register_forward_hook(hook)

    @torch.no_grad()
    def extract(self, texts: list[str], layer_index: int) -> list[ExtractionResult]:
        handle = self._register_hook(layer_index)
        # try/finally for hook cleanup — an exception mid-extract would
        # otherwise leak the hook and double-register on the next call.
        # (arch doc §3 explicitly permits try/finally for this one purpose.)
        try:
            return self._extract_impl(texts, layer_index)
        finally:
            handle.remove()

    def _extract_impl(self, texts: list[str], layer_index: int) -> list[ExtractionResult]:
        results: list[ExtractionResult] = []
        for start in range(0, len(texts), self.batch_size):
            sub = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                sub,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )
            device = self.model.get_input_embeddings().weight.device
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            self._captured = None
            # The hook raises _CaptureComplete to short-circuit the forward right
            # after the target layer (no upper layers, no lm_head logits).
            try:
                self.model(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                )
            except _CaptureComplete:
                pass
            assert self._captured is not None, (
                f"forward hook on model.model.layers[{layer_index}] did not fire. "
                f"This architecture may use a different module path "
                f"(e.g. .transformer.h, .decoder.layers). Check model.named_modules()."
            )
            assert self._captured.shape[-1] == self.d_model, (
                f"captured tensor width {self._captured.shape[-1]} != "
                f"config.hidden_size {self.d_model}. Model config lies about itself."
            )
            hidden = self._captured.float().cpu()

            lengths = attention_mask.sum(dim=1).cpu()
            for i, seq_len in enumerate(lengths.tolist()):
                results.append(
                    ExtractionResult(
                        hidden_states=hidden[i, :seq_len].clone(),
                        token_ids=input_ids[i, :seq_len].cpu().tolist(),
                    )
                )
        return results
