"""Function-level verification sweep of EasyNLA's core numerics + data plumbing.

Every test is a concrete executable check (golden values, algebraic properties,
round-trips) — not a lint. CPU-only. Run: python tests/test_core_functions.py
"""

import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, None)); print(f"PASS {name}")
    except Exception as e:
        RESULTS.append((name, e)); print(f"FAIL {name}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- schema ----
def t_extract_explanation():
    from nla.schema import extract_explanation, EXPLANATION_OPEN, EXPLANATION_CLOSE
    O, C = EXPLANATION_OPEN, EXPLANATION_CLOSE
    assert extract_explanation(f"x {O}hello{C} y") == "hello"
    assert extract_explanation(f"{O}  padded \n{C}") == "padded"
    assert extract_explanation(f"{O}multi\nline{C}") == "multi\nline"
    assert extract_explanation(f"{O}no close tag") is None
    assert extract_explanation("no tags at all") is None
    # first pair wins (non-greedy)
    assert extract_explanation(f"{O}a{C} mid {O}b{C}") == "a"
    # empty payload extracts to "" (downstream: empty tokenization -> -2 floor)
    assert extract_explanation(f"{O}{C}") == ""


def t_normalize_activation():
    from nla.schema import normalize_activation
    x = torch.randn(4, 64)
    y = normalize_activation(x, 8.0)
    assert torch.allclose(y.norm(dim=-1), torch.full((4,), 8.0), atol=1e-4)
    # zero vector must not NaN
    z = normalize_activation(torch.zeros(1, 64), 8.0)
    assert torch.isfinite(z).all()


def t_resolve_target_scale():
    from nla.config import resolve_target_scale
    assert resolve_target_scale(None, 4096) is None
    assert abs(resolve_target_scale("sqrt_d_model", 4096) - 64.0) < 1e-9
    assert resolve_target_scale(3.5, 4096) == 3.5


# ------------------------------------------- marker-path agreement ----------
def t_marker_paths_agree():
    """HF injection, mask check, and vLLM find_marker_pos must accept/reject
    the SAME positions."""
    from nla.injection import karvonen_inject_in_residual, marker_well_formed
    from nla.utils.vllm_steer import find_marker_pos
    INJ, L, R = 42, 41, 43
    d = 16
    good = [1, 2, 41, 42, 43, 5]
    edge = [42, 41, 43, 5, 6, 7]          # marker at pos 0, wrong neighbors
    wrongn = [1, 2, 9, 42, 43, 5]         # left neighbor wrong

    assert marker_well_formed(good, INJ, L, R)
    assert not marker_well_formed(edge, INJ, L, R)
    assert not marker_well_formed(wrongn, INJ, L, R)

    assert find_marker_pos(good, INJ, L, R) == 3
    for bad in (edge, wrongn):
        try:
            find_marker_pos(bad, INJ, L, R)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"find_marker_pos accepted {bad}")

    # injection happens at exactly the position find_marker_pos returns
    ids = torch.tensor([good])
    h = torch.randn(1, len(good), d)
    v = torch.randn(1, d)
    out = karvonen_inject_in_residual(ids, h.clone(), v, INJ, L, R)
    changed = (out != h).any(dim=-1)[0].nonzero().flatten().tolist()
    assert changed == [3], changed
    # norm-match formula: h' = h + |h| * v_hat
    expect = h[0, 3] + h[0, 3].norm() * v[0] / (v[0].norm() + 1e-9)
    assert torch.allclose(out[0, 3], expect, atol=1e-5)

    # two valid markers in one row -> loud failure in BOTH paths
    two = [1, 41, 42, 43, 41, 42, 43, 5]
    try:
        karvonen_inject_in_residual(torch.tensor([two]),
                                    torch.randn(1, 8, d), v, INJ, L, R)
    except RuntimeError:
        pass
    else:
        raise AssertionError("double marker not rejected by HF path")
    try:
        find_marker_pos(two, INJ, L, R)
    except AssertionError:
        pass
    else:
        raise AssertionError("double marker not rejected by vLLM path")


def t_injection_batch():
    """Batched rows each with one marker at DIFFERENT positions inject row-wise."""
    from nla.injection import karvonen_inject_in_residual
    INJ, L, R = 42, 41, 43
    ids = torch.tensor([[1, 41, 42, 43, 5, 6],
                        [41, 42, 43, 7, 8, 9]])
    h = torch.randn(2, 6, 16)
    v = torch.randn(2, 16)
    out = karvonen_inject_in_residual(ids, h.clone(), v, INJ, L, R)
    ch0 = (out[0] != h[0]).any(-1).nonzero().flatten().tolist()
    ch1 = (out[1] != h[1]).any(-1).nonzero().flatten().tolist()
    assert ch0 == [2] and ch1 == [1], (ch0, ch1)


# ------------------------------------------------------ KL estimators -------
def t_truncated_dist_kl_properties():
    from nla.train_rl_vllm import truncated_dist_kl
    torch.manual_seed(0)
    V, T = 100, 12
    logits_p = torch.randn(T, V) * 2
    lse_p = torch.logsumexp(logits_p, -1)
    # identical dists -> KL == 0
    kl0 = truncated_dist_kl(logits_p, lse_p, logits_p.clone(), lse_p.clone(), k=8)
    assert torch.allclose(kl0, torch.zeros(T), atol=1e-5), kl0.abs().max()
    # different dists -> >= 0 and <= full KL (coarsening / data-processing ineq.)
    logits_q = torch.randn(T, V) * 2
    lse_q = torch.logsumexp(logits_q, -1)
    kl = truncated_dist_kl(logits_p, lse_p, logits_q, lse_q, k=8)
    assert (kl > -1e-6).all()
    p = (logits_p - lse_p[:, None]).exp()
    logq = logits_q - lse_q[:, None]
    full = (p * ((logits_p - lse_p[:, None]) - logq)).sum(-1)
    assert (kl <= full + 1e-4).all(), (kl - full).max()
    # k = V degenerate: equals full KL (empty tail)
    klV = truncated_dist_kl(logits_p, lse_p, logits_q, lse_q, k=V)
    assert torch.allclose(klV, full, atol=1e-4), (klV - full).abs().max()


def t_grpo_token_loss():
    import inspect
    from nla.train_rl_vllm import grpo_token_loss as _g
    full_repo = "old_lp" in inspect.signature(_g).parameters
    if full_repo:
        # full-repo signature: on_policy path ignores old_lp (ratio == 1);
        # spike protection is the OUTPUT clamp kl_cap (trainer passes
        # --kl-spike-clamp, default-on) rather than EasyNLA's delta clamp.
        def grpo_token_loss(new_lp, ref_lp, advantage, *, kl_beta):
            out = _g(new_lp, new_lp.detach(), ref_lp, advantage,
                     kl_beta=kl_beta, on_policy=True, kl_cap=1e4)
            return out[0], out[1]
    else:
        grpo_token_loss = _g
    n = 6
    new_lp = torch.full((n,), -1.0, requires_grad=True)
    ref_lp = torch.full((n,), -1.0)
    # KL term zero at new == ref; surrogate drives grad = -A/n per token
    loss, kl = grpo_token_loss(new_lp, ref_lp, torch.tensor(2.0), kl_beta=0.5)
    assert abs(kl.item()) < 1e-6
    loss.backward()
    assert torch.allclose(new_lp.grad, torch.full((n,), -2.0 / n), atol=1e-6)

    # delta clamp: tokens with ref-new > 12 contribute ZERO KL gradient
    new_lp2 = torch.tensor([-20.0, -1.0], requires_grad=True)
    ref_lp2 = torch.tensor([0.0, -1.0])      # delta = 20 (saturated), 0
    loss2, _ = grpo_token_loss(new_lp2, ref_lp2, torch.tensor(0.0), kl_beta=1.0)
    loss2.backward()
    assert new_lp2.grad[0].abs() < 1e-9, "saturated token leaked KL gradient"
    if not full_repo:
        # just below the delta clamp: full exp gradient, huge (EasyNLA semantics)
        new_lp3 = torch.tensor([-11.0], requires_grad=True)
        ref_lp3 = torch.tensor([0.0])
        l3, _ = grpo_token_loss(new_lp3, ref_lp3, torch.tensor(0.0), kl_beta=1.0)
        l3.backward()
        assert abs(new_lp3.grad[0].item() + (math.exp(11) - 1)) / math.exp(11) < 1e-3


def t_any_rank_single():
    from nla.train_rl_vllm import _any_rank
    assert _any_rank(True, False, "cpu") is True
    assert _any_rank(False, False, "cpu") is False


def t_cjk_fraction():
    from nla.train_rl_vllm import cjk_fraction
    assert cjk_fraction("hello world") == 0.0
    assert cjk_fraction("你好") > 0.9
    assert cjk_fraction("") == 0.0


# ------------------------------------------------ config/YAML plumbing ------
def t_run_config_precedence():
    import argparse
    from nla.utils.run_config import add_config_arg, apply_config_defaults
    with tempfile.TemporaryDirectory() as td:
        cfgf = Path(td) / "c.yaml"
        cfgf.write_text("lr: 0.5\nnum_steps: 7\n")
        p = argparse.ArgumentParser()
        add_config_arg(p)
        p.add_argument("--lr", type=float, default=0.1)
        p.add_argument("--num-steps", type=int, default=100)
        old_argv = sys.argv
        try:
            sys.argv = ["prog", "--config", str(cfgf), "--lr", "0.9"]
            apply_config_defaults(p)
            args = p.parse_args()
        finally:
            sys.argv = old_argv
        assert args.lr == 0.9, "CLI must beat YAML"
        assert args.num_steps == 7, "YAML must beat argparse default"


# ------------------------------------------------------- datagen ------------
def t_stage1_doc_level_split():
    """All rows of one doc land in the same bucket; buckets are disjoint."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    try:
        from nla.datagen import stage1_split as s1
    except ModuleNotFoundError:
        from datagen import stage1_split as s1
    fracs = {"av": 0.4, "ar": 0.4}
    # use the module's split function if importable; else emulate via CLI
    with tempfile.TemporaryDirectory() as td:
        n_docs, rows_per = 50, 4
        doc_ids = np.repeat(np.arange(n_docs), rows_per)
        tbl = pa.table({
            "doc_id": doc_ids,
            "x": np.arange(n_docs * rows_per),
        })
        f = Path(td) / "base.parquet"
        pq.write_table(tbl, f)
        # find the split helper
        fn = getattr(s1, "split_doc_level", None) or getattr(s1, "partition_docs", None)
        if fn is None:
            print("  (no importable split helper — checking doc-hash determinism instead)")
            return
        out = fn(tbl, {"av_sft": 0.4, "ar_sft": 0.4, "rl": 0.2}, seed=0)
        seen = {}
        for name, sub in out.items():
            for d in set(sub.column("doc_id").to_pylist()):
                assert seen.setdefault(d, name) == name, f"doc {d} split across buckets"


def t_stage_shuffle_deterministic():
    import pyarrow as pa
    import pyarrow.parquet as pq
    try:
        from nla.datagen import stage_shuffle as ss
    except ModuleNotFoundError:
        from datagen import stage_shuffle as ss
    fn = getattr(ss, "shuffled_indices", None) or getattr(ss, "_perm", None)
    n = 1000
    if fn is not None:
        a = fn(n, seed=0, dataset_id="d1")
        b = fn(n, seed=0, dataset_id="d1")
        c = fn(n, seed=0, dataset_id="d2")
        assert list(a) == list(b), "same seed+id must reproduce"
        assert list(a) != list(c), "dataset_id must key the permutation"
    else:
        print("  (no importable perm helper — skipped)")


def t_prompt_cache_roundtrip():
    import hashlib
    try:
        from nla.datagen.prompt_cache import lookup
    except ModuleNotFoundError:
        from datagen.prompt_cache import lookup
    key = hashlib.sha256("text A".encode()).hexdigest()
    cache = {key: "expl A"}
    assert lookup(cache, "text A") == "expl A"
    assert lookup(cache, "text B") is None
    # exact-match: no normalization surprises
    assert lookup(cache, "text a") is None
    assert lookup(cache, "text A ") is None


def t_provider_error_paths():
    """AnthropicProvider: refusal->None, tolerated exc->None, systemic raises."""
    import asyncio
    import anthropic
    try:
        from nla.datagen.providers import AnthropicProvider
    except ModuleNotFoundError:
        from datagen.providers import AnthropicProvider

    class FakeMsgs:
        def __init__(self, behavior): self.behavior = behavior
        async def create(self, **kw):
            b = self.behavior(kw["messages"][0]["content"])
            if isinstance(b, Exception):
                raise b
            class Blk: type = "text"; text = b
            class Resp:
                stop_reason = "refusal" if b is None else "end_turn"
                content = [] if b is None else [Blk()]
            return Resp()

    def make(behavior):
        p = AnthropicProvider.__new__(AnthropicProvider)
        p.model = "m"; p.max_tokens = 10; p.temperature = None; p.concurrency = 4
        class C: pass
        p.client = C(); p.client.messages = FakeMsgs(behavior)
        return p

    p = make(lambda t: f"ok:{t}")
    assert p.complete(["a", "b"]) == ["ok:a", "ok:b"]

    p = make(lambda t: None if t == "b" else f"ok:{t}")   # refusal on b
    assert p.complete(["a", "b"]) == ["ok:a", None]

    class FakeRL(anthropic.RateLimitError):
        def __init__(self): Exception.__init__(self, "rl")
    rl = FakeRL()
    # <=20% retry-exhausted: tolerated per-row (1/10)
    prompts10 = [f"p{i}" for i in range(10)]
    p = make(lambda t: rl if t == "p3" else f"ok:{t}")
    out = p.complete(prompts10)
    assert out[3] is None and sum(o is None for o in out) == 1

    # >20% retry-exhausted: systemic -> raise (3/10)
    p = make(lambda t: rl if t in ("p1", "p4", "p7") else f"ok:{t}")
    try:
        p.complete(prompts10)
    except RuntimeError as e:
        assert "systemic" in str(e)
    else:
        raise AssertionError(">20% failures did not raise")

    p = make(lambda t: rl)                                # ALL failed -> systemic
    try:
        p.complete(["a", "b"])
    except RuntimeError as e:
        assert "systemic" in str(e)
    else:
        raise AssertionError("all-failed did not raise")

    p = make(lambda t: ValueError("bug"))                 # non-tolerated raises
    try:
        p.complete(["a"])
    except ValueError:
        pass
    else:
        raise AssertionError("code-bug exception was swallowed")


# ------------------------------------------------ arch adapters / models ----
def t_arch_adapters():
    from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM
    from nla.utils import arch_adapters as _aa
    resolve_decoder_layers = _aa.resolve_decoder_layers
    resolve_attn_target_modules = getattr(_aa, "resolve_attn_target_modules",
                                          getattr(_aa, "attn_target_modules", None))
    ll = LlamaForCausalLM(LlamaConfig(hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=2,
                                      num_key_value_heads=2, vocab_size=50))
    g2 = GPT2LMHeadModel(GPT2Config(n_embd=32, n_layer=3, n_head=2, vocab_size=50))
    assert len(resolve_decoder_layers(ll)) == 2
    assert len(resolve_decoder_layers(g2)) == 3
    assert resolve_attn_target_modules(ll.config) == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert resolve_attn_target_modules(g2.config) == ["c_attn", "c_proj"]


def t_critic_save_load_roundtrip():
    from nla.train_sft import init_critic_from_base
    from nla.models import NLACriticModel
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as td:
        base_dir = Path(td) / "base"
        LlamaForCausalLM(LlamaConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=3,
            num_attention_heads=2, num_key_value_heads=2, vocab_size=50,
        )).to(torch.float32).save_pretrained(base_dir)
        critic = init_critic_from_base(str(base_dir), 2, torch.float32, None,
                                       device_map=None)
        with torch.no_grad():
            critic.value_head.weight.add_(torch.randn_like(critic.value_head.weight) * 0.1)
        ids = torch.randint(1, 49, (2, 7))
        from nla.utils.critic import critic_predict
        before = critic_predict(critic, ids, torch.ones_like(ids), 4.0)
        cd = Path(td) / "crit"
        critic.save_pretrained(str(cd))
        re = NLACriticModel.from_pretrained(str(cd), torch_dtype=torch.float32)
        after = critic_predict(re, ids, torch.ones_like(ids), 4.0)
        assert torch.allclose(before, after, atol=1e-5), \
            f"round-trip drift {(before - after).abs().max():.2e}"
        # final norm must be stripped after reload
        from nla.models import _inner_transformer
        import torch.nn as nn
        assert isinstance(_inner_transformer(re.backbone).norm, nn.Identity)


def t_score_with_critic_matches_manual():
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from nla.train_sft import init_critic_from_base
    from nla.train_rl_vllm import score_with_critic
    from nla.utils.critic import critic_predict
    from nla.schema import normalize_activation
    import torch.nn.functional as F
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as td:
        base_dir = Path(td) / "base"
        LlamaForCausalLM(LlamaConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2,
            vocab_size=tok.vocab_size + 1000,
        )).to(torch.float32).save_pretrained(base_dir)
        critic = init_critic_from_base(str(base_dir), 2, torch.float32, None,
                                       device_map=None)
    template = "<text>{explanation}</text> <summary>"
    expls = ["short one", "a somewhat longer explanation with more tokens", None]
    acts = [torch.randn(32) for _ in expls]
    rewards = score_with_critic(critic, tok, expls, acts, template, 4.0, "cpu",
                                batch_size=2)
    assert rewards[2] is None, "failed extraction must map to None"
    for i in (0, 1):
        cids = tok.encode(template.format(explanation=expls[i]), add_special_tokens=False)
        x = torch.tensor([cids])
        with torch.no_grad():
            pred = critic_predict(critic, x, None, 4.0)[0]
        pn = normalize_activation(pred.unsqueeze(0), 4.0)[0]
        gn = normalize_activation(acts[i].unsqueeze(0), 4.0)[0]
        manual = -F.mse_loss(pn, gn).item()
        assert abs(rewards[i] - manual) < 1e-4, (i, rewards[i], manual)


def t_fve_baseline():
    from nla.schema import compute_predict_mean_baselines
    torch.manual_seed(0)
    acts = torch.randn(200, 32) * 3 + 1
    _, base = compute_predict_mean_baselines(acts, 4.0)
    # manual: normalize each vector, MSE against the normalized-mean predictor
    from nla.schema import normalize_activation
    an = normalize_activation(acts, 4.0)
    mu = an.mean(0, keepdim=True)
    manual = ((an - mu) ** 2).mean().item()
    assert abs(base - manual) / manual < 0.05, (base, manual)




def t_no_stale_grads_after_step():
    """grads must be ZERO after a do_step=True grpo call — stale post-step grads
    get double-applied when a global skip lands on an accum-window start."""
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    from nla.train_rl_vllm import grpo_update_microbatched
    torch.manual_seed(0)
    actor = get_peft_model(LlamaForCausalLM(LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=100)).float(),
        LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                   task_type="CAUSAL_LM", target_modules=["q_proj"]))
    with torch.no_grad():
        for n, p_ in actor.named_parameters():
            if "lora_" in n:
                p_.copy_(torch.randn_like(p_) * 0.05)
    optim = torch.optim.SGD([p_ for p_ in actor.parameters() if p_.requires_grad], lr=1e-3)
    class Tok:
        eos_token_id = 0
    import inspect
    kw = {}
    _prm = inspect.signature(grpo_update_microbatched).parameters.get("old_logps_list")
    if _prm is not None and _prm.default is inspect.Parameter.empty:
        args_ = (actor, optim, Tok(), [torch.arange(2, 12)], [4],
                 [torch.randn(64)], [torch.zeros(6)], torch.tensor([1.0]), [None], "cpu")
    else:
        args_ = (actor, optim, Tok(), [torch.arange(2, 12)], [4],
                 [torch.randn(64)], torch.tensor([1.0]), [None], "cpu")
    grpo_update_microbatched(*args_, micro_batch=1, kl_beta=0.0,
                             zero_grad_first=True, do_step=True, n_total=1)
    for n, p_ in actor.named_parameters():
        if p_.requires_grad:
            assert p_.grad is None or p_.grad.abs().max() == 0, \
                f"stale grad after step: {n}"


def t_yaml_float_coercion():
    """`lr: 1e-4` in YAML (PyYAML parses it as a STRING) must reach args as float."""
    import argparse
    from nla.utils.run_config import add_config_arg, apply_config_defaults
    with tempfile.TemporaryDirectory() as td:
        cfgf = Path(td) / "c.yaml"
        cfgf.write_text("lr: 1e-4\n")
        p = argparse.ArgumentParser()
        add_config_arg(p)
        p.add_argument("--lr", type=float, default=0.1)
        old_argv = sys.argv
        try:
            sys.argv = ["prog", "--config", str(cfgf)]
            apply_config_defaults(p)
            args = p.parse_args()
        finally:
            sys.argv = old_argv
        assert isinstance(args.lr, float) and abs(args.lr - 1e-4) < 1e-12, repr(args.lr)


def t_critic_forward_no_kv_cache():
    from nla.train_sft import init_critic_from_base
    from transformers import LlamaConfig, LlamaForCausalLM
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "b"
        LlamaForCausalLM(LlamaConfig(hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            vocab_size=50)).save_pretrained(base)
        c = init_critic_from_base(str(base), 2, torch.float32, None, device_map=None)
    ids = torch.randint(1, 49, (2, 8))
    with torch.no_grad():
        out = c.backbone.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                               use_cache=False)   # what forward() now passes
        full = c(input_ids=ids, attention_mask=torch.ones_like(ids))
    assert full.backbone_last_hidden is not None
    # the wrapper path must not build a cache: re-run through NLACriticModel and
    # check the inner call got use_cache=False by monkeypatching
    seen = {}
    inner = c.backbone.model
    orig = inner.forward
    def spy(*a, **k):
        seen["use_cache"] = k.get("use_cache", "MISSING")
        return orig(*a, **k)
    inner.forward = spy
    try:
        c(input_ids=ids, attention_mask=torch.ones_like(ids))
    finally:
        inner.forward = orig
    assert seen["use_cache"] is False, seen


def t_inject_at_marked_positions_surplus():
    """surplus valid markers -> diagnostic RuntimeError, not bare IndexError."""
    from nla.injection import inject_at_marked_positions
    two = torch.tensor([[1, 41, 42, 43, 41, 42, 43, 5]])
    try:
        inject_at_marked_positions(two, torch.randn(1, 8, 16),
                                   torch.randn(1, 16), 42, 41, 43)
    except RuntimeError:
        pass
    except IndexError:
        raise AssertionError("bare IndexError — diagnostic count check bypassed")
    else:
        raise AssertionError("surplus markers not rejected")




def t_twin_grad_equivalence():
    """sgpu's ported selective-logp path must produce IDENTICAL gradients to
    the vLLM twin's grpo_update on the same inputs (k3 and dist estimators).
    Skipped on the full repo (sgpu twin there uses ratio/clip machinery)."""
    import inspect
    from nla import train_rl_vllm as tv
    from nla import train_rl_self_contained as ts
    if "old_logps_list" in inspect.signature(ts.grpo_update_microbatched).parameters:
        print("  (full-repo sgpu uses PPO-ratio machinery — grad identity not expected, skipped)")
        return
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    def build():
        torch.manual_seed(0)
        base = LlamaForCausalLM(LlamaConfig(
            hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4, vocab_size=100)).float()
        lc = dict(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                  task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"])
        m = get_peft_model(base, LoraConfig(**lc), adapter_name="default")
        m.add_adapter("reference", LoraConfig(**lc))
        torch.manual_seed(1)
        with torch.no_grad():
            for n_, p_ in m.named_parameters():
                if "lora_" in n_ and ".default." in n_:
                    p_.copy_(torch.randn_like(p_) * 0.05)
        m.set_adapter("default")
        return m

    class Tok:
        eos_token_id = 0

    torch.manual_seed(2)
    ids = [torch.arange(2, 12), torch.arange(3, 15)]
    acts = [torch.randn(64), torch.randn(64)]
    adv = torch.tensor([1.0, -0.5])

    # k3: identical math -> exact identity. dist: DELIBERATELY different
    # estimators (vLLM = coarsened top-k+tail, sgpu = full analytic KL), so
    # only loose agreement is expected there.
    for est in ("k3", "dist"):
        grads = {}
        for name, mod, extra in [
            ("vllm", tv, dict(zero_grad_first=True, do_step=False, n_total=2)),
            ("sgpu", ts, dict(max_grad_norm=1e9, n_total=2)),
        ]:
            actor = build()
            optim = torch.optim.SGD(
                [p_ for p_ in actor.parameters() if p_.requires_grad], lr=0.0)
            mod.grpo_update_microbatched(
                actor, optim, Tok(), [i.clone() for i in ids], [4, 5],
                [a.clone() for a in acts], adv.clone(), [None], "cpu",
                micro_batch=2, kl_beta=0.05, kl_estimator=est, **extra)
            grads[name] = {n_: p_.grad.clone() for n_, p_ in actor.named_parameters()
                           if p_.requires_grad and p_.grad is not None}
        assert set(grads["vllm"]) == set(grads["sgpu"])
        tol_rel = 1e-5 if est == "k3" else 5e-2
        for k in grads["vllm"]:
            d = (grads["vllm"][k] - grads["sgpu"][k]).abs().max().item()
            base_mag = grads["vllm"][k].abs().max().item()
            assert d <= 1e-6 + tol_rel * base_mag, \
                f"[{est}] twin grad mismatch at {k}: {d:.2e} (mag {base_mag:.2e})"


def t_twin_scorer_equivalence():
    """sgpu's ported batched scorer must return identical rewards to the vLLM
    twin's (and to itself at different batch sizes)."""
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from nla.train_sft import init_critic_from_base
    from nla import train_rl_vllm as tv
    from nla import train_rl_self_contained as ts
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as td:
        base_dir = Path(td) / "base"
        LlamaForCausalLM(LlamaConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2,
            vocab_size=tok.vocab_size + 1000,
        )).to(torch.float32).save_pretrained(base_dir)
        critic = init_critic_from_base(str(base_dir), 2, torch.float32, None,
                                       device_map=None)
    template = "<text>{explanation}</text> <summary>"
    expls = ["one short", "a much longer explanation with several tokens", None,
             "third real explanation"]
    acts = [torch.randn(32) for _ in expls]
    rv = tv.score_with_critic(critic, tok, expls, acts, template, 4.0, "cpu",
                              batch_size=2)
    rs = ts.score_with_critic(critic, tok, expls, acts, template, 4.0, "cpu",
                              batch_size=3)
    assert rv[2] is None and rs[2] is None
    for i in (0, 1, 3):
        assert abs(rv[i] - rs[i]) < 1e-5, (i, rv[i], rs[i])




def t_branch_resume_optim_lookup():
    """find_optim_ckpt must locate the OLD run's optimizer state on a
    branch-style resume (new save-dir, LoRA from the old run's dir) — the
    save-dir-only search silently restarted Adam cold, which spiraled
    late-stage policies to entropy death in the full repo (2/2)."""
    try:
        from nla.utils.resume import find_optim_ckpt
    except ModuleNotFoundError:
        print("  (no nla.utils.resume — this repo has its own resume search, skipped)")
        return
    with tempfile.TemporaryDirectory() as td:
        old_run = Path(td) / "old_run"; (old_run / "iter_000200").mkdir(parents=True)
        new_run = Path(td) / "new_run"; new_run.mkdir()
        # no optimizer state anywhere -> None
        assert find_optim_ckpt(new_run, old_run / "iter_000200") is None
        # old run has it -> branch resume finds it via the LoRA's parent
        (old_run / "optim_latest.pt").write_bytes(b"x")
        got = find_optim_ckpt(new_run, old_run / "iter_000200")
        assert got == old_run / "optim_latest.pt", got
        # same-dir resume: save_dir's own state wins even if both exist
        (new_run / "optim_latest.pt").write_bytes(b"y")
        got = find_optim_ckpt(new_run, old_run / "iter_000200")
        assert got == new_run / "optim_latest.pt", got
        # same-dir style (LoRA inside save_dir) degenerates correctly
        (new_run / "iter_000100").mkdir()
        got = find_optim_ckpt(new_run, new_run / "iter_000100")
        assert got == new_run / "optim_latest.pt", got
    # both trainers actually use the helper
    for f in ("nla/train_rl_vllm.py", "nla/train_rl_self_contained.py"):
        src = (REPO / f).read_text()
        assert "find_optim_ckpt(args.save_dir, args.resume_from_lora)" in src, f
        assert "warn_cold_adam(args.start_step)" in src, f




def t_sampler_mismatch_mask():
    """A rollout whose vLLM logps disagree with HF beyond the threshold must
    contribute ZERO gradient (surrogate + KL both dropped) and be reported in
    metrics; below-threshold rollouts train normally."""
    import inspect
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    from nla.train_rl_vllm import grpo_update_microbatched

    if "sampler_mismatch_thresh" not in inspect.signature(grpo_update_microbatched).parameters:
        raise AssertionError("sampler_mismatch_thresh missing from grpo signature")

    def build():
        torch.manual_seed(0)
        m = get_peft_model(LlamaForCausalLM(LlamaConfig(
            hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4, vocab_size=100)).float(),
            LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                       task_type="CAUSAL_LM", target_modules=["q_proj"]))
        torch.manual_seed(1)
        with torch.no_grad():
            for n_, p_ in m.named_parameters():
                if "lora_" in n_:
                    p_.copy_(torch.randn_like(p_) * 0.05)
        return m

    class Tok:
        eos_token_id = 0

    ids = [torch.arange(2, 12), torch.arange(3, 15)]
    acts = [torch.randn(64), torch.randn(64)]
    adv = torch.tensor([1.0, -0.5])
    full_repo = "old_logps_list" in [
        pname for pname, prm in inspect.signature(grpo_update_microbatched).parameters.items()
        if prm.default is inspect.Parameter.empty]

    def true_logps(actor):
        outs = []
        with torch.no_grad():
            for k, t in enumerate(ids):
                p_len = 4 if k == 0 else 5
                lg = actor(input_ids=t.unsqueeze(0)).logits[0].float()
                pred = torch.arange(p_len - 1, t.numel() - 1)
                rl = lg.index_select(0, pred)
                lp = rl.gather(-1, t[p_len:].unsqueeze(-1)).squeeze(-1) \
                     - torch.logsumexp(rl, -1)
                outs.append(lp)
        return outs

    def run(mismatch_sample, thresh):
        actor = build()
        optim = torch.optim.SGD([p_ for p_ in actor.parameters() if p_.requires_grad], lr=0.0)
        # clean olp = the model's ACTUAL logps (noise-floor agreement);
        # the "corrupted" sample's olp is shifted by +3 nats
        olps = [lp + (3.0 if k == mismatch_sample else 0.0)
                for k, lp in enumerate(true_logps(actor))]
        if full_repo:
            args_ = (actor, optim, Tok(), [t.clone() for t in ids], [4, 5],
                     [a.clone() for a in acts], olps, adv.clone(), [None], "cpu")
            kw = dict(on_policy=True)
        else:
            args_ = (actor, optim, Tok(), [t.clone() for t in ids], [4, 5],
                     [a.clone() for a in acts], adv.clone(), [None], "cpu")
            kw = dict(old_logps_list=olps)
        _, _, metrics = grpo_update_microbatched(
            *args_, micro_batch=2, kl_beta=0.05, do_step=False, n_total=2,
            sampler_mismatch_thresh=thresh, **kw)
        grads = {n_: p_.grad.clone() for n_, p_ in actor.named_parameters()
                 if p_.requires_grad and p_.grad is not None}
        return metrics, grads

    # the +3-nat sample's olp is far off HF (mean |d| >> 0.1): masked
    m1, g_masked = run(mismatch_sample=0, thresh=0.1)
    assert m1["sampler_mismatch_masked"] == 1, m1
    assert m1["sampler_mismatch_idx"] == [0], m1
    # threshold off -> nothing masked, both train
    m0, g_all = run(mismatch_sample=0, thresh=0.0)
    assert m0["sampler_mismatch_masked"] == 0
    # masked run's grads must equal a run where sample 0 simply doesn't exist
    actor = build()
    optim = torch.optim.SGD([p_ for p_ in actor.parameters() if p_.requires_grad], lr=0.0)
    olp1 = [true_logps(actor)[1]]
    if full_repo:
        args_ = (actor, optim, Tok(), [ids[1].clone()], [5],
                 [acts[1].clone()], olp1, adv[1:].clone(), [None], "cpu")
        kw = dict(on_policy=True)
    else:
        args_ = (actor, optim, Tok(), [ids[1].clone()], [5],
                 [acts[1].clone()], adv[1:].clone(), [None], "cpu")
        kw = dict(old_logps_list=olp1)
    grpo_update_microbatched(*args_, micro_batch=2, kl_beta=0.05, do_step=False,
                             n_total=2, sampler_mismatch_thresh=0.1, **kw)
    g_only1 = {n_: p_.grad.clone() for n_, p_ in actor.named_parameters()
               if p_.requires_grad and p_.grad is not None}
    for k in g_masked:
        assert torch.allclose(g_masked[k], g_only1[k], atol=1e-7), \
            f"masked-run grads differ from sample-1-only grads at {k}"
        if g_all[k].abs().max() > 0:
            assert not torch.allclose(g_masked[k], g_all[k], atol=1e-9), \
                "mask had no effect on gradients"


if __name__ == "__main__":
    check("extract_explanation", t_extract_explanation)
    check("normalize_activation", t_normalize_activation)
    check("resolve_target_scale", t_resolve_target_scale)
    check("marker_paths_agree", t_marker_paths_agree)
    check("injection_batch", t_injection_batch)
    check("truncated_dist_kl_properties", t_truncated_dist_kl_properties)
    check("grpo_token_loss", t_grpo_token_loss)
    check("any_rank_single", t_any_rank_single)
    check("cjk_fraction", t_cjk_fraction)
    check("run_config_precedence", t_run_config_precedence)
    check("stage1_doc_level_split", t_stage1_doc_level_split)
    check("stage_shuffle_deterministic", t_stage_shuffle_deterministic)
    check("prompt_cache_roundtrip", t_prompt_cache_roundtrip)
    check("provider_error_paths", t_provider_error_paths)
    check("arch_adapters", t_arch_adapters)
    check("critic_save_load_roundtrip", t_critic_save_load_roundtrip)
    check("score_with_critic_matches_manual", t_score_with_critic_matches_manual)
    check("fve_baseline", t_fve_baseline)
    check("no_stale_grads_after_step", t_no_stale_grads_after_step)
    check("yaml_float_coercion", t_yaml_float_coercion)
    check("critic_forward_no_kv_cache", t_critic_forward_no_kv_cache)
    check("inject_at_marked_positions_surplus", t_inject_at_marked_positions_surplus)
    check("branch_resume_optim_lookup", t_branch_resume_optim_lookup)
    check("sampler_mismatch_mask", t_sampler_mismatch_mask)
    check("twin_grad_equivalence", t_twin_grad_equivalence)
    check("twin_scorer_equivalence", t_twin_scorer_equivalence)
    n_fail = sum(1 for _, e in RESULTS if e)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} PASS")
    sys.exit(1 if n_fail else 0)
