"""Tests for the text_judges eval (nla/utils/text_judges.py + trainer wiring).

All offline: the judge client is mocked. A separate live sanity (real API) is
in __main__ behind ANTHROPIC_API_KEY presence. Run: python tests/test_text_judges.py
"""

import asyncio
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nla.utils.text_judges import (  # noqa: E402
    MATCH_SNIPPET_CHARS, N_MATCH_OPTIONS, RUBRIC_PROMPTS,
    _parse_1_10, _parse_letter, build_match_options, judge_explanations,
)

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, None)); print(f"PASS {name}")
    except Exception as e:
        RESULTS.append((name, e)); print(f"FAIL {name}: {type(e).__name__}: {e}")


def t_rubric_dims():
    assert set(RUBRIC_PROMPTS) == {"unique_info", "coherence", "writing_quality",
                                   "specificity", "repetitiveness"}
    for dim, tmpl in RUBRIC_PROMPTS.items():
        assert "{text}" in tmpl and ("1-10" in tmpl or "1 to 10" in tmpl), dim
        tmpl.format(text="x")  # no stray braces


def t_parse_1_10():
    assert _parse_1_10("7") == 7
    assert _parse_1_10(" 10 ") == 10
    assert _parse_1_10("Score: 3.") == 3
    assert _parse_1_10("ten") is None
    assert _parse_1_10("") is None
    assert _parse_1_10(None) is None
    # 10 must not parse as 1
    assert _parse_1_10("10") == 10


def t_parse_letter():
    assert _parse_letter("B", 4) == 1
    assert _parse_letter(" c ", 4) == 2
    assert _parse_letter("Answer: D", 4) == 3
    assert _parse_letter("E", 4) is None      # out of range for k=4
    assert _parse_letter("", 4) is None
    assert _parse_letter(None, 4) is None


def t_match_options_determinism_and_coverage():
    rng = np.random.default_rng(7)
    # identifier at the END: tails are what get compared, and shared suffixes
    # would make the distractor-vs-true check collide spuriously
    sources = ["x" * int(rng.integers(0, 2000)) + f" source text number {i}"
               for i in range(40)]
    positions = []
    for i in range(40):
        a = build_match_options(sources, i, seed=123)
        b = build_match_options(sources, i, seed=123)
        assert a[1] == b[1] and a[0] == b[0], "not deterministic in (seed, i)"
        tails, true_pos = a
        assert len(tails) == N_MATCH_OPTIONS
        assert all(len(t) <= MATCH_SNIPPET_CHARS + 4 for t in tails), "tail cap"
        # the true source's tail is at true_pos
        want = sources[i][-MATCH_SNIPPET_CHARS:]
        assert tails[true_pos].endswith(want[-50:]), "true source not at true_pos"
        # distractors are NOT the true source
        for p, t in enumerate(tails):
            if p != true_pos:
                assert not t.endswith(want[-50:]), "distractor equals true source"
        positions.append(true_pos)
    # seeded per-sample shuffle -> true answer position roughly uniform
    counts = np.bincount(positions, minlength=N_MATCH_OPTIONS)
    assert counts.min() >= 3, f"true_pos badly skewed: {counts}"
    # different seed -> different task
    c = build_match_options(sources, 0, seed=999)
    assert c[0] != a[0] or c[1] != a[1]


def t_match_options_insufficient():
    assert build_match_options(["only", "", ""], 0, seed=0) == (None, None)
    assert build_match_options(["a", "b", "c", ""], 3, seed=0) == (None, None)  # empty true


class _FakeMessages:
    """Mock messages.create: rubric prompts -> a digit; match prompts -> a letter."""
    def __init__(self, rubric_score="7", match_answer="A", fail_on=()):
        self.rubric_score = rubric_score
        self.match_answer = match_answer
        self.fail_on = fail_on
        self.calls = []

    async def create(self, *, model, max_tokens, messages):
        prompt = messages[0]["content"]
        self.calls.append(prompt)
        for f in self.fail_on:
            if f in prompt:
                raise RuntimeError("simulated API failure")
        text = (self.match_answer if "single letter of your answer" in prompt
                else self.rubric_score)
        class Blk:
            type = "text"
        Blk.text = text
        class Resp:
            content = [Blk()]
        return Resp()


def _mock_client(monkey_target, fake):
    import anthropic
    class FakeClient:
        def __init__(self, **kw):
            self.messages = fake
    return anthropic, FakeClient


def t_judge_aggregation_mocked():
    import anthropic
    fake = _FakeMessages(rubric_score="7", match_answer="A")
    orig = anthropic.AsyncAnthropic
    anthropic.AsyncAnthropic = lambda **kw: type("C", (), {"messages": fake})()
    try:
        sources = [f"unique source text {i} " + "y" * 900 for i in range(8)]
        expls = [f"explanation {i}" for i in range(8)]
        expls[3] = None                        # failed extraction -> skipped
        metrics, per = judge_explanations(expls, sources, seed=5, concurrency=8)
    finally:
        anthropic.AsyncAnthropic = orig

    assert metrics["n_judged"] == 7.0
    for dim in RUBRIC_PROMPTS:
        assert metrics[f"{dim}_mean"] == 7.0, (dim, metrics)
    assert per[3] == {}, "skipped sample must have no scores"
    # judge always answers A -> accuracy == fraction of samples whose true
    # source landed at position 0 (deterministic given the seed)
    expected_hits = sum(
        1 for i in range(8) if i != 3
        and build_match_options(sources, i, seed=5)[1] == 0)
    assert abs(metrics["source_match_acc"] - expected_hits / 7) < 1e-9, metrics
    assert metrics["judge_fail_rate"] == 0.0
    # job count: 7 samples x (5 rubric + 1 match)
    assert len(fake.calls) == 7 * (len(RUBRIC_PROMPTS) + 1)


def t_judge_partial_failures_mocked():
    import anthropic
    # every coherence call fails; match calls return garbage (unparseable)
    fake = _FakeMessages(rubric_score="4", match_answer="zzz",
                         fail_on=("COHERENCE",))
    orig = anthropic.AsyncAnthropic
    anthropic.AsyncAnthropic = lambda **kw: type("C", (), {"messages": fake})()
    try:
        sources = [f"source {i} " + "z" * 800 for i in range(6)]
        expls = [f"expl {i}" for i in range(6)]
        metrics, per = judge_explanations(expls, sources, seed=1, concurrency=4)
    finally:
        anthropic.AsyncAnthropic = orig
    import math
    assert math.isnan(metrics["coherence_mean"]), metrics["coherence_mean"]
    assert metrics["unique_info_mean"] == 4.0
    assert math.isnan(metrics["source_match_acc"])   # no parseable match answers
    # failures = 6 coherence + 6 match
    assert abs(metrics["judge_fail_rate"] - 12 / 36) < 1e-9
    for d in per:
        assert d["coherence"] is None and d["source_match"] is None


def t_judge_no_sources_degrades():
    """Empty sources: rubric dims still scored, source_match becomes nan."""
    import anthropic, math
    fake = _FakeMessages()
    orig = anthropic.AsyncAnthropic
    anthropic.AsyncAnthropic = lambda **kw: type("C", (), {"messages": fake})()
    try:
        metrics, per = judge_explanations(["e1", "e2"], ["", ""], seed=0)
    finally:
        anthropic.AsyncAnthropic = orig
    assert metrics["unique_info_mean"] == 7.0
    assert math.isnan(metrics["source_match_acc"])
    assert len(fake.calls) == 2 * len(RUBRIC_PROMPTS)   # no match jobs at all


def t_trainer_startup_checks():
    """text_judges without ANTHROPIC_API_KEY must die at startup, fast."""
    import subprocess, tempfile
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env["HF_HOME"] = f"/workspace-vast/{os.environ['USER']}/hf_cache"
    side = "/workspace-vast/asherps/nla-data/qwen3_8b_finefineweb_100k/rl_shuf.parquet"
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(
            [sys.executable, "-m", "nla.train_rl_vllm",
             "--av-ckpt", "/nonexistent", "--ar-ckpt", "/nonexistent",
             "--rl-parquet", side, "--sidecar", side, "--save-dir", td,
             "--evals", "base_fve", "text_judges", "--no-wandb"],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
        assert r.returncode != 0 and "ANTHROPIC_API_KEY" in (r.stderr + r.stdout), \
            r.stderr[-400:]
    # bad cadence (not a multiple of eval_every) must also die at startup
    env["ANTHROPIC_API_KEY"] = "sk-dummy"
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(
            [sys.executable, "-m", "nla.train_rl_vllm",
             "--av-ckpt", "/nonexistent", "--ar-ckpt", "/nonexistent",
             "--rl-parquet", side, "--sidecar", side, "--save-dir", td,
             "--evals", "base_fve", "text_judges",
             "--eval-every", "7", "--text-judges-every", "50", "--no-wandb"],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
        assert r.returncode != 0 and "multiple" in (r.stderr + r.stdout), \
            r.stderr[-400:]


def live_sanity():
    """Optional: 2 tiny real judge rounds (12 Opus calls). Needs ANTHROPIC_API_KEY."""
    expls = [
        "The model is reading a recipe blog post about sourdough hydration "
        "ratios; it expects a numbered instruction list to continue, with the "
        "next step likely about autolyse timing.",
        "text text text the model reads text and processes text about text.",
    ]
    sources = [
        "Mixing at 78% hydration gives an open crumb. Step 4: rest the dough "
        "for 40 minutes before the first fold. Step 5:",
        "The committee voted 7-2 to approve the zoning variance for the new "
        "library annex, with construction slated for spring.",
        "Quarterly revenue rose 12% on strong cloud demand, though hardware "
        "margins compressed for the third consecutive quarter.",
        "She tightened the final bolt on the telescope mount as the first "
        "stars appeared over the ridge.",
    ]
    metrics, per = judge_explanations(expls, sources + [""] * 0, seed=0,
                                      concurrency=12)
    print("live metrics:", {k: round(v, 3) for k, v in metrics.items()})
    assert metrics["judge_fail_rate"] < 0.5, "most judge calls failed"
    # the informative explanation should out-score the vacuous one on unique_info
    a, b = per[0].get("unique_info"), per[1].get("unique_info")
    print(f"unique_info: informative={a} vacuous={b}")
    assert a is not None and b is not None and a > b, (a, b)


if __name__ == "__main__":
    check("rubric_dims", t_rubric_dims)
    check("parse_1_10", t_parse_1_10)
    check("parse_letter", t_parse_letter)
    check("match_options_determinism_and_coverage", t_match_options_determinism_and_coverage)
    check("match_options_insufficient", t_match_options_insufficient)
    check("judge_aggregation_mocked", t_judge_aggregation_mocked)
    check("judge_partial_failures_mocked", t_judge_partial_failures_mocked)
    check("judge_no_sources_degrades", t_judge_no_sources_degrades)
    check("trainer_startup_checks", t_trainer_startup_checks)
    n_fail = sum(1 for _, e in RESULTS if e)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} PASS")
    if os.environ.get("ANTHROPIC_API_KEY") and "--live" in sys.argv:
        print("\n--- live sanity (real Opus calls) ---")
        live_sanity()
    sys.exit(1 if n_fail else 0)
