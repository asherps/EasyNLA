"""Opus-judged attributes of generated explanation TEXT (opt-in eval).

Five rubric dimensions (integer 1-10, one judge call per sample x dimension)
plus one objective task, all designed to trend over RL training:

  unique_info     — distinct, specific information content (10 = dense specifics)
  coherence       — internal logical coherence (10 = fully coherent)
  writing_quality — fluency/grammar/clarity; the "writing quality degradation"
                    RL failure mode as a direct judge (10 = clean prose)
  specificity     — could this explanation only describe ITS context, or nearly
                    any text? (10 = pins the context down)
  repetitiveness  — surface + content repetition (LOWER is better; 1 = none)

  source_match    — discriminability: the judge sees the explanation + 4
                    candidate source endings (1 true + 3 distractors drawn from
                    the other eval rows, seeded) and picks which source the
                    model was reading. Metric = accuracy, chance 0.25. Should
                    RISE as explanations get more informative.

Runs on the explanations the trainer's held-out FVE eval ALREADY generated
(no second generation pass). CAVEAT (survivorship): only extraction-SUCCESSFUL
outputs are judged — during a format collapse the degenerate outputs drop out
of the judged pool, so read these means TOGETHER with eval/extraction_rate
(the collapse shows up there, not here). Enable with `--evals base_fve text_judges`;
cadence via `--text-judges-every` (a multiple of --eval-every). Needs
ANTHROPIC_API_KEY. Cost: n_eval_prompts x 6 judge calls per round.

NB: no `temperature` kwarg on the judge calls — newer models reject it.
"""

from __future__ import annotations

import asyncio
import os
import re
import statistics

import numpy as np

JUDGE_MODEL = "claude-opus-4-8"
MAX_EXPL_CHARS = 6000      # judge-input cap per explanation
MATCH_SNIPPET_CHARS = 700  # per-candidate source tail for source_match
N_MATCH_OPTIONS = 4        # 1 true + 3 distractors (chance = 0.25)
_LETTERS = "ABCDEFGH"

# One self-contained rubric per dimension, each answering with ONLY an integer
# 1-10 (max_tokens=8): single-attribute prompts calibrate better than one
# mega-call, and parsing stays trivial. {text} = the extracted explanation.
RUBRIC_PROMPTS: dict[str, str] = {
    "unique_info": """Below is an explanation of a language model's internal activation while reading some text. Rate HOW MUCH UNIQUE, SPECIFIC INFORMATION the explanation contains, on an integer scale 1-10:

  1  = essentially vacuous: generic filler ("the model is processing text") with no specific content
  5  = a few specific pieces of information amid generic filler
  10 = dense with many DISTINCT, specific pieces of information (concrete topics, entities, stances, structures, predictions)

Count only DISTINCT information — repeating one fact five ways is one piece of information. Do NOT judge whether the claims are true, only how much unique specific content is asserted. Respond with ONLY the integer 1-10, nothing else.

EXPLANATION:
{text}""",
    "coherence": """Below is an explanation of a language model's internal activation. Rate the explanation's COHERENCE on an integer scale 1-10:

  1  = incoherent word salad: fragments that don't parse, or claims that directly contradict each other
  5  = mostly readable but with disjointed jumps or internal tension between parts
  10 = fully coherent: every part parses, the parts are mutually consistent, and the whole reads as one sensible description

Judge internal coherence only — NOT whether it is accurate, and NOT its writing style. A list of bullets counts as coherent if the bullets are individually sensible and mutually consistent. Respond with ONLY the integer 1-10, nothing else.

EXPLANATION:
{text}""",
    "writing_quality": """Below is an explanation of a language model's internal activation. Rate its WRITING QUALITY on an integer scale 1-10:

  1  = broken text: severe grammar failures, truncated sentences, garbled or degenerate output
  5  = understandable but rough: awkward phrasing, grammatical slips, clumsy constructions
  10 = clean, fluent, well-formed writing a careful human editor would pass unchanged

Judge surface writing quality only (grammar, fluency, clarity) — NOT the content, accuracy, or amount of information. Terse telegraphic bullet style is fine if well-formed; judge it as bullet-style writing, not as prose. Respond with ONLY the integer 1-10, nothing else.

EXPLANATION:
{text}""",
    "specificity": """Below is an explanation of what a language model was representing while reading ONE particular text. Rate the explanation's SPECIFICITY — how strongly it pins down that particular text — on an integer scale 1-10:

  1  = could describe almost ANY text (e.g. "the model is tracking the topic and tone of a document")
  5  = narrows it to a broad genre/domain but would fit thousands of different texts
  10 = so specific that it could plausibly describe only its one source text among millions

You are NOT shown the source; judge how discriminating the DESCRIPTION itself is. Do NOT judge truthfulness or writing quality. Respond with ONLY the integer 1-10, nothing else.

EXPLANATION:
{text}""",
    "repetitiveness": """Below is one explanation of a language model's internal activation. Rate how REPETITIVE the explanation reads on an integer scale from 1 to 10:

  1  = no repetition: every phrase and point appears once
  10 = highly repetitive: the same words, phrases, sentence templates, or content are repeated over and over

Count BOTH surface repetition (repeated words/phrases/boilerplate sentence structures) AND content repetition (restating the same point in different words). This is about repetitiveness only — do NOT judge whether the explanation is accurate or informative. Works whether the explanation is prose or a list of bullets. Respond with ONLY the integer 1-10, nothing else.

EXPLANATION:
{text}""",
}

MATCH_PROMPT = """A language model produced the EXPLANATION below while reading exactly ONE of the {k} source texts. The explanation describes the model's internal state near the END of its source. Decide which source it was reading.

EXPLANATION:
{text}

{options}

Respond with ONLY the single letter of your answer ({letters}), nothing else."""


def _parse_1_10(text: str) -> int | None:
    m = re.search(r"\b(10|[1-9])\b", text or "")
    return int(m.group(1)) if m else None


def _parse_letter(text: str, k: int) -> int | None:
    """Return the option index the judge picked, or None."""
    m = re.search(rf"\b([{_LETTERS[:k]}])\b", (text or "").strip().upper())
    return _LETTERS.index(m.group(1)) if m else None


def build_match_options(sources: list[str], i: int, seed: int,
                        k: int = N_MATCH_OPTIONS,
                        snippet_chars: int = MATCH_SNIPPET_CHARS):
    """For sample i: pick k-1 distractor sources from the other rows + shuffle.

    Returns (option_texts, true_pos) — deterministic in (seed, i) so every round
    scores the identical task, and the true answer's position is uniform (the
    shuffle is seeded per-sample, not global). Sources are TAIL-truncated: the
    activation sits at the end of the context, so the tail is what the
    explanation describes. Returns (None, None) if there are not enough
    distinct other sources (or the true source is missing)."""
    if not sources[i]:
        return None, None
    others = [
        j for j in range(len(sources))
        if j != i and sources[j]
        # exclude same-doc rows: two positions in one doc yield sources that
        # are prefixes of each other — as a distractor that's the true answer
        # in disguise (rows-per-doc > 1 happens in real eval sets)
        and not (sources[j].startswith(sources[i]) or sources[i].startswith(sources[j]))
    ]
    if len(others) < k - 1:
        return None, None
    rng = np.random.default_rng((seed, i))
    picks = list(rng.choice(others, size=k - 1, replace=False))
    opts = [sources[i]] + [sources[j] for j in picks]
    order = list(rng.permutation(k))
    option_texts = [opts[order.index(pos)] for pos in range(k)]
    true_pos = order[0]  # where the true source landed
    tails = [("... " + t[-snippet_chars:]) if len(t) > snippet_chars else t
             for t in option_texts]
    return tails, int(true_pos)


def judge_explanations(explanations: list[str | None], sources: list[str],
                       *, seed: int = 0, concurrency: int = 64,
                       model: str = JUDGE_MODEL,
                       total_timeout_s: float = 600.0) -> tuple[dict, list[dict]]:
    """Judge each explanation on the 5 rubric dims + source_match.

    explanations[i] = the extracted <explanation> text for eval row i, or None
    for failed extractions (skipped). sources[i] = that row's source text
    (detokenized_text_truncated); empty strings disable source_match per-row.

    Returns (metrics, per_sample):
      metrics: {<dim>_mean (repetitiveness LOWER=better), source_match_acc,
                judge_fail_rate, n_judged}
      per_sample: one dict per input row with the raw scores (None = unscored).

    Auth: ANTHROPIC_API_KEY (SDK default). The SDK's built-in retries handle
    transient 429/5xx; per-call failures degrade to None rather than killing
    the eval round.
    """
    import anthropic

    client = anthropic.AsyncAnthropic(max_retries=6)
    sem = asyncio.Semaphore(concurrency)

    texts = [(e or "").strip()[:MAX_EXPL_CHARS] or None for e in explanations]

    # Build every job up front: (kind, sample_idx, dim_or_true_pos, prompt).
    jobs: list[tuple[str, int, object, str]] = []
    for i, t in enumerate(texts):
        if not t:
            continue
        for dim, tmpl in RUBRIC_PROMPTS.items():
            jobs.append(("rubric", i, dim, tmpl.format(text=t)))
        tails, true_pos = build_match_options(sources, i, seed)
        if tails is not None:
            options = "\n\n".join(
                f"SOURCE {_LETTERS[p]}:\n{tails[p]}" for p in range(len(tails)))
            letters = "/".join(_LETTERS[:len(tails)])
            jobs.append(("match", i, true_pos,
                         MATCH_PROMPT.format(k=len(tails), text=t,
                                             options=options, letters=letters)))

    async def one(prompt: str) -> str | None:
        async with sem:
            try:
                r = await client.messages.create(
                    model=model, max_tokens=8,
                    # NB: no temperature — newer models reject the kwarg.
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as e:
                print(f"  [text_judges] judge call failed: "
                      f"{type(e).__name__}: {str(e)[:80]}", flush=True)
                return None
        if r.content and r.content[0].type == "text":
            return r.content[0].text
        return None

    async def run():
        # Hard wall-clock bound: under DP only rank0 judges while the other
        # ranks wait at the next NCCL collective — an API retry-storm here must
        # degrade (all-None -> nan metrics), not stall past the NCCL watchdog.
        try:
            return await asyncio.wait_for(
                asyncio.gather(*(one(j[3]) for j in jobs)), total_timeout_s)
        except asyncio.TimeoutError:
            print(f"  [text_judges] TIMED OUT after {total_timeout_s:.0f}s — "
                  f"skipping this round (metrics nan).", flush=True)
            return [None] * len(jobs)

    outs = asyncio.run(run())

    per_sample: list[dict] = [{} for _ in explanations]
    n_failed = 0
    match_hits = match_total = 0
    for (kind, i, aux, _), out in zip(jobs, outs):
        if kind == "rubric":
            score = _parse_1_10(out) if out else None
            per_sample[i][aux] = score
            n_failed += score is None
        else:
            pick = _parse_letter(out, N_MATCH_OPTIONS) if out else None
            if pick is None:
                n_failed += 1
                per_sample[i]["source_match"] = None
            else:
                hit = int(pick == aux)
                per_sample[i]["source_match"] = hit
                match_hits += hit
                match_total += 1

    metrics: dict[str, float] = {}
    for dim in RUBRIC_PROMPTS:
        vals = [d[dim] for d in per_sample if isinstance(d.get(dim), int)]
        metrics[f"{dim}_mean"] = float(statistics.mean(vals)) if vals else float("nan")
    metrics["source_match_acc"] = (
        float(match_hits) / match_total if match_total else float("nan"))
    _n_match_jobs = sum(1 for j in jobs if j[0] == "match")
    # acc is conditioned on the judge ANSWERING; this exposes that conditioning
    metrics["match_answer_rate"] = (
        float(match_total) / _n_match_jobs if _n_match_jobs else float("nan"))
    metrics["judge_fail_rate"] = float(n_failed) / len(jobs) if jobs else float("nan")
    metrics["n_judged"] = float(sum(1 for t in texts if t))
    return metrics, per_sample


def require_judge_key():
    """Fail-fast startup check for trainers with text_judges enabled."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "--evals text_judges needs ANTHROPIC_API_KEY set (Opus judge calls)."
        )
