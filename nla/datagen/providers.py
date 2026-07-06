"""Completion provider backends for Stage 2 (API explanation generation).

Stage 2 calls an external LLM to produce natural-language explanations of
source text — these become the `response` column for AV-SFT and the `prompt`
content for AR-SFT. `CompletionProvider` is the pluggable interface: stage 2
code hands it a batch of fully-formed prompts and gets back a batch of
completions. Concurrency, retries, rate limits, and auth are all the
provider's problem.

Swap via `--provider-cls my.module.MyProvider` at stage2 invocation.
"""

import asyncio
import os
import time
from abc import ABC, abstractmethod

import anthropic


class CompletionProvider(ABC):
    """Submit a batch of prompts, get a batch of completions back.

    Stage 2 formats NLA-specific instruction prompts; the provider just maps
    `prompts[i] -> completion[i]` (or None for prompts that exhausted retries).
    A robust sampling engine can be plugged in by wrapping it in a subclass.

    None returns are per-prompt gave-up signals — stage2 drops those rows
    (same path as failed-extract-pattern). This means a chunk can survive
    losing a few prompts to sustained 429/500 storms instead of discarding
    511 good completions because one failed. Gaps ARE tracked: stage2 logs
    a drop count, and the parquet row count tells you exactly how many
    survived.
    """

    @abstractmethod
    def complete(self, prompts: list[str]) -> list[str | None]: ...


class AnthropicProvider(CompletionProvider):
    """Default provider: Anthropic Messages API with bounded async concurrency.

    The SDK handles transport-level retries (408/429/5xx, exponential backoff
    with jitter, respects Retry-After). High `max_retries` extends the retry
    window for sustained rate-limit storms — at max_retries=100 the SDK will
    keep backing off for minutes before giving up on one prompt.

    Per-prompt failures after exhausting retries return None (caller drops
    the row). `gather(return_exceptions=True)` collects these without nuking
    the whole batch — otherwise one stubborn 429 in a chunk of 512 wastes
    the other 511 API calls. ONLY `RateLimitError` and server-side 5xx are
    tolerated; anything else (auth, bad request, unexpected content) still
    raises — those are code bugs, not transient.

    Calls `asyncio.run()` — do not invoke from inside a running event loop.
    Stage 2 is a standalone CLI, so this is fine in practice.
    """

    # Exceptions from which we degrade to None instead of killing the batch.
    # Anything NOT in this tuple is a code bug and should still blow up loud.
    _TOLERATED = (
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
        # 529: transient server overload (esp. at high concurrency). getattr:
        # OverloadedError is top-level in current SDKs but absent in older ones
        # (the dep is unpinned); old SDKs map 529 -> InternalServerError, already
        # tolerated above, so the fallback is a harmless duplicate — NOT
        # APIStatusError, which would silently swallow auth/param errors.
        getattr(anthropic, "OverloadedError", anthropic.InternalServerError),
    )

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 300,
        temperature: float = 1.0,
        concurrency: int = 32,
        max_retries: int = 10,
    ):
        self.client = anthropic.AsyncAnthropic(max_retries=max_retries)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency

    async def _one(self, sem: asyncio.Semaphore, prompt: str) -> str | None:
        # temperature is deprecated/rejected by some newer models (e.g. opus-4-8);
        # pass it only when explicitly set so those models can be used as judges.
        kwargs = {} if self.temperature is None else {"temperature": self.temperature}
        async with sem:
            # SDK overloads reject a plain message dict + **kwargs, but both are
            # valid at runtime; constructing the exact TypedDicts isn't worth it.
            resp = await self.client.messages.create(  # type: ignore[call-overload]
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
        # refusal: source text tripped safety — no answer coming, drop this row.
        # content may be [] or the refusal message; either way, no explanation.
        if resp.stop_reason == "refusal":
            return None
        assert resp.stop_reason in ("end_turn", "max_tokens"), (
            f"unexpected stop_reason={resp.stop_reason!r} (want end_turn/max_tokens/refusal)"
        )
        assert len(resp.content) == 1 and resp.content[0].type == "text", (
            f"expected single text block, got {[b.type for b in resp.content]}"
        )
        text = resp.content[0].text.strip()
        assert text, "empty completion — refusing to emit blank explanation"
        return text

    def complete(self, prompts: list[str]) -> list[str | None]:
        async def _run() -> list[str | None | BaseException]:
            sem = asyncio.Semaphore(self.concurrency)
            return await asyncio.gather(
                *(self._one(sem, p) for p in prompts),
                return_exceptions=True,
            )

        raw = asyncio.run(_run())
        out: list[str | None] = []
        n_failed = 0
        n_refused = 0
        for i, r in enumerate(raw):
            if isinstance(r, str):
                out.append(r)
            elif r is None:
                n_refused += 1
                out.append(None)
            elif isinstance(r, self._TOLERATED):
                n_failed += 1
                out.append(None)
            elif isinstance(r, BaseException):
                # Not a transient — auth/schema/code bug. Blow up loud.
                raise r
            else:
                raise AssertionError(f"gather returned unexpected type at [{i}]: {type(r).__name__}")
        if n_failed or n_refused:
            print(f"  [AnthropicProvider] dropped {n_refused} refused + {n_failed} retry-exhausted of {len(prompts)}")
        # >20% retry-exhausted = systemic (network/rate-limit/key quota), not
        # per-row content. The old ALL-failed check let a 99%-hollow chunk be
        # written as complete — and resume never revisits completed chunks.
        # (Refusals are excluded: those are per-row content decisions.)
        if prompts and n_failed / len(prompts) > 0.2:
            raise RuntimeError(
                f"AnthropicProvider: {n_failed}/{len(prompts)} requests failed "
                f"after retries (>20%) — systemic failure, refusing to emit a "
                f"hollow chunk (it would be marked complete and never retried)."
            )
        return out


class BatchAnthropicProvider(CompletionProvider):
    """Anthropic Message Batches API — 50% cheaper than sync, async, async-style turnaround.

    Each `complete(prompts)` call partitions prompts into sub-batches of
    `max_batch_size` (Anthropic's hard limit is 100k per batch), submits all
    sub-batches concurrently, polls until each one ends, then fetches results.
    The order of returned completions matches the input order via the `custom_id`
    round-trip; missing/failed/expired requests come back as None.

    Authentication: reads `ANTHROPIC_API_KEY_BATCH` first (so the dedicated batch
    key from CLAUDE.md is preferred), falls back to `ANTHROPIC_API_KEY`. The key
    is passed explicitly to the client so the env var name is unambiguous.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 300,
        temperature: float = 1.0,
        max_batch_size: int = 50_000,
        poll_interval_s: float = 30.0,
    ):
        api_key = os.environ.get("ANTHROPIC_API_KEY_BATCH") or os.environ.get("ANTHROPIC_API_KEY")
        assert api_key, "set ANTHROPIC_API_KEY_BATCH (preferred) or ANTHROPIC_API_KEY"
        self.client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=10)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_batch_size = max_batch_size
        self.poll_interval_s = poll_interval_s

    async def _submit_subbatch(self, prompts: list[str], offset: int) -> list[str | None]:
        # Encode position within the sub-batch in custom_id so results map back
        # even when fetched out-of-order. Sub-batch-local index, not global.
        requests = [
            {
                "custom_id": f"r{i:07d}",
                "params": {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    # omit when None — some newer models reject the param
                    # (same convention as AnthropicProvider._one)
                    **({} if self.temperature is None
                       else {"temperature": self.temperature}),
                    "messages": [{"role": "user", "content": p}],
                },
            }
            for i, p in enumerate(prompts)
        ]
        # batch API accepts request dicts at runtime; SDK types want Request objs.
        batch = await self.client.messages.batches.create(requests=requests)  # type: ignore[arg-type]
        bid = batch.id
        print(f"  [batch {bid}] submitted {len(prompts)} prompts (sub-batch offset={offset})")

        # Poll until ended. Anthropic's batch lifecycle: in_progress → ended.
        # The terminal status field is `processing_status`; per-request status
        # comes back in the streamed results.
        t0 = time.monotonic()
        while True:
            await asyncio.sleep(self.poll_interval_s)
            b = await self.client.messages.batches.retrieve(bid)
            if b.processing_status == "ended":
                rc = b.request_counts
                elapsed = time.monotonic() - t0
                print(
                    f"  [batch {bid}] ended in {elapsed/60:.1f}min — "
                    f"succeeded={rc.succeeded} errored={rc.errored} "
                    f"canceled={rc.canceled} expired={rc.expired}"
                )
                break

        out: list[str | None] = [None] * len(prompts)
        first_err = None  # sample failure detail for the all-failed diagnostic
        # SDK quirk: messages.batches.results() returns a coroutine that resolves
        # to an async-iterable. Must await first, then `async for`.
        results_stream = await self.client.messages.batches.results(bid)
        async for entry in results_stream:
            # custom_id "r0000007" -> 7
            idx = int(entry.custom_id[1:])
            r = entry.result
            if r.type != "succeeded":
                # errored/canceled/expired stay None (per-row tolerance), but keep
                # a sample so a SYSTEMIC failure (bad key, invalid params -> every
                # row errored) can be raised below instead of silently producing a
                # 100%-empty chunk that gets marked complete and poisons resume.
                if first_err is None:
                    first_err = f"{r.type}: {getattr(r, 'error', '')!s:.300}"
                continue
            msg = r.message
            if msg.stop_reason == "refusal":
                continue
            if not msg.content or msg.content[0].type != "text":
                continue
            text = msg.content[0].text.strip()
            if text:
                out[idx] = text
        if prompts and all(o is None for o in out) and first_err is not None:
            raise RuntimeError(
                f"[batch {bid}] 0/{len(prompts)} requests succeeded — systemic "
                f"failure (auth/params), refusing to emit an empty chunk. "
                f"First error: {first_err}"
            )
        return out

    def complete(self, prompts: list[str]) -> list[str | None]:
        async def _run() -> list[str | None]:
            sub_batches = [
                (i, prompts[i : i + self.max_batch_size])
                for i in range(0, len(prompts), self.max_batch_size)
            ]
            results_per_sub = await asyncio.gather(
                *(self._submit_subbatch(sb, offset=off) for off, sb in sub_batches),
                return_exceptions=False,
            )
            out: list[str | None] = []
            for r in results_per_sub:
                out.extend(r)
            return out

        return asyncio.run(_run())
