"""Reuse api_explanation across runs by joining on detokenized_text_truncated.

Stage2's API prompt depends on detokenized_text_truncated + template, not on
the model (activations come from the model, the text doesn't). Two runs with
the same tokenizer + corpus slice + positions produce identical text rows —
so a Qwen-14B run can pull explanations from a Qwen-7B run's *_explained.parquet.

No intermediate cache file: the existing explained parquets ARE the cache.
Point --cache-from at them, stage2 builds a text-hash → explanation dict,
and only calls the API for texts it hasn't seen.
"""

import hashlib

import pyarrow.parquet as pq

from nla.datagen._common import load_class


def load_explanation_cache(paths: list[str], storage_cls: str) -> dict[str, str]:
    storage = load_class(storage_cls)()
    cache: dict[str, str] = {}
    for path in paths:
        t = pq.read_table(
            storage.open_read(path),
            columns=["detokenized_text_truncated", "api_explanation"],
        )
        texts = t.column("detokenized_text_truncated").to_pylist()
        expls = t.column("api_explanation").to_pylist()
        for txt, expl in zip(texts, expls, strict=True):
            cache[hashlib.sha256(txt.encode()).hexdigest()] = expl
        print(f"  cache: +{len(texts)} from {path}")
    print(f"  cache: {len(cache)} unique texts loaded")
    return cache


def lookup(cache: dict[str, str], text: str) -> str | None:
    return cache.get(hashlib.sha256(text.encode()).hexdigest())
