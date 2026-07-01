"""Stage 0: Base activation extraction — corpus → base.parquet.

Forward the base model over a corpus, sample ~N token positions per document,
grab the hidden state at each, write RAW vectors to parquet.

Vectors are stored UNNORMALIZED (norm="none" in sidecar). Data-gen never
normalizes — that's a training-time decision. Raw storage preserves
magnitude info and keeps the pipeline flexible.

Output schema (arch doc §2 Stage 0):
    n_raw_tokens                int        1-indexed count of tokens up to and including the extraction position
    detokenized_text_truncated  str        decoded text (skip_special_tokens=True) up to the extraction position
    activation_vector           list[float] RAW hidden state at layer K, position n_raw_tokens-1
    activation_layer            int        K
    doc_id                      str        provenance

The extractor backend is pluggable — anything implementing ActivationExtractor
works. Default: HFExtractor with device_map=auto.
"""

import argparse
import hashlib
import os
import random
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from tqdm import tqdm

from nla.datagen._common import add_storage_args, load_class, make_storage, parse_kwargs
from nla.datagen.extractors import ActivationExtractor
from nla.datagen.sidecar import NLADatasetMeta, NLAExtractionMeta, write_sidecar

_MIN_POSITION = 50  # need enough left-context for the activation to be meaningful


def _schema(d_model: int) -> pa.Schema:
    # FixedSizeList (not variable-length ListArray) — activation vectors are
    # all exactly d_model wide. No offset array → no int32-offset overflow at
    # ~600k rows, and more importantly no uint32 byte-offset overflow in
    # ChunkedArray.take() at 4 GiB values buffer (silently corrupted ~40% of
    # the 100k RL run before this change). Round-trips to numpy trivially:
    # col.combine_chunks().values.to_numpy().reshape(n, d_model).
    return pa.schema([
        ("n_raw_tokens", pa.int64()),
        ("detokenized_text_truncated", pa.string()),
        ("activation_vector", pa.list_(pa.float32(), d_model)),
        ("activation_layer", pa.int64()),
        ("doc_id", pa.string()),
    ])


def _sample_positions(
    token_ids: list[int], n_positions: int, special_ids: set[int], doc_id: str, seed: int
) -> list[int]:
    # Per-document RNG: key on (seed, doc_id) so the SAME document gets the
    # SAME positions regardless of corpus-start/length/chunk-size/ordering.
    # Parallel runs on disjoint slices produce mergeable output.
    rng = random.Random(hashlib.sha256(f"{seed}|{doc_id}".encode()).digest())
    candidates = [
        i for i, tid in enumerate(token_ids)
        if i >= _MIN_POSITION and tid not in special_ids
    ]
    if not candidates:
        return []  # doc too short or all-special past _MIN_POSITION — skip
    k = min(n_positions, len(candidates))
    return rng.sample(candidates, k=k)


def _dataset_id(base_model: str, layer: int, corpus: str, corpus_slice: dict[str, int]) -> str:
    model_tag = base_model.split("/")[-1]
    h = hashlib.sha256(f"{base_model}|{layer}|{corpus}|{corpus_slice}".encode()).hexdigest()[:8]
    return f"base_{model_tag}_L{layer}_{h}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True, help="HF model name/path — also keys the extractor if not overridden")
    p.add_argument("--corpus", required=True, help="HF dataset name, e.g. HuggingFaceFW/fineweb")
    p.add_argument("--corpus-config", default=None, help="HF dataset config name")
    p.add_argument("--corpus-split", default="train")
    p.add_argument("--corpus-start", type=int, default=0)
    p.add_argument("--corpus-length", type=int, required=True, help="number of documents to process")
    p.add_argument("--text-column", default="text")
    p.add_argument("--layer-index", type=int, required=True)
    p.add_argument("--positions-per-doc", type=int, default=10)
    p.add_argument("--chunk-size", type=int, default=256, help="docs per extraction call — also the parquet write granularity")
    p.add_argument("--no-text", action="store_true",
                   help="skip per-position tokenizer.decode + the detokenized_text_truncated "
                        "column (write empty strings). For activation-only extraction where "
                        "the text is never used — avoids many decode calls and their RAM buffer.")
    p.add_argument("--chunks-per-file", type=int, default=0,
                   help="crash-safety: if >0, write durable part-files every N chunks "
                        "(atomically published) and RESUME from the last completed part on "
                        "restart, then concatenate into the single output parquet. 0 = legacy "
                        "single-writer (mid-run crash loses the whole shard). Local output only.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extractor-cls", default="nla.datagen.extractors.HFExtractor")
    p.add_argument("--extractor-kwargs", default=None, help="JSON dict of extra kwargs for the extractor constructor")
    p.add_argument("--output", required=True, help="output parquet path (local or s3://... depending on storage backend)")
    add_storage_args(p)
    args = p.parse_args()

    storage = make_storage(args)

    user_kwargs = parse_kwargs(args.extractor_kwargs)
    assert "model_name" not in user_kwargs, (
        "pass --base-model, not --extractor-kwargs '{\"model_name\": ...}'. "
        "If both are set, the kwargs value would silently win and the sidecar "
        "would record the wrong model (provenance poisoning)."
    )
    extractor_kwargs = {"model_name": args.base_model, **user_kwargs}
    extractor: ActivationExtractor = load_class(args.extractor_cls)(**extractor_kwargs)
    d_model = extractor.d_model
    tokenizer = extractor.tokenizer
    schema = _schema(d_model)

    special_ids = set(tokenizer.all_special_ids)
    # Pad tokens should NEVER appear in res.token_ids — the extractor slices
    # [:seq_len] to strip them. If they leak, the slice is broken and every
    # position index is suspect. Only check if pad is distinct from EOS
    # (otherwise a legit doc-end EOS would false-positive).
    pad_id_to_check = tokenizer.pad_token_id if (
        tokenizer.pad_token_id is not None
        and tokenizer.pad_token_id != tokenizer.eos_token_id
    ) else None

    # Allow a local parquet path as --corpus (skip the HF Hub roundtrip and the
    # `load_dataset("parquet")` builder that needs explicit data_files).
    import os
    if args.corpus.endswith(".parquet") and os.path.exists(args.corpus):
        ds = Dataset.from_parquet(args.corpus)
    else:
        ds = load_dataset(args.corpus, name=args.corpus_config, split=args.corpus_split)
    assert isinstance(ds, Dataset), (
        f"expected a concrete Dataset, got {type(ds).__name__}. "
        f"Pass an explicit split (e.g. --corpus-split train), not a streaming/dict dataset."
    )
    ds = ds.select(range(args.corpus_start, args.corpus_start + args.corpus_length))

    storage.ensure_parent(args.output)
    row_count = 0
    n_docs_skipped = 0
    n_docs_short_sampled = 0

    def _build_chunk_rows(chunk_start: int) -> dict[str, list[Any]]:
        """Forward one chunk of docs, sample positions, return the rows dict."""
        nonlocal n_docs_skipped, n_docs_short_sampled
        chunk = ds.select(range(chunk_start, min(chunk_start + args.chunk_size, len(ds))))
        results = extractor.extract(chunk[args.text_column], args.layer_index)
        rows: dict[str, list[Any]] = {k: [] for k in schema.names}
        for doc_offset, res in enumerate(results):
            doc_idx = args.corpus_start + chunk_start + doc_offset
            doc_id = f"{args.corpus}:{args.corpus_split}:{doc_idx}"
            if pad_id_to_check is not None:
                assert pad_id_to_check not in res.token_ids, (
                    f"pad_token_id {pad_id_to_check} found in res.token_ids for {doc_id}. "
                    f"The extractor's [:seq_len] slice is broken — all position indices "
                    f"are now suspect. Fix the extractor."
                )
            positions = _sample_positions(
                res.token_ids, args.positions_per_doc, special_ids, doc_id, args.seed
            )
            if not positions:
                n_docs_skipped += 1
                continue
            if len(positions) < args.positions_per_doc:
                n_docs_short_sampled += 1
            for pos in positions:
                vec = res.hidden_states[pos]  # raw — normalization is training-side
                n_raw_tokens = pos + 1
                rows["n_raw_tokens"].append(n_raw_tokens)
                rows["detokenized_text_truncated"].append(
                    "" if args.no_text
                    else tokenizer.decode(res.token_ids[:n_raw_tokens], skip_special_tokens=True)
                )
                rows["activation_vector"].append(vec.tolist())
                rows["activation_layer"].append(args.layer_index)
                rows["doc_id"].append(doc_id)
        return rows

    chunk_starts = list(range(0, len(ds), args.chunk_size))
    parts_dir: Path | None = None

    if args.chunks_per_file > 0:
        # Crash-safe path: write durable part-files (one per N chunks), publish
        # each atomically (.tmp -> os.replace), and skip already-complete parts
        # on restart. A mid-run kill loses at most the in-progress part.
        assert "://" not in args.output, "--chunks-per-file requires a local output path"
        parts_dir = Path(f"{args.output}.parts")
        parts_dir.mkdir(parents=True, exist_ok=True)

        def _part_rows(p: Path) -> int | None:
            try:
                return pq.ParquetFile(str(p)).metadata.num_rows
            except Exception:
                return None  # missing or torn write

        for gi in range(0, len(chunk_starts), args.chunks_per_file):
            group = chunk_starts[gi : gi + args.chunks_per_file]
            part = parts_dir / f"part_{group[0]:08d}.parquet"
            done_rows = _part_rows(part)
            if done_rows is not None:
                row_count += done_rows
                continue  # resume: this part is already on disk
            if part.exists():
                part.unlink()  # torn write from a prior crash — recompute
            tmp = parts_dir / f"part_{group[0]:08d}.parquet.tmp"
            with pq.ParquetWriter(str(tmp), schema) as writer:
                for chunk_start in tqdm(group, desc=f"part@{group[0]}"):
                    rows = _build_chunk_rows(chunk_start)
                    writer.write_table(pa.Table.from_pydict(rows, schema=schema))
                    row_count += len(rows["doc_id"])
            os.replace(tmp, part)  # atomic publish

        # Concatenate durable parts -> the single shard parquet the merge/launch
        # paths expect, streamed row-group by row-group (bounded memory).
        with pq.ParquetWriter(storage.open_write(args.output), schema) as writer:
            for part in sorted(parts_dir.glob("part_*.parquet")):
                pf = pq.ParquetFile(str(part))
                for rg in range(pf.num_row_groups):
                    writer.write_table(pf.read_row_group(rg))
    else:
        with pq.ParquetWriter(storage.open_write(args.output), schema) as writer:
            for chunk_start in tqdm(chunk_starts, desc="chunks"):
                rows = _build_chunk_rows(chunk_start)
                writer.write_table(pa.Table.from_pydict(rows, schema=schema))
                row_count += len(rows["doc_id"])

    corpus_slice = {"start": args.corpus_start, "length": args.corpus_length}
    meta = NLADatasetMeta(
        dataset_id=_dataset_id(args.base_model, args.layer_index, args.corpus, corpus_slice),
        stage="base",
        row_count=row_count,
        extraction=NLAExtractionMeta(
            base_model=args.base_model,
            d_model=d_model,
            layer_index=args.layer_index,
            norm="none",  # raw — normalization is training-side
            corpus=args.corpus,
            corpus_slice=corpus_slice,
            positions_per_doc=args.positions_per_doc,
        ),
        created_by="nla.datagen.stage0_extract",
    )
    write_sidecar(storage, args.output, meta)
    # Only now that BOTH the merged parquet and its sidecar exist (the shard is
    # "complete" per the launcher's check) is it safe to drop the part-files.
    if parts_dir is not None:
        shutil.rmtree(parts_dir, ignore_errors=True)
    print(f"wrote {row_count} rows → {args.output}")
    print(f"  skipped {n_docs_skipped} docs (too short / all-special past position {_MIN_POSITION})")
    print(f"  short-sampled {n_docs_short_sampled} docs (fewer than {args.positions_per_doc} valid positions)")
    print(f"sidecar → {args.output}.nla_meta.yaml")


if __name__ == "__main__":
    main()
