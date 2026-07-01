"""Stage 1: Three-way split — base.parquet → {av_sft,ar_sft,rl}_raw.parquet.

Partition at the DOCUMENT level (by doc_id), not the row level. Stage 0
samples ~N positions per document; row-level split would leak the same
document's context across AO/AR/RL subsets, contaminating the SL ↔ RL
boundary.
"""

import argparse
import random
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen._common import add_storage_args, make_storage
from nla.datagen.sidecar import read_sidecar, write_sidecar


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="base.parquet from stage0")
    p.add_argument("--av-sft-frac", type=float, default=0.3)
    p.add_argument("--ar-sft-frac", type=float, default=0.3)
    p.add_argument("--rl-frac", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    add_storage_args(p)
    args = p.parse_args()

    storage = make_storage(args)

    fracs = (args.av_sft_frac, args.ar_sft_frac, args.rl_frac)
    assert all(f >= 0 for f in fracs), f"fractions must be non-negative, got {fracs}"
    total = sum(fracs)
    assert abs(total - 1.0) < 1e-6, f"fractions must sum to 1.0, got {total}"

    base_meta = read_sidecar(storage, args.base)
    assert base_meta.stage == "base", f"expected stage=base, got stage={base_meta.stage!r}"

    # Read ONLY doc_id to compute the split — avoids pyarrow int32 list-offset
    # overflow on activation_vector (1M rows × d=3584 = 3.6B elements > 2.1B).
    pf = pq.ParquetFile(storage.open_read(args.base))
    doc_id_col = pf.read(columns=["doc_id"]).column("doc_id").to_pylist()

    # sorted() makes the subsequent shuffle deterministic — set iteration
    # order depends on hash seed / Python version. Without this, --seed would
    # not reproduce the same split across environments.
    doc_ids = sorted(set(doc_id_col))
    rng = random.Random(args.seed)
    rng.shuffle(doc_ids)

    n_docs = len(doc_ids)
    n_ao = int(n_docs * args.av_sft_frac)
    n_ar = int(n_docs * args.ar_sft_frac)
    buckets = {
        "av_sft": set(doc_ids[:n_ao]),
        "ar_sft": set(doc_ids[n_ao : n_ao + n_ar]),
        "rl": set(doc_ids[n_ao + n_ar :]),
    }

    # Stream: filter each row-group into 3 writers. Individual buckets are
    # small enough (≤500k rows × d ≈ 1.8B elements) to stay under the int32
    # limit at write time; only the full table exceeded it.
    schema = pf.schema_arrow
    out_paths = {s: f"{args.output_dir.rstrip('/')}/{s}_raw.parquet" for s in buckets}
    for path in out_paths.values():
        storage.ensure_parent(path)
    writers = {s: pq.ParquetWriter(storage.open_write(out_paths[s]), schema) for s in buckets}
    row_counts = {s: 0 for s in buckets}

    # iter_batches chunks at batch_size regardless of how row-groups were
    # written on disk — avoids overflow even when merge_base wrote one giant
    # row group. 65536 rows × d=3584 ≈ 235M elements, well under int32.
    for batch in pf.iter_batches(batch_size=65536):
        batch_docs = batch.column("doc_id").to_pylist()
        for stage, bucket_ids in buckets.items():
            mask = pa.array([d in bucket_ids for d in batch_docs], type=pa.bool_())
            subset = batch.filter(mask)
            if subset.num_rows > 0:
                writers[stage].write_table(pa.Table.from_batches([subset]))
                row_counts[stage] += subset.num_rows

    for w in writers.values():
        w.close()

    for stage, bucket_ids in buckets.items():
        sub_meta = replace(
            base_meta,
            dataset_id=f"{base_meta.dataset_id}__{stage}_raw",
            stage="base",  # still base-schema rows — stage3 produces the real av_sft/ar_sft/rl parquets
            row_count=row_counts[stage],
            parent_datasets=[base_meta.dataset_id],
            created_by="nla.datagen.stage1_split",
            created_at="",
            git_commit="",
        )
        write_sidecar(storage, out_paths[stage], sub_meta)
        print(f"{stage}: {len(bucket_ids)} docs → {row_counts[stage]} rows → {out_paths[stage]}")


if __name__ == "__main__":
    main()
