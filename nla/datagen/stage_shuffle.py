"""Row shuffle — permute rows of any parquet deterministically.

Run after stage3 to randomize training order (docs are already in random
order from fineweb, but positions-within-doc are sequential from stage0).

In-memory: reads full table, permutes via pyarrow's take(), writes.
Memory = dataset size in columnar form (1M rows × d=4096 float32 ≈ 16GB —
well within typical dev-box RAM). Seeded + keyed on dataset_id for
reproducibility.
"""

import argparse
import hashlib
import random
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen._common import add_storage_args, make_storage
from nla.datagen.sidecar import read_sidecar, write_sidecar

# pyarrow 18.1.0 ChunkedArray.take() on list types has a SILENT uint32
# byte-offset overflow when the source values buffer exceeds 2^32 bytes
# (4 GiB). For float32 at d=3584 that's row 299,593. No error — take()
# combines the source to a single chunk, computes byte offsets in uint32,
# and quietly wraps. Hit on the 100k RL run (500k rows = 6.7 GiB → ~40% of
# output rows got vectors from (offset mod 2^32)). This is DIFFERENT from
# the int32 element-offset limit (2.1B elements) that pq.read_table catches.
#
# We don't yet trust take() on FixedSizeList either (same values buffer,
# might be same gather kernel). So: numpy fancy-indexing for anything over
# 2 GiB. numpy uses int64 everywhere.
_TAKE_VALUES_BYTES_LIMIT = 2 * 2**30


def _values_nbytes(col: pa.ChunkedArray) -> int:
    return sum(ch.values.nbytes for ch in col.chunks)


def _take_fixed_size_list_via_numpy(col: pa.ChunkedArray, perm: np.ndarray) -> pa.Array:
    assert pa.types.is_fixed_size_list(col.type), (
        f"expected fixed_size_list, got {col.type}. Variable-length list "
        f"can't use this path (ragged → reshape fails). stage0 writes "
        f"activation_vector as fixed_size_list[d_model]; if you're seeing "
        f"plain list here, the input parquet predates that change."
    )
    n = len(col)
    d = col.type.list_size
    flat = col.combine_chunks().values.to_numpy(zero_copy_only=False)
    shuffled_2d = flat.reshape(n, d)[perm]
    return pa.FixedSizeListArray.from_arrays(
        pa.array(shuffled_2d.reshape(-1), type=col.type.value_type), d
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    add_storage_args(p)
    args = p.parse_args()

    storage = make_storage(args)
    in_meta = read_sidecar(storage, args.input)

    # Key on (seed, dataset_id) so same input + same seed → same permutation.
    rng = random.Random(hashlib.sha256(f"{args.seed}|{in_meta.dataset_id}".encode()).digest())

    table = pq.read_table(storage.open_read(args.input))
    perm = list(range(table.num_rows))
    rng.shuffle(perm)

    perm_pa = pa.array(perm, type=pa.int64())
    perm_np = np.asarray(perm, dtype=np.int64)
    out_cols: list[pa.Array | pa.ChunkedArray] = []
    for name in table.column_names:
        col = table.column(name)
        if pa.types.is_fixed_size_list(col.type) and _values_nbytes(col) > _TAKE_VALUES_BYTES_LIMIT:
            print(f"  {name}: {_values_nbytes(col) / 2**30:.2f} GiB values — numpy gather (pyarrow take() unsafe at this scale)")
            out_cols.append(_take_fixed_size_list_via_numpy(col, perm_np))
        else:
            out_cols.append(col.take(perm_pa))
    shuffled = pa.table(out_cols, names=table.column_names)

    storage.ensure_parent(args.output)
    pq.write_table(shuffled, storage.open_write(args.output), row_group_size=65536)

    out_meta = replace(
        in_meta,
        dataset_id=f"{in_meta.dataset_id}__shuf{args.seed}",
        row_count=table.num_rows,
        parent_datasets=[in_meta.dataset_id],
        created_by="nla.datagen.stage_shuffle",
        created_at="",
        git_commit="",
    )
    write_sidecar(storage, args.output, out_meta)
    print(f"shuffled {table.num_rows} rows → {args.output}")


if __name__ == "__main__":
    main()
