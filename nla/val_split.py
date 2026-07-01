"""Doc-hash train/val split, shared by the RL trainers and the held-out evals.

Why not a row boundary: rl_shuf.parquet is row-SHUFFLED, so each doc's ~10 rows
are scattered uniformly. A row-index split ("train on the first 90%, eval past
it") leaves ~zero docs fully unseen (P = 0.1^rows_per_doc per doc), so the
doc-disjoint eval filters found 0 rows and every base_fve/hallucination eval
was silently empty (nan) on auto-split runs. The split must be BY DOC:
hold out a deterministic pseudo-random ~val_permille/1000 of doc_ids (crc32 —
seed/order-free, no doc table needed), train on every row of the other docs.
"""

import zlib


def val_doc_permille(val_rows: int, total_rows: int) -> int:
    """Permille of docs to hold out so the val split is ~val_rows rows."""
    return max(1, min(500, round(1000 * val_rows / max(1, total_rows))))


def is_val_doc(doc_id, permille: int) -> bool:
    """True iff doc_id falls in the held-out val bucket (~permille/1000 of docs)."""
    return (zlib.crc32(str(doc_id).encode("utf-8")) % 1000) < permille
