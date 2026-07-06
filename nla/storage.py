"""Storage backend abstraction for parquet + sidecar IO.

Used by both datagen (write parquets + dataset sidecars) and training
(read parquets + dataset sidecars). Swap backend via --storage-cls.

Model-checkpoint sidecars are NOT routed through this — HF from_pretrained
and FSDP DCP are local-fs-only, so checkpoint IO stays pathlib.
"""

import hashlib
import importlib
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

import pyarrow as pa

from nla.schema import SIDECAR_SUFFIX


def dataset_sidecar_path(parquet_path: str) -> str:
    """Sidecar path for a parquet — string-only, no filesystem access.

    Strips the `@[slice]` path syntax. Dataset sidecars are ALWAYS the
    suffix-append form ({parquet}.nla_meta.yaml); the directory form
    ({dir}/nla_meta.yaml) is for model checkpoints which are local-only
    and handled by schema.sidecar_path_for.
    """
    return parquet_path.split("@[")[0] + SIDECAR_SUFFIX


class Storage(ABC):
    """Abstract storage backend. Methods take path strings; backends decide
    what they resolve to (local Path, S3 key, etc.).

    pyarrow's ParquetWriter/ParquetFile accept file-like objects, so
    open_write/open_read are enough — no need to expose the backend's
    path-like type.
    """

    @abstractmethod
    def open_write(self, path: str) -> BinaryIO: ...

    @abstractmethod
    def open_read(self, path: str) -> BinaryIO: ...

    @abstractmethod
    def write_text(self, path: str, content: str) -> None: ...

    @abstractmethod
    def read_text(self, path: str) -> str: ...

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def ensure_parent(self, path: str) -> None:
        """Make parent directories if applicable. No-op for object stores."""


class LocalStorage(Storage):
    """pathlib.Path-backed storage. Default for standalone/dev use."""

    def open_write(self, path: str) -> BinaryIO:
        # pa.OSFile, NOT Path.open("wb") — pyarrow.ParquetWriter keeps a ref
        # to its sink past close(); a plain Python file handle never gets
        # flushed (footer lost — 767MB of chunks but no PAR1 footer magic).
        # Observed w/ pyarrow 18.1.0: `with ParquetWriter(Path(p).open("wb"), ...)`
        # writes 0 bytes for small tables, truncated for large. pa.OSFile is
        # pyarrow-managed and flushes correctly on writer.close(). Cost us
        # 45 min of 8×H100 stage0 before diagnosis (Mar 10 2026).
        return pa.OSFile(str(path), "wb")  # type: ignore[return-value]

    def open_read(self, path: str) -> BinaryIO:
        return pa.OSFile(str(path), "rb")  # type: ignore[return-value]

    def write_text(self, path: str, content: str) -> None:
        Path(path).write_text(content)

    def read_text(self, path: str) -> str:
        return Path(path).read_text()

    def exists(self, path: str) -> bool:
        return Path(path).exists()

    def ensure_parent(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def is_remote(path: str) -> bool:
    """Heuristic: paths with :// are remote (s3://, gs://, etc.)."""
    return "://" in path.split("@[")[0]


def _load_storage(cls_path: str) -> Storage:
    mod_path, _, cls_name = cls_path.rpartition(".")
    cls = getattr(importlib.import_module(mod_path), cls_name)
    assert isinstance(cls, type) and issubclass(cls, Storage), (
        f"{cls_path!r} must be a Storage subclass"
    )
    return cls()


def _local_cache_path(remote_path: str, cache_dir: str) -> tuple[str, str]:
    """Deterministic local cache path for a remote path. Returns (local_base, slice_suffix)."""
    base = remote_path.split("@[")[0]
    slice_suffix = remote_path[len(base):]  # "" or "@[start:end]"
    cache_key = hashlib.sha256(base.encode()).hexdigest()[:16]
    local_base = str(Path(cache_dir) / cache_key / Path(base).name)
    return local_base, slice_suffix


def fetch_sidecar_to_local_cache(
    remote_parquet_path: str,
    *,
    storage_cls: str,
    cache_dir: str = "/tmp/nla_fetch_cache",
) -> str:
    """Fetch ONLY the sidecar for a remote parquet — no parquet download.

    Used by training actors (different Ray process from NLADataSource) that
    need token IDs + templates but not the data. Returns the local parquet
    path (sidecar is at {path}.nla_meta.yaml) so sidecar_path_for works.
    """
    local_base, _ = _local_cache_path(remote_parquet_path, cache_dir)
    local_sc = dataset_sidecar_path(local_base)
    if Path(local_sc).exists():
        return local_base

    remote = _load_storage(storage_cls)
    remote_sc = dataset_sidecar_path(remote_parquet_path.split("@[")[0])
    assert remote.exists(remote_sc), (
        f"no sidecar at {remote_sc!r} — dataset was built without write_sidecar?"
    )
    LocalStorage().ensure_parent(local_sc)
    LocalStorage().write_text(local_sc, remote.read_text(remote_sc))
    return local_base


def fetch_to_local_cache(
    remote_path: str,
    *,
    storage_cls: str,
    cache_dir: str = "/tmp/nla_fetch_cache",
) -> str:
    """Copy a remote parquet + sidecar to a deterministic local cache path.

    Cache key is hash(remote_path) so repeat runs are idempotent — if the
    local file already exists, skip the fetch. No size/mtime check; delete
    the cache dir to force a refetch.

    Returns the local parquet path (with @[slice] suffix preserved if present
    on the remote path — the parquet reader applies the slice locally).
    """
    local_base, slice_suffix = _local_cache_path(remote_path, cache_dir)

    if Path(local_base).exists():
        return local_base + slice_suffix

    remote = _load_storage(storage_cls)
    local = LocalStorage()
    local.ensure_parent(local_base)

    base = remote_path.split("@[")[0]
    # tmp + atomic rename: a preempted/killed copy must not leave a truncated
    # file that the bare exists() check above treats as a valid cache hit.
    _tmp = local_base + ".tmp"
    with remote.open_read(base) as src, local.open_write(_tmp) as dst:
        shutil.copyfileobj(src, dst)
    os.replace(_tmp, local_base)

    remote_sc = dataset_sidecar_path(base)
    if remote.exists(remote_sc):
        local.write_text(dataset_sidecar_path(local_base), remote.read_text(remote_sc))

    return local_base + slice_suffix
