"""Re-export shim — storage abstraction moved to nla/storage.py (shared
between datagen and training). Import from nla.storage directly in new code."""

from nla.storage import LocalStorage, Storage, dataset_sidecar_path

# Backward-compat alias — datagen code used this name.
sidecar_path_str = dataset_sidecar_path

__all__ = ["Storage", "LocalStorage", "sidecar_path_str", "dataset_sidecar_path"]
