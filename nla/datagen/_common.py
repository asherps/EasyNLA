"""Shared helpers for stage CLIs."""

import argparse
import importlib
import json
from typing import Any

from transformers import AutoTokenizer

from nla.datagen.storage import Storage


def load_class(path: str) -> type:
    """Load a class by import path: 'module.submodule.ClassName'."""
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, attr)
    assert isinstance(cls, type), f"{path!r} resolved to {cls!r}, expected a class"
    return cls


def parse_kwargs(kwargs_json: str | None) -> dict[str, Any]:
    return json.loads(kwargs_json) if kwargs_json else {}


def add_storage_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--storage-cls", default="nla.datagen.storage.LocalStorage",
                   help="Storage backend class (subclass nla.datagen.storage.Storage for S3/GCS)")
    p.add_argument("--storage-kwargs", default=None,
                   help="JSON dict of kwargs for the storage constructor")


def make_storage(args: argparse.Namespace) -> Storage:
    return load_class(args.storage_cls)(**parse_kwargs(args.storage_kwargs))


def load_tokenizer(model_name: str) -> Any:
    """Central tokenizer loader. Stage0's extractor and stage3 both go through here."""
    return AutoTokenizer.from_pretrained(model_name)
