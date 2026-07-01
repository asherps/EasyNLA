"""Read/write NLA dataset sidecar YAML files.

The sidecar schema is defined by nla/schema.py (shared single source of truth).
Every parquet gets a `{parquet_path}.nla_meta.yaml` written alongside it.
Training-side code (nla/config.py) reads these and asserts against the live
tokenizer — this is the contract that catches prompt-format drift and
token-ID drift before they silently poison a run.

NLATokenMeta and sidecar path resolution live in nla.schema (shared with
nla/config.py so datagen-writes and training-reads use identical logic).
"""

import dataclasses
import datetime
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from nla.datagen.storage import Storage, sidecar_path_str
from nla.schema import NLATokenMeta, sidecar_path_for

SCHEMA_VERSION = 1


@dataclass
class NLAExtractionMeta:
    base_model: str
    d_model: int
    layer_index: int
    norm: str  # data-gen always writes "none" (raw vectors). Training decides
    # how/whether to normalize at load time. Field stays for forward-compat
    # and so training can assert on what it's reading.
    corpus: str
    corpus_slice: dict[str, int]
    positions_per_doc: int


@dataclass
class NLAApiSummaryMeta:
    model: str
    max_tokens: int
    temperature: float
    instruction_prompt: str


@dataclass
class NLADatasetMeta:
    dataset_id: str
    stage: str  # base | av_sft | ar_sft | rl
    row_count: int
    extraction: NLAExtractionMeta
    kind: str = "nla_dataset"
    schema_version: int = SCHEMA_VERSION
    keep_debug_metadata: bool = True
    tokens: NLATokenMeta | None = None
    prompt_templates: dict[str, str] = field(default_factory=dict)
    api_summaries: NLAApiSummaryMeta | None = None
    parent_datasets: list[str] = field(default_factory=list)
    created_at: str = ""
    created_by: str = ""
    git_commit: str = ""


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""


_TRAINING_STAGES = {"av_sft", "ar_sft", "rl"}


def serialize_sidecar(meta: NLADatasetMeta) -> str:
    """Validate + serialize to YAML string. Pure — no IO."""
    if meta.stage in _TRAINING_STAGES:
        assert meta.tokens is not None, (
            f"stage={meta.stage!r} requires NLATokenMeta — training-side config.py "
            f"reads injection/pm token IDs from it. Set meta.tokens before writing."
        )
        assert meta.prompt_templates.get("actor"), (
            f"stage={meta.stage!r} requires prompt_templates['actor'] — training MUST use "
            f"the exact template or injection position drifts."
        )
        if meta.stage == "ar_sft":
            assert meta.tokens.critic_suffix_ids is not None, (
                "stage=ar_sft requires critic_suffix_ids for last-token extraction verification"
            )
            assert meta.prompt_templates.get("critic"), (
                "stage=ar_sft requires prompt_templates['critic']"
            )
    if not meta.created_at:
        meta.created_at = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    if not meta.git_commit:
        meta.git_commit = _git_commit()

    # asdict() gives None for unset nested dataclasses; drop None-valued top-level
    # keys so the YAML stays readable and downstream readers don't have to
    # special-case "tokens: null" vs "tokens absent".
    d = {k: v for k, v in asdict(meta).items() if v is not None}
    return yaml.safe_dump(d, sort_keys=False)


def deserialize_sidecar(text: str) -> NLADatasetMeta:
    """Parse + validate from YAML string. Pure — no IO."""
    d = yaml.safe_load(text)
    assert d["kind"] == "nla_dataset", f"not an NLA dataset sidecar: kind={d['kind']!r}"
    assert d["schema_version"] == SCHEMA_VERSION, (
        f"sidecar schema version {d['schema_version']} != expected {SCHEMA_VERSION}"
    )

    d["extraction"] = NLAExtractionMeta(**d["extraction"])
    if d.get("tokens") is not None:
        d["tokens"] = NLATokenMeta(**d["tokens"])
    if d.get("api_summaries") is not None:
        d["api_summaries"] = NLAApiSummaryMeta(**d["api_summaries"])

    known = {f.name for f in dataclasses.fields(NLADatasetMeta)}
    return NLADatasetMeta(**{k: v for k, v in d.items() if k in known})


def write_sidecar(storage: Storage, parquet_path: str, meta: NLADatasetMeta) -> None:
    storage.write_text(sidecar_path_str(parquet_path), serialize_sidecar(meta))


def read_sidecar(storage: Storage, parquet_path: str) -> NLADatasetMeta:
    return deserialize_sidecar(storage.read_text(sidecar_path_str(parquet_path)))


# Back-compat wrappers for code that still passes pathlib.Path directly
# (training-side config.py uses these — keep until that's refactored).
def write_sidecar_local(parquet_path: Path, meta: NLADatasetMeta) -> None:
    sidecar_path_for(parquet_path).write_text(serialize_sidecar(meta))


def read_sidecar_local(parquet_path: Path) -> NLADatasetMeta:
    return deserialize_sidecar(sidecar_path_for(parquet_path).read_text())
