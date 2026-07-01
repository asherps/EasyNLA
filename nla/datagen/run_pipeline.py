"""Full-pipeline orchestrator — run stages 0→3 from a single YAML config.

Output paths are derived from config.output_dir — no manual path threading.
Each stage shells out to its existing CLI (so you can always re-run a single
stage by hand with the printed command).

    python -m nla.datagen.run_pipeline --config configs/datagen/qwen7b_fineweb_1M.yaml
    python -m nla.datagen.run_pipeline --config ... --stages 2,3   # resume partway
"""

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from nla.datagen.model_presets import resolve as resolve_model_preset


_TRAIN_STAGES = ("av_sft", "ar_sft", "rl")


def _paths(output_dir: str) -> dict[str, str]:
    return {
        "base": f"{output_dir}/base.parquet",
        "av_sft_raw": f"{output_dir}/splits/av_sft_raw.parquet",
        "ar_sft_raw": f"{output_dir}/splits/ar_sft_raw.parquet",
        "rl_raw": f"{output_dir}/splits/rl_raw.parquet",
        "av_sft_explained": f"{output_dir}/splits/av_sft_explained.parquet",
        "ar_sft_explained": f"{output_dir}/splits/ar_sft_explained.parquet",
        "av_sft": f"{output_dir}/av_sft.parquet",
        "ar_sft": f"{output_dir}/ar_sft.parquet",
        "rl": f"{output_dir}/rl.parquet",
        "av_sft_shuf": f"{output_dir}/av_sft_shuf.parquet",
        "ar_sft_shuf": f"{output_dir}/ar_sft_shuf.parquet",
        "rl_shuf": f"{output_dir}/rl_shuf.parquet",
    }


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def _storage_args(cfg: dict[str, Any]) -> list[str]:
    args = ["--storage-cls", cfg["storage_cls"]]
    if cfg.get("storage_kwargs"):
        args += ["--storage-kwargs", json.dumps(cfg["storage_kwargs"])]
    return args


def _stage0(cfg: dict[str, Any], p: dict[str, str]) -> None:
    s0 = cfg["stage0"]
    common = [
        "--base-model", cfg["base_model"],
        "--corpus", cfg["corpus"]["name"],
        "--corpus-split", cfg["corpus"]["split"],
        "--corpus-start", str(cfg["corpus"]["start"]),
        "--corpus-length", str(cfg["corpus"]["length"]),
        "--text-column", cfg["corpus"].get("text_column", "text"),
        "--layer-index", str(cfg["layer_index"]),
        "--positions-per-doc", str(s0["positions_per_doc"]),
        "--chunk-size", str(s0["chunk_size"]),
        "--seed", str(s0["seed"]),
        "--output", p["base"],
        *_storage_args(cfg),
    ]
    if cfg["corpus"].get("config"):
        common += ["--corpus-config", cfg["corpus"]["config"]]
    if s0.get("extractor_kwargs"):
        common += ["--extractor-kwargs", json.dumps(s0["extractor_kwargs"])]

    if s0.get("multigpu", False):
        repo = Path(__file__).resolve().parents[2]
        _run(["bash", str(repo / "scripts/datagen/stage0_multigpu.sh"), *common])
    else:
        _run([sys.executable, "-m", "nla.datagen.stage0_extract", *common])


def _stage1(cfg: dict[str, Any], p: dict[str, str]) -> None:
    s1 = cfg["stage1"]
    _run([
        sys.executable, "-m", "nla.datagen.stage1_split",
        "--base", p["base"],
        "--av-sft-frac", str(s1["av_sft_frac"]),
        "--ar-sft-frac", str(s1["ar_sft_frac"]),
        "--rl-frac", str(s1["rl_frac"]),
        "--seed", str(s1["seed"]),
        "--output-dir", f"{cfg['output_dir']}/splits",
        *_storage_args(cfg),
    ])


def _stage2(cfg: dict[str, Any], p: dict[str, str]) -> None:
    s2 = cfg["stage2"]
    args = [
        "--provider-cls", s2["provider_cls"],
        "--chunk-size", str(s2["chunk_size"]),
        *_storage_args(cfg),
    ]
    if s2.get("provider_kwargs"):
        args += ["--provider-kwargs", json.dumps(s2["provider_kwargs"])]
    for path in s2.get("cache_from", []):
        args += ["--cache-from", path]
    if s2.get("cache_storage_cls"):
        args += ["--cache-storage-cls", s2["cache_storage_cls"]]
    for side in ("av_sft", "ar_sft"):
        _run([
            sys.executable, "-m", "nla.datagen.stage2_api_explain",
            "--input", p[f"{side}_raw"],
            "--output", p[f"{side}_explained"],
            *args,
        ])


def _stage3(cfg: dict[str, Any], p: dict[str, str]) -> None:
    s3 = cfg["stage3"]
    debug_flag = "--keep-debug-metadata" if s3["keep_debug_metadata"] else "--no-keep-debug-metadata"
    storage = _storage_args(cfg)
    _run([sys.executable, "-m", "nla.datagen.stage3_build",
          "--input", p["av_sft_explained"], "--stage", "av_sft", "--output", p["av_sft"], debug_flag, *storage])
    _run([sys.executable, "-m", "nla.datagen.stage3_build",
          "--input", p["ar_sft_explained"], "--stage", "ar_sft", "--output", p["ar_sft"], debug_flag, *storage])
    _run([sys.executable, "-m", "nla.datagen.stage3_build",
          "--input", p["rl_raw"], "--stage", "rl", "--output", p["rl"], debug_flag, *storage])


def _shuffle(cfg: dict[str, Any], p: dict[str, str]) -> None:
    sh = cfg["shuffle"]
    storage = _storage_args(cfg)
    for side in _TRAIN_STAGES:
        _run([
            sys.executable, "-m", "nla.datagen.stage_shuffle",
            "--input", p[side],
            "--output", p[f"{side}_shuf"],
            "--seed", str(sh["seed"]),
            *storage,
        ])


_STAGES = {"0": _stage0, "1": _stage1, "2": _stage2, "3": _stage3, "shuffle": _shuffle}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--stages", default=None,
                    help="comma-separated stages to run, e.g. '2,3' to resume. "
                         "Default: 0,1,2,3 plus shuffle if config.shuffle.enabled")
    ap.add_argument("--override", nargs="*", default=[],
                    help="dotted-key overrides, e.g. 'corpus.start=25000 output_dir=/tmp/n1'")
    args = ap.parse_args()

    cfg = resolve_model_preset(yaml.safe_load(Path(args.config).read_text()))
    for ov in args.override:
        key, _, val = ov.partition("=")
        d = cfg
        *path, leaf = key.split(".")
        for k in path:
            d = d.setdefault(k, {})
        d[leaf] = yaml.safe_load(val)
    p = _paths(cfg["output_dir"])

    if args.stages is None:
        stages = ["0", "1", "2", "3"]
        if cfg.get("shuffle", {}).get("enabled", False):
            stages.append("shuffle")
    else:
        stages = args.stages.split(",")
    for s in stages:
        assert s in _STAGES, f"unknown stage {s!r}, valid: {sorted(_STAGES)}"

    print(f"=== pipeline: {args.config} → {cfg['output_dir']} ===")
    print(f"    base_model={cfg['base_model']} layer={cfg['layer_index']}")
    print(f"    corpus={cfg['corpus']['name']} docs={cfg['corpus']['length']} positions/doc={cfg['stage0']['positions_per_doc']}")
    print(f"    split={cfg['stage1']['av_sft_frac']}/{cfg['stage1']['ar_sft_frac']}/{cfg['stage1']['rl_frac']}")
    print(f"    stages={stages}")

    for s in stages:
        print(f"\n{'='*20} STAGE {s} {'='*20}")
        _STAGES[s](cfg, p)

    print(f"\n=== done ===")
    suffix = "_shuf" if "shuffle" in stages else ""
    final_paths = [p[k + suffix] for k in _TRAIN_STAGES]
    for k, fp in zip(_TRAIN_STAGES, final_paths, strict=True):
        print(f"  {k}: {fp}")

    upload = cfg.get("upload")
    if upload:
        # Bucket root, cp tool, detach — all live in the plugin fn. We just
        # hand over the file list + a subdir name. *_explained.parquet go up
        # too — they're the cache-from source for future runs with a different
        # model but same tokenizer (detokenized_text_truncated is the join key,
        # always present in explained regardless of keep_debug_metadata).
        mod, _, name = upload["fn"].rpartition(".")
        upload_fn = getattr(importlib.import_module(mod), name)
        to_upload = [f for fp in final_paths for f in (fp, f"{fp}.nla_meta.yaml")]
        for side in ("av_sft", "ar_sft"):
            to_upload += [p[f"{side}_explained"], f"{p[f'{side}_explained']}.nla_meta.yaml"]
        upload_fn(to_upload, upload["subdir"])


if __name__ == "__main__":
    main()
