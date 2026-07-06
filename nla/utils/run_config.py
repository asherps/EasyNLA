"""Lightweight YAML run-config layer for the trainers.

A run is defined by a YAML whose keys are argparse *dests* (underscores), e.g.::

    # configs/rl_vllm.yaml
    lr: 1.0e-4
    critic_lr: 8.0e-5
    batch_prompts: 256

Precedence: hardcoded argparse defaults  <  YAML  <  explicit CLI flags. So you
keep CLI overrides for quick one-offs (``--config base.yaml --lr 2e-5``) while
the canonical run lives in a single file you check into git.

Trainer usage::

    p = argparse.ArgumentParser()
    add_config_arg(p)
    ...  # p.add_argument(...) for everything else
    apply_config_defaults(p)         # YAML -> argparse defaults
    args = p.parse_args()            # CLI overrides YAML
    ...
    save_resolved_config(args, args.save_dir)   # snapshot merged config to the ckpt dir
"""
import argparse
from pathlib import Path

import yaml


def add_config_arg(parser):
    parser.add_argument(
        "--config", default=None,
        help="YAML run config; its keys (argparse dests, underscores) become "
             "defaults. CLI flags still override. The fully-resolved config is "
             "saved to <save_dir>/run_config.yaml for reproducibility.",
    )


def apply_config_defaults(parser):
    """Pre-parse ``--config``, load the YAML, set it as parser defaults.

    Validates every YAML key against the parser's dests so a typo
    (``learning_rate`` vs ``lr``, or a dashed key) fails loudly instead of being
    silently ignored. Returns the config path (or None).
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args()
    if not known.config:
        return None
    with open(known.config) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"--config {known.config}: top level must be a mapping (key: value)")
    valid = {a.dest for a in parser._actions if a.dest not in ("help", "config")}
    unknown = sorted(set(cfg) - valid)
    if unknown:
        raise SystemExit(
            f"--config {known.config}: unknown keys {unknown}.\n"
            f"Keys must be argparse dests with UNDERSCORES (e.g. batch_prompts, not "
            f"batch-prompts). Valid keys: {sorted(valid)}"
        )
    # Coerce YAML values through the matching argparse type= converter:
    # PyYAML 1.1 parses `lr: 1e-4` (no dot) as the STRING "1e-4", which then
    # bypasses argparse's type= (applied to CLI strings only, not defaults)
    # and survives until a confusing TypeError inside the optimizer.
    _by_dest = {a.dest: a for a in parser._actions}
    for _k, _v in list(cfg.items()):
        _a = _by_dest.get(_k)
        if _a is not None and _a.type is not None and isinstance(_v, str):
            cfg[_k] = _a.type(_v)
    parser.set_defaults(**cfg)
    # A `required=True` arg can't be satisfied by set_defaults (argparse still
    # demands it on the CLI), so clear `required` for any arg the YAML supplies.
    for a in parser._actions:
        if a.dest in cfg and getattr(a, "required", False):
            a.required = False
    return known.config


def save_resolved_config(args, save_dir):
    """Dump the merged (defaults+YAML+CLI) config next to the checkpoint.

    One file in git/the ckpt dir fully reproduces the run; eval scripts can read
    it back. Only plain YAML-serialisable values are kept (all argparse args are).
    """
    d = {k: v for k, v in vars(args).items() if k != "config"}
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "run_config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=True, default_flow_style=False)
    return path
