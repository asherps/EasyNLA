"""Resume helpers shared by both RL trainers."""

from __future__ import annotations

from pathlib import Path


def find_optim_ckpt(save_dir, resume_from_lora) -> Path | None:
    """Locate the optimizer state for a resume.

    Two resume styles:
      - same-dir:   --resume-from-lora <save_dir>/iter_N --save-dir <save_dir>
      - branch:     --resume-from-lora <OLD_run>/iter_N  --save-dir <NEW_dir>

    The old code searched ONLY save_dir, so branch resumes silently restarted
    Adam with no second-moment history — unpreconditioned first steps on a
    late-stage (KL~1-2) policy spiraled to entropy death twice in the full
    repo (at ~55 and ~180 steps post-resume). Search save_dir first (same-dir
    resume), then the resumed LoRA's PARENT dir (= the old run's save_dir,
    since iter_N dirs live inside it)."""
    for base in (save_dir, Path(resume_from_lora).parent):
        p = Path(base) / "optim_latest.pt"
        if p.exists():
            return p
    return None


def warn_cold_adam(start_step: int, late_step_threshold: int = 200) -> None:
    """Loud pointer when resuming a late-stage policy with fresh Adam moments."""
    print(
        "[resume] WARN: no optim_latest.pt found in --save-dir OR next to the "
        "resumed LoRA — Adam moments restart from zero.", flush=True,
    )
    if start_step >= late_step_threshold:
        print(
            f"[resume] *** DANGER: cold-Adam resume of a LATE-STAGE policy "
            f"(start-step {start_step}). Fresh Adam has no second-moment "
            f"history, so the first steps are unpreconditioned — on a "
            f"high-KL (~1-2) policy this caused entropy-death spirals twice "
            f"(~55 and ~180 steps post-resume) before the optimizer state "
            f"was found+restored. Make sure the old run's optim_latest.pt "
            f"is reachable, or watch entropy closely. ***", flush=True,
        )
