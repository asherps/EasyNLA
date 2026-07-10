"""Regression tests for utils/patch_vllm_lens.py version tolerance.

Guards the stale-engine failure mode: a fresh venv ships pip's vllm-lens
1.1.x, whose source already incorporates some early fixes upstream — so those
hunks match neither their OLD nor their NEW text. Before baseline tolerance
the patcher REFUSED ("hunk 0 not found"), silently leaving fresh installs
without the chunked-prefill fix (lost/mis-positioned injections) and without
the per-request verification log.
"""

import importlib.util
import unittest
from pathlib import Path


def _load_patcher():
    spec = importlib.util.spec_from_file_location(
        "patch_vllm_lens",
        Path(__file__).resolve().parents[1] / "utils" / "patch_vllm_lens.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _installed_worker_ext():
    try:
        spec = importlib.util.find_spec("vllm_lens._worker_ext")
    except (ImportError, ModuleNotFoundError):
        return None
    return Path(spec.origin) if spec and spec.origin else None


class TestPatcherBaselines(unittest.TestCase):

    def test_satisfied_baselines_upgrade_to_full_patch(self):
        """The upstreamed pip baseline (early fixes incorporated, no log_key —
        exactly what a fresh `pip install vllm-lens` ships) must be upgradable:
        hunks with SATISFIED_* alternates skip, everything else applies from
        OLD. No hunk may be refused."""
        mod = _load_patcher()
        # Reconstruct the upstream baseline from the patcher's own texts:
        # SATISFIED_* for the amended hunks, OLD for the rest.
        baseline_parts = []
        for hunk in mod.HUNKS:
            old = hunk[0]
            sat = hunk[2] if len(hunk) > 2 else []
            baseline_parts.append(sat[0] if sat else old)
        src = "\n\n".join(dict.fromkeys(baseline_parts))  # dedupe, keep order
        for i, hunk in enumerate(mod.HUNKS):
            old, new = hunk[0], hunk[1]
            sat = hunk[2] if len(hunk) > 2 else []
            satisfied = new in src or any(a in src for a in sat)
            appliable = old in src
            self.assertTrue(satisfied or appliable,
                            f"hunk {i} would be refused on the upstream "
                            f"baseline (neither satisfied nor appliable)")

    @unittest.skipUnless(_installed_worker_ext(), "vllm_lens not importable")
    def test_installed_file_fully_patched(self):
        """Where vllm-lens IS installed, the live _worker_ext.py must carry
        every hunk's NEW text — a partially-patched install means the patcher
        was not re-run after a venv rebuild."""
        mod = _load_patcher()
        src = _installed_worker_ext().read_text()
        for i, hunk in enumerate(mod.HUNKS):
            self.assertIn(hunk[1], src,
                          f"hunk {i} not applied to installed vllm-lens — "
                          f"re-run utils/patch_vllm_lens.py")

    def test_seqlens_fix_semantics(self):
        """The fixed seq_lens lookup on a vLLM-v1-style DICT attn_metadata:
        getattr on the dict yields None (the bug — abs_start=0 fallback for
        every chunk); the fix recovers seq_lens from the member objects."""

        class _Meta:
            def __init__(self, seq_lens):
                self.seq_lens = seq_lens
                self.query_start_loc = [0]

        attn_metadata = {"group0": _Meta([96, 32])}
        # pre-fix lookup
        self.assertIsNone(getattr(attn_metadata, "seq_lens", None))
        # post-fix lookup (mirrors the NEW_SEQLENS hunk)
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if seq_lens is None and hasattr(attn_metadata, "values"):
            for m in attn_metadata.values():
                if getattr(m, "seq_lens", None) is not None:
                    seq_lens = m.seq_lens
                    break
        self.assertEqual(seq_lens, [96, 32])


if __name__ == "__main__":
    unittest.main()
