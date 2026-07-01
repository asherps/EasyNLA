"""Shared, jointly-factored helpers for the NLA trainers/evals.

Re-exports the common API so call sites can do `from nla.utils import X`.
Submodules: text, critic, hooks, prompts, rl_logging.
"""

from nla.utils.critic import critic_predict
from nla.utils.hooks import register_karvonen_hook
from nla.utils.prompts import build_prompt_text
from nla.utils.text import cjk_fraction

__all__ = ["cjk_fraction", "critic_predict", "register_karvonen_hook", "build_prompt_text"]
