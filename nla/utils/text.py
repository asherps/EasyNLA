"""Small text helpers shared across trainers/evals."""

import unicodedata


def cjk_fraction(text: str) -> float:
    """Fraction of CJK characters in `text` (0.0 for empty).

    The injection-failure smoke signal: when injection silently fails the actor
    echoes the CJK marker glyph and free-associates Chinese, so a high CJK
    fraction flags broken injection. English explanations are ~0%.
    """
    if not text:
        return 0.0
    return sum(1 for c in text if "CJK" in unicodedata.name(c, "")) / len(text)
