"""Shared constants for the CAT-Psych adaptive psychometric system.

This module is the single source of truth for trait labels, trait order,
and the standard disclaimer text.  Every module that needs these values
should import them from here instead of maintaining a local copy.
"""

from __future__ import annotations


# ── Big Five dimension labels (english key → chinese display) ────────────
TRAIT_LABELS: dict[str, str] = {
    "extraversion": "外向",
    "agreeableness": "和谐",
    "conscientiousness": "尽责",
    "emotional_stability": "情绪稳定",
    "intellect": "智力/开放",
}

# Canonical display order used by reports, CLI output, and simulations.
TRAIT_ORDER: list[str] = [
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "emotional_stability",
    "intellect",
]

# ── Standard disclaimer ─────────────────────────────────────────────────
DISCLAIMER: str = "本系统仅作为心理特质筛查与辅助参考工具，绝对不可替代专业精神科临床诊断。"
DISCLAIMER_ASCII: str = (
    "This system is only a psychological trait screening and reference tool. "
    "It must not replace professional psychiatric clinical diagnosis."
)
