from __future__ import annotations

from llm.deepseek_client import analyze_personality
from llm.prompt_templates import CAT_CATEGORY_KEYS


def test_deepseek_fallback_returns_valid_category(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    payload = analyze_personality(
        ocean_scores={
            "extraversion": 42.0,
            "agreeableness": 58.0,
            "conscientiousness": 55.0,
            "emotional_stability": 44.0,
            "intellect": 61.0,
        },
        user_comments=["我社交时像掉线，但遇到新技术会自己钻进去。"],
    )

    assert payload["category_key"] in CAT_CATEGORY_KEYS
    assert payload["analysis"]
