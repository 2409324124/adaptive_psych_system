from __future__ import annotations

import json
from pathlib import Path

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
    assert payload["category_key"] not in payload["analysis"]


def test_cat_mapping_contains_persona_template_fields() -> None:
    mapping_path = Path(__file__).resolve().parents[1] / "data" / "cat_mapping.json"
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))

    assert set(payload.keys()) == set(CAT_CATEGORY_KEYS)
    for category, profile in payload.items():
        assert profile["name"]
        assert profile["image"].startswith("/static/cats/")
        assert "%" in profile["image_position"]
        assert profile["persona_seed"]
        assert profile["tone"]
        assert len(profile["supporting_motifs"]) == 3
        assert len(profile["taboo_phrases"]) == 4
        assert category not in profile["persona_seed"]


def test_ipip_translation_file_covers_all_items() -> None:
    root = Path(__file__).resolve().parents[1]
    items_payload = json.loads((root / "data" / "ipip_items.json").read_text(encoding="utf-8"))
    zh_payload = json.loads((root / "data" / "ipip_items_zh.json").read_text(encoding="utf-8"))

    item_ids = {item["id"] for item in items_payload["items"]}
    assert set(zh_payload.keys()) == item_ids
    assert all(str(text).strip() for text in zh_payload.values())
