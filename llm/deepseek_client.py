from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - environment fallback
    OpenAI = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - environment fallback
    load_dotenv = None

from engine.constants import TRAIT_LABELS
from llm.prompt_templates import CAT_CATEGORY_KEYS, deepseek_system_prompt, deepseek_user_prompt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
if load_dotenv is not None:
    load_dotenv(ROOT / ".env")


@lru_cache(maxsize=1)
def load_cat_profiles() -> dict[str, dict[str, object]]:
    payload = json.loads((DATA_DIR / "cat_mapping.json").read_text(encoding="utf-8"))
    return {key: value for key, value in payload.items() if key in CAT_CATEGORY_KEYS}


def analyze_personality(
    ocean_scores: dict[str, float],
    user_comments: list[str],
    *,
    structured_summary: dict[str, object] | None = None,
) -> dict[str, str]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    cat_profiles = load_cat_profiles()
    structured_summary = structured_summary or _build_fallback_summary(ocean_scores)
    suggested_category = _pick_category(ocean_scores, user_comments)

    if not api_key or OpenAI is None:
        return _fallback_analysis(
            ocean_scores,
            user_comments,
            structured_summary=structured_summary,
            category=suggested_category,
            cat_profiles=cat_profiles,
        )

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        completion = client.chat.completions.create(
            model=model,
            temperature=0.8,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": deepseek_system_prompt()},
                {
                    "role": "user",
                    "content": deepseek_user_prompt(
                        ocean_scores=ocean_scores,
                        user_comments=user_comments,
                        structured_summary=structured_summary,
                        cat_profiles=cat_profiles,
                        suggested_category=suggested_category,
                    ),
                },
            ],
        )
        content = completion.choices[0].message.content or ""
        payload = json.loads(content)
        category_key = str(payload.get("category_key", "")).strip()
        analysis = str(payload.get("analysis", "")).strip()
        if category_key not in CAT_CATEGORY_KEYS or not analysis:
            raise ValueError("DeepSeek returned an invalid category or empty analysis.")
        return {
            "category_key": category_key,
            "analysis": _sanitize_analysis(analysis, category_key, cat_profiles),
        }
    except Exception:
        return _fallback_analysis(
            ocean_scores,
            user_comments,
            structured_summary=structured_summary,
            category=suggested_category,
            cat_profiles=cat_profiles,
        )


def _fallback_analysis(
    ocean_scores: dict[str, float],
    user_comments: list[str],
    *,
    structured_summary: dict[str, object],
    category: str,
    cat_profiles: dict[str, dict[str, object]],
) -> dict[str, str]:
    profile = cat_profiles[category]
    headline = structured_summary.get("headline_trait", {})
    lowest = structured_summary.get("lowest_trait", {})
    evidence_level = str(structured_summary.get("evidence_level", "中等偏稳"))
    caution_lines = structured_summary.get("cautions", [])
    comment_hint = user_comments[-1].strip() if user_comments else "这次你没额外解释太多，那我就只看答题轨迹。"
    role_name = str(profile["name"])
    tone = str(profile["tone"])
    persona_seed = str(profile["persona_seed"])
    motifs = "、".join(profile.get("supporting_motifs", [])[:2])

    analysis = (
        f"{role_name}先接管一下。{persona_seed}"
        f"这轮分数里，你最突出的信号落在「{headline.get('label', '某个维度')}」({headline.get('score', 0):.1f})，"
        f"而相对偏低的是「{lowest.get('label', '某个维度')}」({lowest.get('score', 0):.1f})。"
        f"这就让你更像那种会用 {motifs or tone} 去推进任务的人，不是模板化答题，也不是纯靠情绪乱冲。"
        f"你补的那句『{comment_hint}』也很有味道，本喵会把它当成你节奏感的一部分。"
        f"总之，这次把你归到我这条线，不是随手贴标签，而是因为你的特质确实和这套人设咬合得上。"
        f"{caution_lines[-1] if caution_lines else '这份结果是风格化筛查参考，不等同于临床诊断。'}"
        f"当前证据强度先记作「{evidence_level}」，后面如果再测一轮，轮廓还会更清楚。"
    )
    return {
        "category_key": category,
        "analysis": _sanitize_analysis(analysis, category, cat_profiles),
    }


def _sanitize_analysis(analysis: str, category_key: str, cat_profiles: dict[str, dict[str, object]]) -> str:
    cleaned = " ".join(analysis.replace("\n", " ").split())
    for key, profile in cat_profiles.items():
        cleaned = cleaned.replace(key, str(profile["name"]))
    if "根据模型判断" in cleaned or "根据系统" in cleaned:
        cleaned = cleaned.replace("根据模型判断", "按这轮表现看")
        cleaned = cleaned.replace("根据系统", "按这轮轨迹")
    return cleaned


def _pick_category(ocean_scores: dict[str, float], user_comments: list[str]) -> str:
    comments = " ".join(user_comments).lower()
    extraversion = ocean_scores.get("extraversion", 50.0)
    agreeableness = ocean_scores.get("agreeableness", 50.0)
    conscientiousness = ocean_scores.get("conscientiousness", 50.0)
    emotional_stability = ocean_scores.get("emotional_stability", 50.0)
    intellect = ocean_scores.get("intellect", 50.0)

    if intellect >= 58 and conscientiousness >= 55:
        return "Maine Coon"
    if "难绷" in comments or "黑" in comments or emotional_stability < 45:
        return "Siberian Black"
    if extraversion >= 58:
        return "Orange Tabby"
    if intellect >= 60:
        return "Siamese"
    if conscientiousness >= 58 and agreeableness >= 50:
        return "British Shorthair"
    if agreeableness >= 58 and emotional_stability >= 50:
        return "Ragdoll"
    if emotional_stability >= 57 and extraversion < 50:
        return "Russian Blue"
    return "Scottish Fold"


def _build_fallback_summary(ocean_scores: dict[str, float]) -> dict[str, object]:
    if not ocean_scores:
        return {
            "headline_trait": {"label": "智力/开放", "score": 50.0},
            "lowest_trait": {"label": "情绪稳定", "score": 50.0},
            "high_traits": [],
            "low_traits": [],
            "cautions": ["这份结果是风格化筛查参考，不等同于临床诊断。"],
            "evidence_level": "中等偏稳",
        }
    ranked = sorted(ocean_scores.items(), key=lambda item: item[1], reverse=True)
    top_trait, top_score = ranked[0]
    bottom_trait, bottom_score = ranked[-1]
    label_map = TRAIT_LABELS
    return {
        "headline_trait": {"trait": top_trait, "label": label_map.get(top_trait, top_trait), "score": round(top_score, 1)},
        "lowest_trait": {"trait": bottom_trait, "label": label_map.get(bottom_trait, bottom_trait), "score": round(bottom_score, 1)},
        "high_traits": [],
        "low_traits": [],
        "cautions": ["这份结果是风格化筛查参考，不等同于临床诊断。"],
        "evidence_level": "中等偏稳",
    }
