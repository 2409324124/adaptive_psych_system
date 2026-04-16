from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - environment fallback
    OpenAI = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - environment fallback
    load_dotenv = None

from llm.prompt_templates import CAT_CATEGORY_KEYS, deepseek_system_prompt

ROOT = Path(__file__).resolve().parents[1]
if load_dotenv is not None:
    load_dotenv(ROOT / ".env")


def analyze_personality(ocean_scores: dict[str, float], user_comments: list[str]) -> dict[str, str]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if not api_key or OpenAI is None:
        return _fallback_analysis(ocean_scores, user_comments)

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        completion = client.chat.completions.create(
            model=model,
            temperature=0.7,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": deepseek_system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "ocean_scores": ocean_scores,
                            "user_comments": user_comments,
                        },
                        ensure_ascii=False,
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
        return {"category_key": category_key, "analysis": analysis}
    except Exception:
        return _fallback_analysis(ocean_scores, user_comments)


def _fallback_analysis(ocean_scores: dict[str, float], user_comments: list[str]) -> dict[str, str]:
    category = _pick_category(ocean_scores, user_comments)
    top_trait = max(ocean_scores, key=ocean_scores.get) if ocean_scores else "intellect"
    comment_hint = user_comments[-1].strip() if user_comments else "这次先不多说，喵会自己从你的答题轨迹里看门道。"
    trait_label = {
        "extraversion": "外向",
        "agreeableness": "和谐",
        "conscientiousness": "尽责",
        "emotional_stability": "情绪稳定",
        "intellect": "智力/开放",
    }.get(top_trait, top_trait)
    analysis = (
        f"哼，先别太得意。按这轮分数看，你最突出的信号落在「{trait_label}」这一侧，"
        f"所以本喵先把你归到 {category} 这条人设线上。"
        f"你刚才的吐槽里还有一句让我记住了：『{comment_hint}』。"
        "综合来看，你不是那种模板化答题的人，更像会带着自己的节奏和执念推进任务的人。"
        "这份结果只是风格化筛查参考，不是临床判断，但已经很够本喵给你贴一个像样的角色标签了。"
    )
    return {"category_key": category, "analysis": analysis}


def _pick_category(ocean_scores: dict[str, float], user_comments: list[str]) -> str:
    comments = " ".join(user_comments).lower()
    extraversion = ocean_scores.get("extraversion", 50.0)
    agreeableness = ocean_scores.get("agreeableness", 50.0)
    conscientiousness = ocean_scores.get("conscientiousness", 50.0)
    emotional_stability = ocean_scores.get("emotional_stability", 50.0)
    intellect = ocean_scores.get("intellect", 50.0)

    if intellect >= 58 and conscientiousness >= 55:
        return "Maine Coon"
    if "黑" in comments or "夜" in comments or emotional_stability < 45:
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
