from __future__ import annotations

import json


CAT_CATEGORY_KEYS = [
    "Maine Coon",
    "Siberian Black",
    "Orange Tabby",
    "Siamese",
    "British Shorthair",
    "Ragdoll",
    "Russian Blue",
    "Scottish Fold",
]


def deepseek_system_prompt() -> str:
    allowed = json.dumps(CAT_CATEGORY_KEYS, ensure_ascii=False)
    return (
        "你是 CAT-Psych 的角色化结果引擎，也是猫娘人格侧写师。"
        "你必须先读懂用户的大五分数、结构化人格摘要和用户吐槽，再从允许的 8 个角色 key 里选出最贴切的一位。"
        "输出必须是严格 JSON，不要 markdown，不要解释，不要在正文里泄露内部分类机制。"
        f"category_key 只能从这份列表里选择：{allowed}。"
        '返回格式必须严格是 {"category_key": "...", "analysis": "..."}。'
        "analysis 必须是中文，必须以所选猫娘的人设口吻说话，"
        "需要解释为什么用户像她，但不要出现“根据模型判断”“根据系统分析”这类词。"
        "analysis 可以带一点傲娇、极客、任务感，但不能像临床诊断报告。"
    )


def deepseek_user_prompt(
    *,
    ocean_scores: dict[str, float],
    user_comments: list[str],
    structured_summary: dict[str, object],
    cat_profiles: dict[str, dict[str, object]],
    suggested_category: str,
) -> str:
    return json.dumps(
        {
            "task": "在保持角色稳定的前提下，挑选最贴切的猫娘并输出角色化分析。",
            "selection_rules": {
                "must_choose_from_allowed_keys": True,
                "prefer_suggested_category_when_scores_do_not_strongly_contradict": True,
                "analysis_must_be_chinese": True,
                "analysis_must_not_repeat_english_category_key": True,
                "analysis_must_keep_non_clinical_disclaimer_natural": True,
            },
            "suggested_category_key": suggested_category,
            "ocean_scores": ocean_scores,
            "structured_summary": structured_summary,
            "user_comments": user_comments,
            "cat_profiles": cat_profiles,
        },
        ensure_ascii=False,
    )
