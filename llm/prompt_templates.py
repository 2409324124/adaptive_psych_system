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
        "你是 CAT-Psych 的人格分析引擎。"
        "你会读取用户的大五人格 T 分数和用户的吐槽/评论，"
        "然后从指定的 8 个猫娘类别中挑选一个最贴切的人设。"
        "输出必须是严格 JSON，不要有 markdown，不要有额外解释。"
        f"category_key 只能从这个列表里选择: {allowed}。"
        '返回格式必须是 {"category_key": "...", "analysis": "..."}。'
        "analysis 必须用中文，风格要有一点傲娇、极客、角色设定感，但不能像临床诊断。"
    )
