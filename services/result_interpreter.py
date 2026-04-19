from __future__ import annotations

from dataclasses import dataclass

from engine.constants import TRAIT_LABELS


@dataclass(frozen=True)
class ResultInterpreter:
    low_evidence_threshold: int = 2
    standout_threshold: float = 55.0
    low_threshold: float = 45.0

    def interpret(
        self,
        *,
        irt_t_scores: dict[str, float],
        dimension_answer_counts: dict[str, int],
    ) -> dict[str, object]:
        ranked = sorted(irt_t_scores.items(), key=lambda item: item[1], reverse=True)
        top_trait, top_score = ranked[0]
        bottom_trait, bottom_score = ranked[-1]

        standout = [
            self._trait_summary(trait, score, direction="high")
            for trait, score in ranked
            if score >= self.standout_threshold
        ]
        lower = [
            self._trait_summary(trait, score, direction="low")
            for trait, score in reversed(ranked)
            if score <= self.low_threshold
        ]

        low_evidence_traits = [
            TRAIT_LABELS.get(trait, trait)
            for trait, count in dimension_answer_counts.items()
            if count < self.low_evidence_threshold
        ]

        if standout:
            overview = f"这一轮最突出的信号，落在「{TRAIT_LABELS.get(top_trait, top_trait)}」这一侧。"
        elif lower:
            overview = f"这一轮里相对偏低的信号，主要落在「{TRAIT_LABELS.get(bottom_trait, bottom_trait)}」这一侧。"
        else:
            overview = "这轮回答整体还比较居中，没有哪一个维度明显把其他维度拉开。"

        range_summary = (
            f"当前 IRT 倾向里，最高的是「{TRAIT_LABELS.get(top_trait, top_trait)}」({top_score:.1f})，"
            f"最低的是「{TRAIT_LABELS.get(bottom_trait, bottom_trait)}」({bottom_score:.1f})。"
        )

        cautions = []
        if low_evidence_traits:
            cautions.append("以下维度证据还偏少：" + "、".join(low_evidence_traits) + "。相关解读先按暂定信号看待。")
        cautions.append("这是一份风格化筛查结果，不等同于临床诊断。")

        structured_summary = {
            "headline_trait": self._trait_payload(top_trait, top_score),
            "lowest_trait": self._trait_payload(bottom_trait, bottom_score),
            "high_traits": [
                self._trait_payload(trait, score)
                for trait, score in ranked
                if score >= self.standout_threshold
            ],
            "low_traits": [
                self._trait_payload(trait, score)
                for trait, score in reversed(ranked)
                if score <= self.low_threshold
            ],
            "cautions": cautions,
            "evidence_level": "暂定" if low_evidence_traits else "中等偏稳",
            "low_evidence_traits": low_evidence_traits,
        }

        return {
            "overview": overview,
            "range_summary": range_summary,
            "highlights": standout or ["暂时没有哪个维度明显高出中段。"],
            "lowlights": lower or ["暂时没有哪个维度明显低出中段。"],
            "cautions": cautions,
            "low_evidence_traits": low_evidence_traits,
            "structured_summary": structured_summary,
        }

    def _trait_summary(self, trait: str, score: float, *, direction: str) -> str:
        label = TRAIT_LABELS.get(trait, trait)
        if direction == "high":
            return f"「{label}」目前偏高一些（{score:.1f}）。"
        return f"「{label}」目前偏低一些（{score:.1f}）。"

    def _trait_payload(self, trait: str, score: float) -> dict[str, object]:
        return {
            "trait": trait,
            "label": TRAIT_LABELS.get(trait, trait),
            "score": round(score, 1),
        }
