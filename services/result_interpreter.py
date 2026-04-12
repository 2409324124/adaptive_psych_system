from __future__ import annotations

from dataclasses import dataclass


TRAIT_LABELS = {
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "conscientiousness": "Conscientiousness",
    "emotional_stability": "Emotional stability",
    "intellect": "Intellect / openness",
}


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
            overview = (
                f"Current responses lean most strongly toward {TRAIT_LABELS.get(top_trait, top_trait).lower()}."
            )
        elif lower:
            overview = (
                f"Current responses are relatively lower on {TRAIT_LABELS.get(bottom_trait, bottom_trait).lower()} than on the other tracked traits."
            )
        else:
            overview = (
                "Current responses cluster near the middle range, with no single trait standing far away from the rest."
            )

        range_summary = (
            f"Highest current IRT tendency is {TRAIT_LABELS.get(top_trait, top_trait)} ({top_score:.1f}); "
            f"lowest is {TRAIT_LABELS.get(bottom_trait, bottom_trait)} ({bottom_score:.1f})."
        )

        cautions = []
        if low_evidence_traits:
            cautions.append(
                "Lower evidence is still present for "
                + ", ".join(low_evidence_traits)
                + ". Treat those readings as provisional."
            )
        cautions.append(
            "These outputs are screening-oriented tendency estimates, not clinical findings."
        )

        return {
            "overview": overview,
            "range_summary": range_summary,
            "highlights": standout or ["No trait is currently far above the mid-range band."],
            "lowlights": lower or ["No trait is currently far below the mid-range band."],
            "cautions": cautions,
            "low_evidence_traits": low_evidence_traits,
        }

    def _trait_summary(self, trait: str, score: float, *, direction: str) -> str:
        label = TRAIT_LABELS.get(trait, trait)
        if direction == "high":
            return f"{label} is currently on the higher side ({score:.1f})."
        return f"{label} is currently on the lower side ({score:.1f})."
