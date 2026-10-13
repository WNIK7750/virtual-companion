from __future__ import annotations

import uuid

from ..models import FeedbackAnalysis, FeedbackExport, FeedbackFinding


class RecordingAnalyzer:
    """Deterministic analyzer fixture; it never calls a provider or executes feedback text."""

    def __init__(self, *, analysis_model: str, analysis_prompt_sha256: str) -> None:
        self.analysis_model = analysis_model
        self.analysis_prompt_sha256 = analysis_prompt_sha256
        self.calls: list[str] = []

    def analyze(self, export: FeedbackExport) -> FeedbackAnalysis:
        self.calls.append(export.export_id)
        findings: list[FeedbackFinding] = []
        for item in export.payload.get("feedback", []):
            feedback_id = item["feedback_id"]
            reaction = item.get("freeform_reaction")
            tags = item.get("quick_tags") or []
            observation = reaction or ", ".join(tags) or "feedback received"
            findings.append(
                FeedbackFinding(
                    observation=observation,
                    likely_layer="unknown",
                    hypothesis=(
                        "Synthetic recording hypothesis; human review is required before action."
                    ),
                    confidence="low",
                    evidence_feedback_ids=(feedback_id,),
                    alternative_explanations=("用户情境或上下文可能导致相同反应。",),
                    recommended_actions=("人工复核该反馈并决定是否新增评测用例。",),
                    new_eval_cases=(f"feedback-{feedback_id}",),
                    needs_human_confirmation=True,
                )
            )
        return FeedbackAnalysis.create(
            analysis_id=str(uuid.uuid4()),
            input_export_id=export.export_id,
            input_export_sha256=export.payload_sha256,
            analysis_model=self.analysis_model,
            analysis_prompt_sha256=self.analysis_prompt_sha256,
            findings=findings,
        )
