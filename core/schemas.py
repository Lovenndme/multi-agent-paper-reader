"""Pydantic output models for all four agents."""

from typing import List, Optional
from pydantic import BaseModel, Field


class MethodOutput(BaseModel):
    """MethodAgent output: research methods and technical contributions."""

    research_problem: str = Field(description="The core problem this paper addresses")
    proposed_method: str = Field(description="High-level description of the proposed method/model")
    key_components: List[str] = Field(description="Key technical components or modules")
    innovations: List[str] = Field(description="Technical innovations compared to prior work")
    differences_from_prior: str = Field(
        description="How this method differs from existing approaches"
    )
    implementation_details: Optional[str] = Field(
        default=None, description="Notable implementation specifics"
    )


class ExperimentOutput(BaseModel):
    """ExperimentAgent output: datasets, metrics, and results."""

    datasets: List[str] = Field(description="Datasets used for evaluation")
    metrics: List[str] = Field(description="Evaluation metrics used")
    main_results: str = Field(description="Summary of main experimental results")
    comparison_with_baselines: str = Field(
        description="Performance comparison against baseline methods"
    )
    ablation_study: Optional[str] = Field(
        default=None, description="Key findings from ablation studies, if any"
    )
    notable_findings: List[str] = Field(description="Noteworthy experimental findings")


class CriticOutput(BaseModel):
    """CriticAgent output: critical review and assessment."""

    novelty_score: int = Field(ge=1, le=5, description="Novelty score from 1 (low) to 5 (high)")
    novelty_justification: str = Field(description="Justification for the novelty score")
    strengths: List[str] = Field(description="Key strengths of this work")
    limitations: List[str] = Field(description="Identified limitations of the paper")
    potential_improvements: List[str] = Field(description="Suggested directions for improvement")
    broader_impact: Optional[str] = Field(
        default=None, description="Potential broader impact or societal implications"
    )


class SummaryOutput(BaseModel):
    """SummaryAgent output: integrated structured paper notes."""

    one_sentence_summary: str = Field(description="One-sentence summary of the paper")
    core_contributions: List[str] = Field(
        description="Core contributions, typically 3-5 bullet points"
    )
    method_highlights: str = Field(
        description="Concise description of the key method and architecture"
    )
    experiment_highlights: str = Field(
        description="Most important experimental results and comparisons"
    )
    limitations_and_future_work: str = Field(
        description="Main limitations and promising future directions"
    )
    reading_notes: Optional[str] = Field(
        default=None, description="Additional reading notes or personal observations"
    )
