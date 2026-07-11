"""Pydantic output models for all four agents."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """Traceable evidence from the original paper."""

    id: str = Field(description="Evidence snippet ID, for example E003, T001, or F002")
    section: str = Field(description="Paper section where the evidence appears")
    page: str = Field(description="Page label or page range, for example p.3 or pp.4-5")
    quote: str = Field(description="Short original-paper quote or faithful paraphrase")
    note: str = Field(description="How this evidence supports the analysis")


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
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Key evidence snippets supporting method claims",
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
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Key evidence snippets supporting experiment claims",
    )


class NoveltyDimensionScore(BaseModel):
    """One evidence-backed component of the novelty assessment."""

    dimension: str = Field(
        description=(
            "Stable dimension key: problem_originality, method_originality, "
            "prior_work_difference, or generality"
        )
    )
    score: int = Field(ge=1, le=5, description="Dimension score from 1 to 5")
    reason: str = Field(description="Evidence-grounded reason for this dimension score")
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Evidence IDs supporting this dimension score",
    )


class CriticOutput(BaseModel):
    """CriticAgent output: critical review and assessment."""

    novelty_score: int = Field(ge=1, le=5, description="Novelty score from 1 (low) to 5 (high)")
    novelty_justification: str = Field(description="Justification for the novelty score")
    novelty_dimensions: List[NoveltyDimensionScore] = Field(
        default_factory=list,
        description="Four evidence-backed novelty dimension scores",
    )
    strengths: List[str] = Field(description="Key strengths of this work")
    limitations: List[str] = Field(description="Identified limitations of the paper")
    potential_improvements: List[str] = Field(description="Suggested directions for improvement")
    broader_impact: Optional[str] = Field(
        default=None, description="Potential broader impact or societal implications"
    )
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Key evidence snippets supporting critical review claims",
    )


class NoveltyAssessment(BaseModel):
    """Backend-calculated novelty result."""

    score: float = Field(ge=1, le=5)
    label: str
    dimensions: List[NoveltyDimensionScore] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ReliabilityBreakdown(BaseModel):
    """Deterministic components of analysis reliability."""

    parsing: int = Field(ge=0, le=20)
    coverage: int = Field(ge=0, le=35)
    citations: int = Field(ge=0, le=30)
    output_integrity: int = Field(ge=0, le=15)


class ReliabilityAssessment(BaseModel):
    """Deterministic reliability score for the generated analysis."""

    score: int = Field(ge=0, le=100)
    raw_score: int = Field(ge=0, le=100)
    score_cap: int = Field(ge=0, le=100)
    level: str
    label: str
    breakdown: ReliabilityBreakdown
    warnings: List[str] = Field(default_factory=list)


class AnalysisAssessment(BaseModel):
    """Combined novelty and reliability assessment returned by the API."""

    novelty: NoveltyAssessment
    reliability: ReliabilityAssessment


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
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Most important evidence carried into the final note",
    )


class ComparisonPaper(BaseModel):
    """One saved paper participating in a comparison workspace."""

    history_id: str
    label: str = Field(description="Stable paper label such as P1")
    title: str
    filename: str


class ComparisonCell(BaseModel):
    """One paper's answer for a shared comparison dimension."""

    paper_label: str = Field(description="Paper label such as P1")
    summary: str = Field(description="Concise paper-specific finding for this dimension")
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Prefixed evidence IDs such as P1:E003 or P2:T001",
    )


class ComparisonDimension(BaseModel):
    """A shared axis used to compare all selected papers."""

    key: str = Field(description="Stable snake_case dimension key")
    title: str = Field(description="Short Simplified Chinese dimension title")
    category: Literal["overview", "method", "experiment", "critique"]
    description: str = Field(description="What is being compared and why it matters")
    cells: List[ComparisonCell] = Field(default_factory=list)
    synthesis: str = Field(description="Cross-paper conclusion for this dimension")
    comparability: Literal["direct", "conditional", "not_comparable"]
    warning: Optional[str] = Field(
        default=None,
        description="Why the values are only conditionally comparable or not comparable",
    )


class ComparisonOutput(BaseModel):
    """Evidence-grounded structured comparison across saved papers."""

    title: str
    focus: str
    executive_summary: str
    papers: List[ComparisonPaper] = Field(default_factory=list)
    common_ground: List[str] = Field(default_factory=list)
    key_differences: List[str] = Field(default_factory=list)
    dimensions: List[ComparisonDimension] = Field(default_factory=list)
    research_gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ComparisonAssessment(BaseModel):
    """Deterministic coverage indicators for a comparison result."""

    score: int = Field(ge=0, le=100)
    label: str
    evidence_coverage: int = Field(ge=0, le=100)
    paper_coverage: int = Field(ge=0, le=100)
    referenced_claims: int = Field(ge=0)
    total_claims: int = Field(ge=0)
    warnings: List[str] = Field(default_factory=list)
