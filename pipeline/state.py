"""Typed per-stage pipeline state.

WP1 splits `pipeline.runner.run_pipeline` into stage-level free
functions threaded by a `PipelineState` object. Each stage's result
record is a frozen dataclass attached to the state so downstream
stages — and interrogation tools like `interrogate.rerun` — can read
prior-stage outputs without inspecting locals or re-querying the
graph.

The `activity_to_stage` table maps `p-plan` step IRI fragments
(emitted by `traceability.plan_execution.emit_stage_activity`) to
their pipeline stage numbers. Kept in sync with
`traceability.plan_execution.STEP_NAMES`; covered by a unit test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rdflib import Dataset

if TYPE_CHECKING:
    from compute.base import ComputeBackend, ExecutionMetadata
    from traceability.audit import AuditReport
    from traceability.validation import ValidationReport


@dataclass(frozen=True)
class StructuralResult:
    model_hash: str
    params: dict[str, Any]
    triples_loaded: int


@dataclass(frozen=True)
class SymbolicResult:
    sym_result: Any
    sym_meta: "ExecutionMetadata"
    proofs: dict[str, Any]
    proof_results: dict[str, Any]


@dataclass(frozen=True)
class NumericalResult:
    step_result: Any
    step_meta: "ExecutionMetadata"
    step_summary: dict[str, Any]
    dist_result: Any
    dist_meta: "ExecutionMetadata"
    dist_summary: dict[str, Any]


@dataclass(frozen=True)
class EvidenceBindingResult:
    evidence_node_count: int


@dataclass(frozen=True)
class AttestationStageResult:
    attestation_uris: dict[str, str] | None


@dataclass(frozen=True)
class ClosureRuleResult:
    report: "ValidationReport"


@dataclass(frozen=True)
class AuditStageResult:
    report: "AuditReport"


@dataclass(frozen=True)
class ReportStageResult:
    persisted_graphs: dict[str, int]
    backend_name: str


@dataclass
class PipelineState:
    """Mutable container threaded through every pipeline stage.

    The `ds` Dataset is mutated in place by each stage (named-graph
    writes); per-stage result records are assigned to the matching
    field as each stage completes. Stages downstream of a given stage
    read its result via `state.<stage>.<field>`.
    """

    ds: Dataset
    compute_backend: "ComputeBackend"
    engineer_name: str
    auto_attest: bool = False
    skip_attestation: bool = False
    backend_name: str = "local"
    compute_name: str = "local"

    structural: StructuralResult | None = None
    symbolic: SymbolicResult | None = None
    numerical: NumericalResult | None = None
    evidence: EvidenceBindingResult | None = None
    attestation: AttestationStageResult | None = None
    closure_rules: ClosureRuleResult | None = None
    audit: AuditStageResult | None = None
    report: ReportStageResult | None = None

    activity_to_stage: dict[str, int] = field(default_factory=lambda: {
        "OntologyAssembly":    0,
        "LoadStructural":      1,
        "SymbolicAnalysis":    2,
        "NumericalSimulation": 3,
        "BindEvidence":        4,
        "AssembleRTM":         5,
        "Attest":              6,
        "ValidateShapes":      6,
        "AuditTrace":          7,
        "Report":              7,
        "Interrogate":         8,
    })
