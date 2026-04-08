"""Auxiliary proof system — reimplemented from gds-proof for self-containment.

Provides ProofBuilder for constructing multi-lemma SymPy proof scripts and
verify_proof for independent re-execution. Each proof script targets a
specific requirement and is bound to a model version via content hash.

Three lemma kinds:
- EQUALITY: simplify(expr - expected) == 0  or  expr.doit() == expected
- BOOLEAN:  simplify(expr) is sympy.true
- QUERY:    sympy.ask(expr, context) is True

Reference: gds-proof/gds_proof/analysis/proof.py
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import sympy
from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LemmaKind(StrEnum):
    EQUALITY = "equality"
    BOOLEAN = "boolean"
    QUERY = "query"


class ProofStatus(StrEnum):
    UNCHECKED = "UNCHECKED"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

# Pydantic type alias for SymPy expressions
SympyExpr = Any


class Lemma(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    kind: LemmaKind
    expr: SympyExpr
    expected: SympyExpr | None = None
    assumptions: dict[str, dict] = {}
    depends_on: list[str] = []
    description: str = ""


class LemmaResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    lemma_name: str
    passed: bool
    actual_value: SympyExpr | None = None
    error: str | None = None


class ProofResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: ProofStatus
    proof_hash: str | None = None
    lemma_results: list[LemmaResult] = []
    failure_summary: str | None = None


class ProofScript(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    target_invariant: str
    model_hash: str
    claim: str
    lemmas: list[Lemma]

    def to_evidence(self) -> dict[str, Any]:
        """Serialize as JSON-compatible dict for storage and re-verification."""
        return {
            "name": self.name,
            "target_invariant": self.target_invariant,
            "model_hash": self.model_hash,
            "claim": self.claim,
            "lemmas": [
                {
                    "name": lem.name,
                    "kind": lem.kind.value,
                    "expr": sympy.srepr(lem.expr),
                    "expected": (
                        sympy.srepr(lem.expected)
                        if lem.expected is not None
                        else None
                    ),
                    "assumptions": lem.assumptions,
                    "depends_on": lem.depends_on,
                    "description": lem.description,
                }
                for lem in self.lemmas
            ],
        }

    @classmethod
    def from_evidence(cls, data: dict[str, Any]) -> ProofScript:
        """Restore from an evidence dict (inverse of to_evidence)."""
        lemmas = []
        for ld in data["lemmas"]:
            lemmas.append(
                Lemma(
                    name=ld["name"],
                    kind=LemmaKind(ld["kind"]),
                    expr=sympy.sympify(ld["expr"]),
                    expected=(
                        sympy.sympify(ld["expected"])
                        if ld.get("expected") is not None
                        else None
                    ),
                    assumptions=ld.get("assumptions", {}),
                    depends_on=ld.get("depends_on", []),
                    description=ld.get("description", ""),
                )
            )
        return cls(
            name=data["name"],
            target_invariant=data["target_invariant"],
            model_hash=data["model_hash"],
            claim=data["claim"],
            lemmas=lemmas,
        )


# ---------------------------------------------------------------------------
# Lemma verification helpers
# ---------------------------------------------------------------------------


def _build_assumption_subs(
    assumptions: dict[str, dict],
) -> dict[sympy.Symbol, sympy.Symbol]:
    return {
        sympy.Symbol(name): sympy.Symbol(name, **asm)
        for name, asm in assumptions.items()
    }


def _build_q_context(assumptions: dict[str, dict]) -> sympy.Basic:
    facts = []
    for name, asm in assumptions.items():
        sym = sympy.Symbol(name)
        for prop, val in asm.items():
            if val:
                q_pred = getattr(sympy.Q, prop, None)
                if q_pred is not None:
                    facts.append(q_pred(sym))
    if not facts:
        return sympy.Q.is_true(sympy.true)
    return sympy.And(*facts)


# ---------------------------------------------------------------------------
# Lemma verification
# ---------------------------------------------------------------------------


def verify_lemma(lemma: Lemma) -> LemmaResult:
    """Verify a single lemma independently."""
    try:
        assumption_subs = _build_assumption_subs(lemma.assumptions)
        expr = lemma.expr.subs(assumption_subs)

        if lemma.kind == LemmaKind.EQUALITY:
            if lemma.expected is None:
                return LemmaResult(
                    lemma_name=lemma.name,
                    passed=False,
                    error="EQUALITY lemma requires 'expected'",
                )
            expected = lemma.expected.subs(assumption_subs)
            diff = sympy.simplify(expr - expected)
            if diff == sympy.Integer(0):
                return LemmaResult(
                    lemma_name=lemma.name, passed=True, actual_value=sympy.Integer(0)
                )
            evaluated = expr.doit()
            diff = sympy.simplify(evaluated - expected)
            if diff == sympy.Integer(0):
                return LemmaResult(
                    lemma_name=lemma.name, passed=True, actual_value=evaluated
                )
            return LemmaResult(
                lemma_name=lemma.name,
                passed=False,
                actual_value=diff,
                error=f"simplify(expr - expected) = {diff}, not 0",
            )

        if lemma.kind == LemmaKind.BOOLEAN:
            result = sympy.simplify(expr)
            passed = result is sympy.true
            return LemmaResult(
                lemma_name=lemma.name, passed=passed, actual_value=result
            )

        if lemma.kind == LemmaKind.QUERY:
            context = _build_q_context(lemma.assumptions)
            ask_result = sympy.ask(expr, context)
            passed = ask_result is True
            return LemmaResult(
                lemma_name=lemma.name,
                passed=passed,
                actual_value=sympy.true if passed else sympy.false,
            )

        return LemmaResult(
            lemma_name=lemma.name,
            passed=False,
            error=f"Unknown LemmaKind: {lemma.kind}",
        )

    except (TypeError, ValueError, RecursionError, AttributeError) as exc:
        return LemmaResult(lemma_name=lemma.name, passed=False, error=str(exc))


# ---------------------------------------------------------------------------
# Proof script verification
# ---------------------------------------------------------------------------


def verify_proof(script: ProofScript, model_hash: str) -> ProofResult:
    """Verify a complete proof script against a model hash.

    Each lemma is verified independently. A single failing lemma
    fails the entire proof. The proof must be bound to the given
    model_hash.
    """
    from evidence.hashing import hash_proof

    if script.model_hash != model_hash:
        return ProofResult(
            status=ProofStatus.FAILED,
            failure_summary=(
                f"model_hash mismatch: proof authored for "
                f"{script.model_hash!r}, caller provided {model_hash!r}"
            ),
        )

    lemma_results: list[LemmaResult] = []
    for lemma in script.lemmas:
        result = verify_lemma(lemma)
        lemma_results.append(result)
        if not result.passed:
            return ProofResult(
                status=ProofStatus.FAILED,
                proof_hash=hash_proof(script, model_hash),
                lemma_results=lemma_results,
                failure_summary=f"Lemma '{lemma.name}' failed: {result.error}",
            )

    return ProofResult(
        status=ProofStatus.VERIFIED,
        proof_hash=hash_proof(script, model_hash),
        lemma_results=lemma_results,
    )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class ProofBuilder:
    """Chainable builder for constructing proof scripts."""

    def __init__(
        self,
        model_hash: str,
        target_invariant: str,
        name: str,
        claim: str,
    ) -> None:
        self._model_hash = model_hash
        self._target_invariant = target_invariant
        self._name = name
        self._claim = claim
        self._lemmas: list[Lemma] = []

    def lemma(
        self,
        name: str,
        kind: LemmaKind,
        expr: sympy.Basic,
        expected: sympy.Basic | None = None,
        assumptions: dict[str, dict] | None = None,
        depends_on: list[str] | None = None,
        description: str = "",
    ) -> ProofBuilder:
        self._lemmas.append(
            Lemma(
                name=name,
                kind=kind,
                expr=expr,
                expected=expected,
                assumptions=assumptions or {},
                depends_on=depends_on or [],
                description=description,
            )
        )
        return self

    def build(self) -> ProofScript:
        if not self._lemmas:
            raise ValueError("A ProofScript must contain at least one lemma.")
        return ProofScript(
            name=self._name,
            target_invariant=self._target_invariant,
            model_hash=self._model_hash,
            claim=self._claim,
            lemmas=list(self._lemmas),
        )
