"""Strict contracts for deterministic summary facts and a grounded draft."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SummaryFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{1,80}$")
    label: str
    value: Any
    unit: str | None = None
    measured_on: str | None = None
    provenance: str
    category: Literal["patient", "valve", "echo", "laboratory", "exposure", "model", "comparator", "limitation", "comparison"]


class AllowedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,60}$")
    text: str
    evidence_ids: list[str] = Field(min_length=1)


class PatientSummaryFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_id: str
    source_kind: Literal["synthetic", "real", "mixed", "unknown"]
    selected_family: str | None
    returned_family: str | None
    model_version: str | None
    comparator_id: str
    comparison_category: str
    illustrative_unvalidated: bool = True
    facts: list[SummaryFact]
    allowed_actions: list[AllowedAction] = Field(default_factory=list)
    family_comparison_included: bool = False


class FindingItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def no_markup(cls, value: str) -> str:
        if ("<" in value or ">" in value or "](" in value or "http://" in value.lower()
                or "https://" in value.lower() or "javascript:" in value.lower()):
            raise ValueError("generated HTML and arbitrary links are not allowed")
        return value.strip()


class RecommendationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_id: str
    rationale: str
    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("rationale")
    @classmethod
    def no_markup(cls, value: str) -> str:
        if ("<" in value or ">" in value or "](" in value or "http://" in value.lower()
                or "https://" in value.lower() or "javascript:" in value.lower()):
            raise ValueError("generated HTML and arbitrary links are not allowed")
        return value.strip()


class ClinicianFindings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    findings: list[FindingItem]
    interpretation: list[FindingItem]
    recommendations: list[RecommendationItem]
    limitations: list[FindingItem]
