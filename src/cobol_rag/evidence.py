from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Iterable


class EvidenceState(StrEnum):
    """Why a claim can or cannot be answered from analyzed evidence."""

    PROVEN = "proven"
    PROVEN_ABSENT = "proven_absent"
    ANALYSIS_GAP = "analysis_gap"
    RETRIEVAL_MISS = "retrieval_miss"
    EVIDENCE_REJECTED = "evidence_rejected"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class EvidenceRecord:
    """Typed boundary record passed from retrieval to claim execution."""

    evidence_id: str
    program: str
    evidence_type: str
    source_file: str
    text: str
    score: float | None = None
    entity_key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: Any) -> "EvidenceRecord":
        metadata = dict(getattr(result, "metadata", {}) or {})
        return cls(
            evidence_id=str(
                metadata.get("source_id")
                or metadata.get("evidence_id")
                or metadata.get("source_file")
                or ""
            ),
            program=str(metadata.get("program", "")).upper(),
            evidence_type=str(metadata.get("chunk_type", "")),
            source_file=str(
                metadata.get("source_file") or metadata.get("evidence_path") or ""
            ),
            text=str(getattr(result, "text", "") or ""),
            score=getattr(result, "score", None),
            entity_key=str(metadata.get("entity_key", "")),
            metadata=metadata,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceDisposition:
    state: EvidenceState
    capability: str = ""
    reasons: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload


def disposition_for_results(
    results: Iterable[Any],
    *,
    capability: str = "",
    guard_status: str = "pass",
    reasons: Iterable[str] = (),
) -> EvidenceDisposition:
    records = tuple(EvidenceRecord.from_result(result) for result in results)
    reason_values = tuple(str(reason) for reason in reasons if str(reason))
    if guard_status == "insufficient":
        state = EvidenceState.EVIDENCE_REJECTED if records else EvidenceState.RETRIEVAL_MISS
    elif records:
        state = EvidenceState.PROVEN
    else:
        state = EvidenceState.RETRIEVAL_MISS
    return EvidenceDisposition(
        state=state,
        capability=capability,
        reasons=reason_values,
        evidence_ids=tuple(record.evidence_id for record in records if record.evidence_id),
    )
