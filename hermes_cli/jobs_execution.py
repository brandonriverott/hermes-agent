"""Provider-neutral evidence contracts shared by every Jobs executor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from hermes_cli import jobs_exec as jx
from hermes_cli import jobs_receipts


MAX_PRESERVED_BYTES = 64 * 1024


class AdapterError(RuntimeError):
    """An executor refused before provider work could safely begin."""


@dataclass(frozen=True)
class ArtifactClaim:
    """One executor-authored byte claim for independent readback."""

    name: str
    path: Path
    digest: str
    size: int


@dataclass(frozen=True)
class ReliabilityExecution:
    """Provider-neutral evidence returned to the reliability dispatcher."""

    status: str
    commit: Optional[str]
    worktree: Path
    artifacts: tuple[ArtifactClaim, ...]
    executor_exit_digest: str
    output_capture_digest: str
    builder_handoff: Optional[Mapping[str, object]] = None
    tester_handoff: Optional[Mapping[str, object]] = None
    reviewer_handoff: Optional[Mapping[str, object]] = None
    failure_reason_code: Optional[str] = None
    http_status: Optional[int] = None
    safety_gate: bool = False


def claim_artifact(path: Path, *, name: str) -> ArtifactClaim:
    """Capture an artifact's byte identity without granting it authority."""

    clean_name = str(name or "").strip()
    if not clean_name:
        raise ValueError("artifact name must not be empty")
    source = Path(path)
    data = source.read_bytes()
    return ArtifactClaim(
        name=clean_name,
        path=source,
        digest=jobs_receipts.digest_bytes(data),
        size=len(data),
    )


def bounded_text(
    path: Path, *, maximum: int = MAX_PRESERVED_BYTES
) -> Optional[str]:
    """Read a bounded UTF-8 tail, dropping any first partial line."""

    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise ValueError("maximum must be a positive integer")
    marker = f"{jx.TRUNCATED}\n"
    marker_size = len(marker.encode("utf-8"))
    if maximum <= marker_size:
        raise ValueError("maximum must leave room for the truncation marker")
    budget = maximum - marker_size
    source = Path(path)
    try:
        size = source.stat().st_size
        with source.open("rb") as handle:
            if size > budget:
                handle.seek(size - budget)
            raw = handle.read(budget)
    except OSError:
        return None
    text = raw.decode("utf-8", "replace")
    if size > budget:
        _, separator, rest = text.partition("\n")
        text = marker + (rest if separator else "")
    return text


def capped_utf8(text: str, *, maximum: int = MAX_PRESERVED_BYTES) -> bytes:
    """Encode a valid UTF-8 tail no larger than ``maximum`` bytes."""

    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise ValueError("maximum must be a positive integer")
    body = str(text).encode("utf-8")
    if len(body) <= maximum:
        return body
    body = body[-maximum:]
    while body and (body[0] & 0xC0) == 0x80:
        body = body[1:]
    return body
