"""ComputeBackend protocol and ExecutionMetadata dataclass."""

from __future__ import annotations

import platform
import socket
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from rdflib import URIRef


@dataclass(frozen=True)
class ExecutionMetadata:
    """Where and how an analysis stage was executed.

    Emitted as PROV triples on the corresponding rtm:SymbolicAnalysis or
    rtm:NumericalSimulation activity. Captures the audit-relevant facts:
    the host the compute actually ran on, the container image (if any)
    that pinned the toolchain, the container ID (for log retrieval),
    and a precise timestamp.

    URI contract (consumed by evidence.binding):
      executor_uri() -> urn:adcs:executor:<suffix>
      location_uri() -> urn:adcs:location:<location_kind>:<hostname-or-unknown>
    The shapes are byte-identical to pre-WP1 evidence/binding.py
    construction; centralizing them here lets WP3 / WP4 reuse the same
    IRI shape when adding new evidence types.
    """
    location_kind: str             # "local" | "docker" | "remote-server"
    hostname: str                  # uname -n inside the runner
    image_digest: str = ""         # sha256:... of the container image (Docker only)
    image_label: str = ""          # human-readable tag, e.g. adcs-compute:latest
    container_id: str = ""         # short docker container ID (Docker only)
    python_version: str = ""       # platform.python_version()
    started_at: str = ""           # ISO8601 UTC
    ended_at: str = ""             # ISO8601 UTC

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def executor_uri(self) -> URIRef:
        """Stable IRI for the prov:SoftwareAgent that ran this stage.

        Prefers container_id (Docker runs) and falls back to hostname
        (local runs); 'unknown' is the final sentinel. Colons in the
        suffix are replaced with dashes so the URN parses cleanly.
        """
        suffix = (self.container_id or self.hostname or "unknown").replace(":", "-")
        return URIRef(f"urn:adcs:executor:{suffix}")

    def location_uri(self) -> URIRef:
        """Stable IRI for the prov:Location where this stage ran."""
        host = self.hostname or "unknown"
        return URIRef(f"urn:adcs:location:{self.location_kind}:{host}")


@runtime_checkable
class ComputeBackend(Protocol):
    """Pluggable host for analysis stages.

    Implementations expose three methods that mirror the existing
    analysis.* entry points but additionally return ExecutionMetadata:

      run_symbolic_analysis(params) -> (sym_result, metadata)
      run_step_response(params)      -> (step_result, metadata)
      run_disturbance_rejection(params) -> (dist_result, metadata)
    """

    name: str

    def describe(self) -> str:
        ...

    def run_symbolic_analysis(self, params: dict) -> tuple[Any, ExecutionMetadata]:
        ...

    def run_step_response(self, params: dict) -> tuple[Any, ExecutionMetadata]:
        ...

    def run_disturbance_rejection(self, params: dict) -> tuple[Any, ExecutionMetadata]:
        ...


def _local_metadata(stage: str) -> ExecutionMetadata:
    """Build an ExecutionMetadata for an in-process run."""
    now = datetime.now(timezone.utc).isoformat()
    return ExecutionMetadata(
        location_kind="local",
        hostname=socket.gethostname(),
        python_version=platform.python_version(),
        started_at=now,
        ended_at=now,
    )


def get_compute_backend(name: str, **kwargs) -> ComputeBackend:
    """Factory. Imports backend modules lazily so docker isn't loaded
    when --compute=local."""
    if name == "local":
        from compute.local import LocalCompute
        return LocalCompute(**kwargs)
    if name == "docker":
        from compute.docker_compute import DockerCompute
        return DockerCompute(**kwargs)
    raise ValueError(f"Unknown compute backend {name!r}. Choose: local | docker")
