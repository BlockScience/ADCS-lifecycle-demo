"""LocalBackend — persist the Dataset as files on disk."""

from __future__ import annotations

from pathlib import Path

from rdflib import Dataset, URIRef

from pipeline.backends.base import BackendUnavailable
from pipeline.dataset import export_trig, export_union_turtle, triples_by_graph

_DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent.parent / "output"


class LocalBackend:
    name = "local"

    def probe(self, output_dir: Path | None = None) -> None:
        """Verify the output directory is writable (preflight check)."""
        target = Path(output_dir) if output_dir is not None else _DEFAULT_OUTPUT
        try:
            target.mkdir(parents=True, exist_ok=True)
            sentinel = target / ".probe"
            sentinel.write_text("ok\n")
            sentinel.unlink()
        except OSError as exc:
            raise BackendUnavailable(
                f"local output directory {target} is not writable: {exc}"
            ) from exc

    def persist(self, ds: Dataset, output_dir: Path) -> dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        export_union_turtle(ds, output_dir / "rtm.ttl")
        export_trig(ds, output_dir / "rtm.trig")
        return triples_by_graph(ds)

    def record_uri(self, layer: str) -> URIRef | None:
        """LocalBackend has no remote IRI; rtm:flexoRecord is omitted."""
        return None

    def describe(self) -> str:
        return "Local filesystem (rdflib Dataset, persisted as output/rtm.ttl + output/rtm.trig)"
