"""Phase I — backend abstraction tests.

LocalBackend is exercised live (it writes to a tmp_path). FlexoBackend
and FuskeiBackend use httpx.MockTransport so the API call sequence is
asserted without needing a running Flexo / Fuseki instance. Live tests
against an actual stack land in Phase J.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import httpx
import pytest
from rdflib import Dataset, URIRef

from ontology.prefixes import G_EVIDENCE, G_ONTOLOGY, NAMED_GRAPHS
from pipeline.backends import get_backend
from pipeline.backends.flexo import FlexoBackend
from pipeline.backends.fuseki import FuskeiBackend
from pipeline.backends.local import LocalBackend
from pipeline.runner import run_pipeline


@pytest.fixture(scope="module")
def pipeline_dataset() -> Dataset:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return run_pipeline(auto_attest=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_factory_returns_local():
    assert isinstance(get_backend("local"), LocalBackend)


def test_factory_returns_flexo():
    backend = get_backend("flexo")
    assert isinstance(backend, FlexoBackend)


def test_factory_returns_fuseki():
    backend = get_backend("fuseki")
    assert isinstance(backend, FuskeiBackend)


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("not-a-backend")


# ---------------------------------------------------------------------------
# LocalBackend (live)
# ---------------------------------------------------------------------------

def test_local_backend_writes_ttl_and_trig(pipeline_dataset, tmp_path):
    backend = LocalBackend()
    counts = backend.persist(pipeline_dataset, tmp_path)
    assert (tmp_path / "rtm.ttl").exists()
    assert (tmp_path / "rtm.trig").exists()
    # Returned counts cover the named graphs we expect
    assert any(G_ONTOLOGY in iri for iri in counts)
    assert any(G_EVIDENCE in iri for iri in counts)


def test_local_backend_describe():
    assert "Local filesystem" in LocalBackend().describe()


# ---------------------------------------------------------------------------
# FlexoBackend (mocked httpx)
# ---------------------------------------------------------------------------

def _flexo_mock_transport(record: list[dict]) -> httpx.MockTransport:
    """A MockTransport that records every request and simulates a
    Flexo instance with no pre-existing resources.

    HEAD requests return 404 (resources don't exist yet) so the
    head-first ensure_* code proceeds to PUT. For master specifically
    we simulate "auto-created with the repo" by returning 200 only
    AFTER the repo PUT has been recorded.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        record.append({
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "content": request.content.decode("utf-8", errors="replace"),
        })
        # /login returns a token
        if request.url.path.endswith("/login"):
            return httpx.Response(200, json={"token": "fake-token-abc123"})
        # HEAD: simulate fresh state, except master returns 204 once
        # the repo PUT has happened (master is auto-created with the repo).
        if request.method == "HEAD":
            path = request.url.path
            if path.endswith("/branches/master"):
                repo_put_seen = any(
                    r["method"] == "PUT"
                    and r["url"].endswith("/repos/lifecycle")
                    for r in record
                )
                return httpx.Response(204 if repo_put_seen else 404)
            return httpx.Response(404)
        # All other (PUT, POST, GET) requests succeed
        return httpx.Response(200, text="")
    return httpx.MockTransport(handler)


def test_flexo_backend_uses_pre_issued_token_when_present(pipeline_dataset, monkeypatch):
    """If FLEXO_TOKEN is set, no /login call is made."""
    record: list[dict] = []
    monkeypatch.setenv("FLEXO_TOKEN", "preset-token-xyz")

    backend = FlexoBackend(url="http://flexo.test", auth_url="http://auth.test")
    # Patch httpx.Client to use our mock transport
    orig_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = _flexo_mock_transport(record)
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)
    backend.persist(pipeline_dataset, Path("."))

    login_calls = [r for r in record if r["url"].endswith("/login")]
    assert not login_calls, "should not call /login when FLEXO_TOKEN is set"
    # Token from env appears in auth headers
    assert all(
        r["headers"].get("authorization") == "Bearer preset-token-xyz"
        for r in record
    ), "all requests must carry the pre-issued token"


def test_flexo_backend_logs_in_when_no_token(pipeline_dataset, monkeypatch):
    """Without FLEXO_TOKEN, the backend calls /login first."""
    record: list[dict] = []
    monkeypatch.delenv("FLEXO_TOKEN", raising=False)

    backend = FlexoBackend(url="http://flexo.test", auth_url="http://auth.test",
                            user="alice", password="secret")
    orig_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = _flexo_mock_transport(record)
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)
    backend.persist(pipeline_dataset, Path("."))

    login_calls = [r for r in record if r["url"].endswith("/login")]
    assert len(login_calls) == 1, "expected exactly one /login call"


def test_flexo_backend_creates_org_repo_branches(pipeline_dataset, monkeypatch):
    """The backend ensures org, repo, master branch, and one branch per
    named graph via idempotent PUTs, then loads data via POST .../update."""
    record: list[dict] = []
    monkeypatch.setenv("FLEXO_TOKEN", "fake")
    backend = FlexoBackend(url="http://flexo.test", org="adcs-demo", repo="lifecycle")

    orig_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = _flexo_mock_transport(record)
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)
    backend.persist(pipeline_dataset, Path("."))

    puts = [r for r in record if r["method"] == "PUT"]
    posts = [r for r in record if r["method"] == "POST"]

    put_paths = [r["url"].replace("http://flexo.test", "") for r in puts]
    assert "/orgs/adcs-demo" in put_paths, "org PUT missing"
    assert "/orgs/adcs-demo/repos/lifecycle" in put_paths, "repo PUT missing"

    # master is auto-created by the repo PUT on the real Flexo, so the
    # backend only HEADs to verify rather than PUTting. Confirm we did
    # HEAD it, and we did NOT try to PUT it (PUT would 400 — see the
    # ConstraintViolationException debugged against starforge during
    # Phase J).
    heads = [r for r in record if r["method"] == "HEAD"]
    head_paths = [r["url"].replace("http://flexo.test", "") for r in heads]
    assert "/orgs/adcs-demo/repos/lifecycle/branches/master" in head_paths, (
        "master branch HEAD missing"
    )
    assert "/orgs/adcs-demo/repos/lifecycle/branches/master" not in put_paths, (
        "master branch should NOT be PUT — Flexo auto-creates it with the repo"
    )

    # Data is loaded via SPARQL UPDATE (not via PUT /graph), per the
    # known Flexo PUT/graph quirk documented in
    # flexo-conflict-resolution-policy-research/experiments/experiment-1.
    update_posts = [r for r in posts if r["url"].endswith("/update")]
    assert update_posts, "expected POST .../update calls for data load"
    assert all(
        r["headers"].get("content-type") == "application/sparql-update"
        for r in update_posts
    ), "data-load POSTs must use application/sparql-update content type"
    assert all("INSERT DATA" in r["content"] for r in update_posts), (
        "data-load POSTs must use SPARQL INSERT DATA"
    )


def test_flexo_backend_describe_reflects_token_origin():
    assert "pre-issued" in FlexoBackend(token="abc").describe()
    assert "login@" in FlexoBackend(token=None).describe()


# ---------------------------------------------------------------------------
# FuskeiBackend (mocked httpx)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# record_uri (WP4 c4) — backend-specific URI for "where does this layer live?"
# ---------------------------------------------------------------------------

def test_local_backend_record_uri_returns_none():
    """LocalBackend has no remote IRI; rtm:flexoRecord is omitted."""
    assert LocalBackend().record_uri("evidence") is None


def test_flexo_backend_record_uri_shape():
    """FlexoBackend.record_uri(layer) → urn:adcs:flexo:<org>/<repo>/<layer>."""
    backend = FlexoBackend(url="http://flexo.test", org="adcs-demo", repo="lifecycle")
    assert str(backend.record_uri("evidence")) == "urn:adcs:flexo:adcs-demo/lifecycle/evidence"
    assert str(backend.record_uri("attestations")) == "urn:adcs:flexo:adcs-demo/lifecycle/attestations"


def test_flexo_backend_branch_prefix_applies(monkeypatch):
    """WP4 c14 — FLEXO_BRANCH_PREFIX shapes branch names + record_uri."""
    monkeypatch.setenv("FLEXO_BRANCH_PREFIX", "cert/2026-06-12-001/")
    backend = FlexoBackend(url="http://flexo.test", org="adcs-demo", repo="lifecycle")
    assert backend.branch_prefix == "cert/2026-06-12-001/"
    assert str(backend.record_uri("evidence")) == (
        "urn:adcs:flexo:adcs-demo/lifecycle/cert/2026-06-12-001/evidence"
    )


def test_fuseki_backend_record_uri_shape():
    """FuskeiBackend.record_uri encodes the base URL + layer name."""
    backend = FuskeiBackend(url="http://fuseki.test/adcs")
    uri = str(backend.record_uri("evidence"))
    assert uri.startswith("urn:adcs:fuseki:")
    assert uri.endswith("/evidence")


# ---------------------------------------------------------------------------
# Probe (WP4 §4.1) — preflight reachability checks
# ---------------------------------------------------------------------------

def test_local_backend_probe_succeeds_on_writable_dir(tmp_path):
    """LocalBackend.probe() writes + deletes a .probe sentinel."""
    LocalBackend().probe(output_dir=tmp_path)
    # Sentinel cleaned up
    assert not (tmp_path / ".probe").exists()


def test_local_backend_probe_fails_on_unwritable_dir(tmp_path, monkeypatch):
    """probe() raises BackendUnavailable if the directory is not writable."""
    from pipeline.backends.base import BackendUnavailable

    target = tmp_path / "readonly"
    target.mkdir()
    target.chmod(0o555)
    try:
        with pytest.raises(BackendUnavailable, match="not writable"):
            LocalBackend().probe(output_dir=target)
    finally:
        target.chmod(0o755)  # restore so pytest can clean up


def test_flexo_backend_probe_succeeds_on_reachable_org(monkeypatch):
    """probe() HEADs the org endpoint; 200/204/404 are all OK."""
    record: list[dict] = []
    monkeypatch.setenv("FLEXO_TOKEN", "fake-token")
    backend = FlexoBackend(url="http://flexo.test", org="adcs-demo")

    orig_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = _flexo_mock_transport(record)
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)
    backend.probe()  # should not raise

    heads = [r for r in record if r["method"] == "HEAD"]
    head_paths = [r["url"].replace("http://flexo.test", "") for r in heads]
    assert "/orgs/adcs-demo" in head_paths, (
        "probe must HEAD /orgs/<org> to verify reachability"
    )


def test_flexo_backend_probe_fails_on_unreachable_url(monkeypatch):
    """probe() raises BackendUnavailable when HTTP fails."""
    from pipeline.backends.base import BackendUnavailable

    monkeypatch.setenv("FLEXO_TOKEN", "fake-token")
    backend = FlexoBackend(url="http://flexo.test")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated connection refused")

    orig_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)
    with pytest.raises(BackendUnavailable, match="unreachable"):
        backend.probe()


def test_flexo_backend_probe_fails_on_bad_auth(monkeypatch):
    """probe() raises BackendUnavailable when /login returns no token."""
    from pipeline.backends.base import BackendUnavailable

    monkeypatch.delenv("FLEXO_TOKEN", raising=False)
    backend = FlexoBackend(url="http://flexo.test", auth_url="http://auth.test")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/login"):
            return httpx.Response(200, json={})  # no token in response
        return httpx.Response(200)

    orig_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)
    with pytest.raises(BackendUnavailable, match="auth failed"):
        backend.probe()


def test_fuseki_backend_probe_succeeds(monkeypatch):
    """Fuseki probe HEADs /data; 200 succeeds."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    backend = FuskeiBackend(url="http://fuseki.test/adcs")
    orig_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)
    backend.probe()  # should not raise


def test_fuseki_backend_probe_fails_on_unreachable(monkeypatch):
    """Fuseki probe raises BackendUnavailable when HTTP fails."""
    from pipeline.backends.base import BackendUnavailable

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated")

    backend = FuskeiBackend(url="http://fuseki.test/adcs")
    orig_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)
    with pytest.raises(BackendUnavailable, match="unreachable"):
        backend.probe()


# ---------------------------------------------------------------------------
# FuskeiBackend (mocked httpx) — persist path
# ---------------------------------------------------------------------------

def test_fuseki_backend_puts_via_graph_store_protocol(pipeline_dataset, monkeypatch):
    """Each named graph is PUT to <base>/data?graph=<iri> with Turtle."""
    record: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        record.append({
            "method": request.method,
            "url": str(request.url),
            "params": dict(request.url.params),
            "headers": dict(request.headers),
        })
        return httpx.Response(200)

    backend = FuskeiBackend(url="http://fuseki.test/adcs")
    orig_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)
    backend.persist(pipeline_dataset, Path("."))

    puts = [r for r in record if r["method"] == "PUT"]
    assert puts, "expected PUT calls"
    assert all(r["url"].startswith("http://fuseki.test/adcs/data") for r in puts)
    # Each PUT carries the graph IRI as a query parameter
    assert all("graph" in r["params"] for r in puts)
    # Content type is text/turtle
    assert all(r["headers"].get("content-type") == "text/turtle" for r in puts)
