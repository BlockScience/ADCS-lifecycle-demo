"""TxnLogBackend tests — WP4 c10."""

from __future__ import annotations

import httpx
import pytest

from pipeline.backends.base import BackendUnavailable
from pipeline.backends.txnlog import SERVICE_IRI, TxnLogBackend


def test_service_iri_is_canonical():
    assert str(SERVICE_IRI) == "urn:adcs:service:transaction-log-store"


def test_describe_includes_url_and_db(monkeypatch):
    monkeypatch.delenv("ADCS_TXNLOG_URL", raising=False)
    backend = TxnLogBackend(url="http://couchdb.test", db="adcs-txnlogs", user="x", password="y")
    d = backend.describe()
    assert "couchdb.test" in d.lower() or "http" in d
    assert "adcs-txnlogs" in d


def test_probe_succeeds_when_db_exists(monkeypatch):
    """HEAD returns 204 → probe succeeds without creating the db."""
    calls = []
    def handler(request):
        calls.append((request.method, str(request.url)))
        if request.method == "HEAD":
            return httpx.Response(204)
        return httpx.Response(500)

    backend = TxnLogBackend(url="http://couchdb.test", db="adcs-txnlogs")
    orig_init = httpx.Client.__init__
    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        orig_init(self, *args, **kwargs)
    monkeypatch.setattr(httpx.Client, "__init__", patched_init)
    backend.probe()
    assert any(c[0] == "HEAD" for c in calls)
    # Did NOT need to PUT
    assert not any(c[0] == "PUT" for c in calls)


def test_probe_creates_db_on_404(monkeypatch):
    calls = []
    def handler(request):
        calls.append((request.method, str(request.url)))
        if request.method == "HEAD":
            return httpx.Response(404)
        if request.method == "PUT":
            return httpx.Response(201, json={"ok": True})
        return httpx.Response(500)

    backend = TxnLogBackend(url="http://couchdb.test", db="adcs-txnlogs")
    orig_init = httpx.Client.__init__
    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        orig_init(self, *args, **kwargs)
    monkeypatch.setattr(httpx.Client, "__init__", patched_init)
    backend.probe()
    assert any(c[0] == "PUT" for c in calls), "Expected PUT to create the db"


def test_probe_fails_on_unreachable(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("simulated")
    backend = TxnLogBackend(url="http://couchdb.test", db="adcs-txnlogs")
    orig_init = httpx.Client.__init__
    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        orig_init(self, *args, **kwargs)
    monkeypatch.setattr(httpx.Client, "__init__", patched_init)
    with pytest.raises(BackendUnavailable, match="unreachable"):
        backend.probe()


def test_probe_fails_on_auth_error(monkeypatch):
    def handler(request):
        return httpx.Response(401)
    backend = TxnLogBackend(url="http://couchdb.test", db="adcs-txnlogs")
    orig_init = httpx.Client.__init__
    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        orig_init(self, *args, **kwargs)
    monkeypatch.setattr(httpx.Client, "__init__", patched_init)
    with pytest.raises(BackendUnavailable, match="authentication"):
        backend.probe()


def test_put_document_returns_url(monkeypatch):
    def handler(request):
        if request.method == "PUT":
            return httpx.Response(201, json={"ok": True, "id": "txn-1", "rev": "1-abc"})
        return httpx.Response(500)
    backend = TxnLogBackend(url="http://couchdb.test", db="adcs-txnlogs")
    orig_init = httpx.Client.__init__
    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        orig_init(self, *args, **kwargs)
    monkeypatch.setattr(httpx.Client, "__init__", patched_init)
    url = backend.put_document("txn-1", {"method": "PUT /orgs", "response": {"status": 200}})
    assert url == "http://couchdb.test/adcs-txnlogs/txn-1"


def test_put_document_treats_409_as_success(monkeypatch):
    """Re-PUT same id → 409 conflict; treat as idempotent success."""
    def handler(request):
        return httpx.Response(409, json={"error": "conflict"})
    backend = TxnLogBackend(url="http://couchdb.test", db="adcs-txnlogs")
    orig_init = httpx.Client.__init__
    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        orig_init(self, *args, **kwargs)
    monkeypatch.setattr(httpx.Client, "__init__", patched_init)
    url = backend.put_document("txn-1", {"x": 1})
    assert url.endswith("/txn-1")
