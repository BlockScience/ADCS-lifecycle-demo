"""TransactionLogger tests — WP4 c11."""

from __future__ import annotations

import pytest
from rdflib import Dataset, URIRef
from rdflib.namespace import RDF

from ontology.prefixes import EARL, G_AUDIT, PROV, RTM
from traceability.transaction_log import (
    REDACTED,
    SENSITIVE_BODY_KEYS,
    SENSITIVE_HEADER_KEYS,
    TransactionLogger,
    _redact_body,
    _redact_headers,
)


class _FakeStore:
    """Stand-in for TxnLogBackend that records puts."""

    def __init__(self):
        self.puts: list[tuple[str, dict]] = []

    def put_document(self, doc_id: str, document: dict) -> str:
        self.puts.append((doc_id, document))
        return f"http://couchdb.test/adcs-txnlogs/{doc_id}"


def test_redact_headers_strips_sensitive_keys():
    headers = {"Authorization": "Bearer abc", "Content-Type": "text/turtle"}
    out = _redact_headers(headers)
    assert out["Authorization"] == REDACTED
    assert out["Content-Type"] == "text/turtle"


def test_redact_body_replaces_sensitive_keys_recursively():
    body = {
        "username": "alice",
        "password": "hunter2",
        "nested": {"token": "abc", "ok": "fine"},
        "list": [{"secret": "x"}, {"y": 1}],
    }
    out = _redact_body(body)
    assert out["password"] == REDACTED
    assert out["nested"]["token"] == REDACTED
    assert out["nested"]["ok"] == "fine"
    assert out["list"][0]["secret"] == REDACTED


def test_logger_emits_activity_and_evidence_with_store():
    ds = Dataset(default_union=True)
    store = _FakeStore()
    service = URIRef("urn:adcs:service:flexo-mms")
    caller = URIRef("urn:adcs:executor:test-host")

    with TransactionLogger(ds, store, service, caller, "PUT /orgs/foo") as logger:
        logger.set_request(body={"x": 1}, headers={"Authorization": "Bearer abc"})
        logger.set_response(status=200, body={"ok": True})

    # Document landed in the store with redacted authorization
    assert len(store.puts) == 1
    _, doc = store.puts[0]
    assert doc["request"]["headers"]["Authorization"] == REDACTED
    assert doc["method"] == "PUT /orgs/foo"
    assert doc["response"]["status"] == 200

    # RDF emitted in <adcs:audit>: activity + evidence
    g = ds.graph(URIRef(G_AUDIT))
    activities = list(g.subjects(RDF.type, PROV.Activity))
    assert len(activities) == 1
    activity = activities[0]
    assert (activity, PROV.used, service) in g
    assert (activity, PROV.wasAssociatedWith, caller) in g
    txn_ids = list(g.objects(activity, RTM.transactionId))
    assert len(txn_ids) == 1

    evidences = list(g.subjects(RDF.type, RTM.Evidence))
    assert len(evidences) == 1
    evidence = evidences[0]
    assert (evidence, PROV.wasGeneratedBy, activity) in g
    assert list(g.objects(evidence, RTM.contentHash))
    refs = list(g.objects(evidence, RTM.documentRef))
    assert refs and str(refs[0]).startswith("http://couchdb.test/")
    assert (evidence, PROV.atLocation, service) in g


def test_logger_skips_evidence_when_store_is_none():
    """Without a store, no documentRef → no rtm:Evidence emitted.

    Activity triples still land so the call is at least recorded.
    """
    ds = Dataset(default_union=True)
    service = URIRef("urn:adcs:service:flexo-mms")
    caller = URIRef("urn:adcs:executor:test-host")

    with TransactionLogger(ds, None, service, caller, "GET /") as logger:
        logger.set_response(status=200)

    g = ds.graph(URIRef(G_AUDIT))
    assert list(g.subjects(RDF.type, PROV.Activity))
    assert not list(g.subjects(RDF.type, RTM.Evidence))


def test_logger_captures_exception_in_document():
    ds = Dataset(default_union=True)
    store = _FakeStore()
    service = URIRef("urn:adcs:service:flexo-mms")
    caller = URIRef("urn:adcs:executor:test-host")

    with pytest.raises(RuntimeError):
        with TransactionLogger(ds, store, service, caller, "PUT /orgs/foo") as logger:
            logger.set_request(body={"x": 1})
            raise RuntimeError("simulated network failure")

    _, doc = store.puts[0]
    assert "exception" in doc
    assert doc["exception"]["type"] == "RuntimeError"
    assert "simulated" in doc["exception"]["message"]


def test_logger_swallows_store_failure_silently():
    """A store.put_document failure must NOT mask the user's call outcome."""
    ds = Dataset(default_union=True)

    class _BrokenStore:
        def put_document(self, doc_id, document):
            raise RuntimeError("couchdb down")

    service = URIRef("urn:adcs:service:flexo-mms")
    caller = URIRef("urn:adcs:executor:test-host")

    # Should not raise
    with TransactionLogger(ds, _BrokenStore(), service, caller, "PUT /orgs/foo") as logger:
        logger.set_response(status=200)

    # Activity emitted; evidence skipped (no doc_url available)
    g = ds.graph(URIRef(G_AUDIT))
    assert list(g.subjects(RDF.type, PROV.Activity))
    assert not list(g.subjects(RDF.type, RTM.Evidence))


def test_sensitive_key_constants_are_populated():
    """Sanity: the allowlists aren't empty."""
    assert "authorization" in SENSITIVE_HEADER_KEYS
    assert "password" in SENSITIVE_BODY_KEYS
