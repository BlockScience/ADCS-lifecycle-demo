"""TransactionLogger — capture service invocations as evidence.

WP4 §"Service-invocation events + transaction logs". Each cross-process
or cross-network service call (Flexo HTTP, Docker daemon subprocess,
reproduce subprocess) gets wrapped in a TransactionLogger. The logger:

1. Generates a unique transaction id at __enter__
2. Captures request/response payloads via set_request / set_response
3. On __exit__:
   a. PUTs a JSON document to the txnlog store (CouchDB or compatible)
   b. Emits RDF triples in <adcs:audit>:
        <activity> a prov:Activity ;
                   rtm:transactionId "<id>" ;
                   prov:wasAssociatedWith <caller> ;
                   prov:used <service> ;
                   prov:startedAtTime / endedAtTime
        <evidence> a rtm:Evidence ;
                   rtm:contentHash "sha256:..." ;
                   rtm:documentRef <store-url> ;
                   prov:atLocation <urn:adcs:service:transaction-log-store> ;
                   prov:wasGeneratedBy <activity>

Secrets in headers + body are replaced with literal "<REDACTED>" before
PUT, per the per-service allowlist below.

If `store` is None, the logger is a no-op (used for --backend=local where
the txnlog store isn't part of the canonical run).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from rdflib import Dataset, Literal, URIRef
from rdflib.namespace import RDF, XSD

from ontology.prefixes import G_AUDIT, PROV, RTM

# Sensitive header names (case-insensitive); replaced with <REDACTED>.
SENSITIVE_HEADER_KEYS = {
    "authorization", "cookie", "set-cookie",
    "x-api-key", "x-auth-token",
    "proxy-authorization",
}

# Sensitive body keys (case-insensitive in JSON). Replaced with <REDACTED>.
SENSITIVE_BODY_KEYS = {
    "password", "passwd", "token", "secret", "api_key", "apikey",
    "access_token", "refresh_token",
}

REDACTED = "<REDACTED>"


def _redact_headers(headers: dict[str, str] | None) -> dict[str, str]:
    if not headers:
        return {}
    return {
        k: REDACTED if k.lower() in SENSITIVE_HEADER_KEYS else v
        for k, v in headers.items()
    }


def _redact_body(body: Any) -> Any:
    """Walk a JSON-ish structure replacing sensitive values."""
    if isinstance(body, dict):
        return {
            k: REDACTED if k.lower() in SENSITIVE_BODY_KEYS else _redact_body(v)
            for k, v in body.items()
        }
    if isinstance(body, list):
        return [_redact_body(x) for x in body]
    return body


def _hash_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


class TransactionLogger:
    """Context manager that records one service invocation as evidence."""

    def __init__(
        self,
        ds: Dataset,
        store,  # TxnLogBackend | None
        service_iri: URIRef,
        caller_iri: URIRef,
        method: str,
    ) -> None:
        self.ds = ds
        self.store = store
        self.service_iri = service_iri
        self.caller_iri = caller_iri
        self.method = method
        self.txn_id = f"txn-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.started_at: str | None = None
        self.ended_at: str | None = None
        self.request: dict | None = None
        self.response: dict | None = None

    def __enter__(self) -> "TransactionLogger":
        self.started_at = datetime.now(timezone.utc).isoformat()
        return self

    def set_request(
        self,
        body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.request = {
            "headers": _redact_headers(headers),
            "body": _redact_body(body),
        }

    def set_response(
        self,
        status: int | str,
        body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.response = {
            "status": status,
            "headers": _redact_headers(headers),
            "body": _redact_body(body),
        }

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.ended_at = datetime.now(timezone.utc).isoformat()

        # If the caller didn't call set_response (e.g. an exception), still
        # record what we know — the wire log shows a partial invocation.
        document = {
            "transaction_id": self.txn_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "service": str(self.service_iri),
            "caller": str(self.caller_iri),
            "method": self.method,
            "request": self.request or {},
            "response": self.response or {"status": "incomplete"},
        }
        if exc_type is not None:
            document["exception"] = {
                "type": exc_type.__name__,
                "message": str(exc_val)[:500],
            }

        # PUT to the store (if configured); emit RDF either way
        doc_url: str | None = None
        content_hash: str | None = None
        if self.store is not None:
            payload = json.dumps(document, default=str, sort_keys=True).encode("utf-8")
            content_hash = _hash_bytes(payload)
            try:
                doc_url = self.store.put_document(self.txn_id, document)
            except Exception:
                # Don't let a txnlog failure crash the user's run — the
                # service call itself already succeeded or failed on its
                # own merits. Log path is best-effort.
                doc_url = None

        self._emit_rdf(doc_url=doc_url, content_hash=content_hash)

    def _emit_rdf(self, doc_url: str | None, content_hash: str | None) -> None:
        g = self.ds.graph(URIRef(G_AUDIT))

        activity = URIRef(f"urn:adcs:activity:invocation-{self.txn_id}")
        evidence = URIRef(f"urn:adcs:evidence:txn-{self.txn_id}")

        # Activity triples
        g.add((activity, RDF.type, PROV.Activity))
        g.add((activity, RTM.transactionId, Literal(self.txn_id)))
        g.add((activity, PROV.wasAssociatedWith, self.caller_iri))
        g.add((activity, PROV.used, self.service_iri))
        if self.started_at:
            g.add((activity, PROV.startedAtTime,
                   Literal(self.started_at, datatype=XSD.dateTime)))
        if self.ended_at:
            g.add((activity, PROV.endedAtTime,
                   Literal(self.ended_at, datatype=XSD.dateTime)))

        # Evidence triples — emitted only if we actually wrote a document
        # to the store (so rtm:contentHash + rtm:documentRef are populated
        # and the TransactionLogShape closure rule is satisfied).
        if doc_url and content_hash:
            g.add((evidence, RDF.type, RTM.Evidence))
            g.add((evidence, RTM.contentHash, Literal(content_hash)))
            g.add((evidence, RTM.documentRef,
                   Literal(doc_url, datatype=XSD.anyURI)))
            g.add((evidence, PROV.atLocation, self.service_iri))
            g.add((evidence, PROV.wasGeneratedBy, activity))
