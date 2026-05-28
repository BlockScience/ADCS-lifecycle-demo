"""TxnLogBackend — CouchDB-backed service for transaction wire logs.

WP4 §"Service-invocation events + transaction logs". The fourth
service in the three-remote story: a separate container holding the
JSON wire-log documents that the TransactionLogger writes during
service invocations. Its own URI; its own organizational auspices;
demonstrated locally but architected to move under a different host
(or different org) with only env-var changes.

CouchDB is the demo's pick — JSON-native, Apache-2.0, trivially
deployable via `docker run couchdb:3`. Any HTTP document store
(MinIO, custom FastAPI, etc.) could replace it without changing the
RDF model.

Configuration via env (kwargs override):
  ADCS_TXNLOG_URL       base URL                    default http://localhost:5984
  ADCS_TXNLOG_DB        database name               default adcs-txnlogs
  ADCS_TXNLOG_USER      basic-auth username         default adcs
  ADCS_TXNLOG_PASSWORD  basic-auth password         default adcs
"""

from __future__ import annotations

import json
import os

import httpx
from rdflib import URIRef

from pipeline.backends.base import BackendUnavailable

SERVICE_IRI = URIRef("urn:adcs:service:transaction-log-store")


class TxnLogBackend:
    """Minimal CouchDB client for PUTting + GETting wire-log documents."""

    name = "txnlog"

    def __init__(
        self,
        url: str | None = None,
        db: str | None = None,
        user: str | None = None,
        password: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.url = (url or os.environ.get("ADCS_TXNLOG_URL", "http://localhost:5984")).rstrip("/")
        self.db = db or os.environ.get("ADCS_TXNLOG_DB", "adcs-txnlogs")
        self.user = user or os.environ.get("ADCS_TXNLOG_USER", "adcs")
        self.password = password or os.environ.get("ADCS_TXNLOG_PASSWORD", "adcs")
        self.timeout = timeout

    # --- Auth + URL helpers -----------------------------------------------

    def _auth(self) -> tuple[str, str] | None:
        if self.user and self.password:
            return (self.user, self.password)
        return None

    def _db_url(self) -> str:
        return f"{self.url}/{self.db}"

    def _doc_url(self, doc_id: str) -> str:
        return f"{self._db_url()}/{doc_id}"

    # --- Preflight --------------------------------------------------------

    def probe(self) -> None:
        """HEAD the db; create it on 404. Raise on auth or connection failure.

        Symmetric with FlexoBackend.probe — verifies reachability + auth
        without writing user data. CouchDB returns 401 on bad auth and
        404 on db-absent (which we treat as "create the db" since this
        is the txnlog store's responsibility, not the operator's).
        """
        auth = self._auth()
        try:
            with httpx.Client(timeout=self.timeout, auth=auth) as client:
                head = client.head(self._db_url())
                if head.status_code == 401:
                    raise BackendUnavailable(
                        f"txnlog store at {self.url} rejected authentication; "
                        f"check ADCS_TXNLOG_USER / ADCS_TXNLOG_PASSWORD."
                    )
                if head.status_code in (200, 204):
                    return
                if head.status_code == 404:
                    # Create the db idempotently
                    put = client.put(self._db_url())
                    if put.status_code not in (201, 202, 412):
                        # 412 Precondition Failed = race-created by another
                        # client between HEAD and PUT; treat as success.
                        raise BackendUnavailable(
                            f"txnlog store PUT {self._db_url()} failed: "
                            f"{put.status_code} {put.text[:200]}"
                        )
                    return
                raise BackendUnavailable(
                    f"txnlog store HEAD {self._db_url()} returned "
                    f"{head.status_code}: {head.text[:200]}"
                )
        except httpx.HTTPError as exc:
            raise BackendUnavailable(
                f"txnlog store at {self.url} is unreachable: {exc}"
            ) from exc

    # --- Document I/O -----------------------------------------------------

    def put_document(self, doc_id: str, document: dict) -> str:
        """PUT a JSON document under `doc_id`; return the resolvable URL.

        409 Conflict (doc already exists at the chosen id) is treated as
        idempotent success — txn ids are unique per run, so a conflict
        means an earlier identical write already happened.
        """
        url = self._doc_url(doc_id)
        with httpx.Client(timeout=self.timeout, auth=self._auth()) as client:
            response = client.put(
                url,
                headers={"Content-Type": "application/json"},
                content=json.dumps(document, default=str).encode("utf-8"),
            )
            if response.status_code in (201, 202, 409):
                return url
            response.raise_for_status()
            return url

    def get_document(self, doc_id: str) -> dict:
        url = self._doc_url(doc_id)
        with httpx.Client(timeout=self.timeout, auth=self._auth()) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()

    # --- StoreBackend protocol surface ------------------------------------

    def describe(self) -> str:
        return f"TxnLog store (CouchDB) at {self.url}/{self.db}"
