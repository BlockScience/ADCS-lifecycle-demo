"""FlexoBackend — push the RTM Dataset into Flexo MMS via the Layer 1 API.

Maps each named graph in the local Dataset to a Flexo branch:
  <rtm:ontology>       -> branch `ontology`
  <adcs:structural>    -> branch `structural`
  <adcs:evidence>      -> branch `evidence`
  ...etc.

Workflow per Flexo's Layer 1 REST API (openmbee/flexo-mms-layer1-service):
  1. PUT /orgs/{org}                 (create org if absent)
  2. PUT /orgs/{org}/repos/{repo}    (create repo if absent)
  3. For each named graph as <branchId>:
       a. PUT /orgs/{org}/repos/{repo}/branches/{branchId}
          (with `<> dcterms:title "..."@en ; mms:ref <./master>` Turtle)
       b. POST /orgs/{org}/repos/{repo}/branches/{branchId}/update
          (with SPARQL `INSERT DATA { ... }`; PUT .../graph has a known
          bug that returns 200 but leaves the model empty)

Authentication: two supported modes
  - Pre-issued bearer token (FLEXO_TOKEN): used directly. Matches the
    remote-Flexo pattern at try-layer1.starforge.app.
  - Login flow (FLEXO_AUTH_URL + FLEXO_USER + FLEXO_PASS): GET /login
    against the auth service to retrieve a token, then use it. Matches
    the local Compose-up stack from flexo-mms-deployment.

Environment variables (kwargs override env):
  FLEXO_URL        Layer1 base URL                  default: https://try-layer1.starforge.app
  FLEXO_AUTH_URL   Auth service URL (login mode)    default: http://localhost:8082
  FLEXO_TOKEN      Pre-issued bearer token          default: (empty)
  FLEXO_USER       Username for login flow          default: user01
  FLEXO_PASS       Password for login flow          default: password1
  FLEXO_ORG        Org slug                         default: adcs-demo
  FLEXO_REPO       Repo slug                        default: lifecycle

The default URL targets the remote starforge instance, which is the
collaboration target shared with the team. Set FLEXO_TOKEN to a token
obtained from a collaborator. For a local Compose-up stack instead,
set FLEXO_URL=http://localhost:8080 and rely on the login flow.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from rdflib import Dataset, URIRef

from pipeline.backends.base import BackendUnavailable
from pipeline.dataset import triples_by_graph


# Mapping from named-graph IRI suffix to Flexo branch ID.
# Keeps branch IDs short and human-readable. When FLEXO_BRANCH_PREFIX
# is set (WP4 c14), every branch name is prefixed; enables
# "each run becomes a cert/<id> snapshot" workflows without forcing
# the pattern on the default single-canonical-state run.
def _branch_id(graph_iri: str, prefix: str = "") -> str:
    base = graph_iri.rstrip("/").rsplit("/", 1)[-1]
    return f"{prefix}{base}" if prefix else base


class FlexoBackend:
    name = "flexo"

    def __init__(
        self,
        url: str | None = None,
        auth_url: str | None = None,
        token: str | None = None,
        user: str | None = None,
        password: str | None = None,
        org: str | None = None,
        repo: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.url = (url or os.environ.get("FLEXO_URL", "https://try-layer1.starforge.app")).rstrip("/")
        self.auth_url = (auth_url or os.environ.get("FLEXO_AUTH_URL", "http://localhost:8082")).rstrip("/")
        self.token = token or os.environ.get("FLEXO_TOKEN", "") or None
        self.user = user or os.environ.get("FLEXO_USER", "user01")
        self.password = password or os.environ.get("FLEXO_PASS", "password1")
        self.org = org or os.environ.get("FLEXO_ORG", "adcs-demo")
        self.repo = repo or os.environ.get("FLEXO_REPO", "lifecycle")
        # Flexo INSERT DATA can be slow on larger graphs against a
        # shared remote; default 180s and let env override.
        if timeout is None:
            timeout = float(os.environ.get("FLEXO_TIMEOUT", "180"))
        self.timeout = timeout
        # Preflight probe is fast by design; separate, shorter timeout.
        self.probe_timeout = float(os.environ.get("FLEXO_PROBE_TIMEOUT", "10"))
        # WP4 c14 — optional branch prefix for multi-run isolation
        # (e.g. cert/2026-06-12-001/). Default empty preserves the
        # single-canonical-state behavior; set FLEXO_BRANCH_PREFIX or
        # pass branch_prefix= to scope branches per run.
        self.branch_prefix = os.environ.get("FLEXO_BRANCH_PREFIX", "")
        self._client: httpx.Client | None = None

    # --- Preflight -------------------------------------------------------

    def probe(self) -> None:
        """HEAD the org endpoint to verify Flexo is reachable + auth works.

        Cheap check: does NOT create the org or repo (persist() handles
        idempotent creation). Just verifies HTTP reachability and that
        the token (or login flow) yields a usable Authorization header.
        """
        try:
            with httpx.Client(timeout=self.probe_timeout) as client:
                try:
                    token = self._ensure_token(client)
                except (httpx.HTTPError, RuntimeError) as exc:
                    raise BackendUnavailable(
                        f"Flexo auth failed against {self.auth_url}: {exc}. "
                        f"Set FLEXO_TOKEN (pre-issued) or check FLEXO_USER/FLEXO_PASS."
                    ) from exc
                head = client.head(
                    f"{self.url}/orgs/{self.org}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                # 200/204 = org exists; 404 = org absent but server reachable
                # (persist() will create it). Anything else is a real problem.
                if head.status_code not in (200, 204, 404):
                    raise BackendUnavailable(
                        f"Flexo HEAD {self.url}/orgs/{self.org} returned "
                        f"{head.status_code}: {head.text[:200]}"
                    )
        except httpx.HTTPError as exc:
            raise BackendUnavailable(
                f"Flexo at {self.url} is unreachable: {exc}"
            ) from exc

    # --- Record URI (WP4 c4) ---------------------------------------------

    def record_uri(self, layer: str) -> URIRef | None:
        """IRI for the Flexo branch where this layer's graph lives.

        Layer name maps 1:1 to branch name (see `_branch_id`). Returns
        urn:adcs:flexo:<org>/<repo>/<branch> as a stable, joinable IRI
        consumers can resolve back to a Flexo REST path. Honors the
        configured branch prefix (WP4 c14).
        """
        # Layer-name → branch-name. The layer string IS the branch name
        # for the named graphs the backend persists (see persist()).
        branch = f"{self.branch_prefix}{layer}" if self.branch_prefix else layer
        return URIRef(f"urn:adcs:flexo:{self.org}/{self.repo}/{branch}")

    # --- Auth -------------------------------------------------------------

    def _login(self, client: httpx.Client) -> str:
        """Use FLEXO_USER/FLEXO_PASS against the auth service to get a token."""
        response = client.get(f"{self.auth_url}/login", auth=(self.user, self.password))
        response.raise_for_status()
        token = response.json().get("token")
        if not token:
            raise RuntimeError(f"Auth service at {self.auth_url} returned no token")
        return token

    def _ensure_token(self, client: httpx.Client) -> str:
        if self.token:
            return self.token
        self.token = self._login(client)
        return self.token

    def _headers(self, token: str, content_type: str = "text/turtle") -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
        }

    # --- Resource creation (idempotent) ----------------------------------

    def _put_resource(self, client: httpx.Client, token: str, url: str, title: str,
                       extra_body: str = "") -> None:
        """PUT a Flexo resource with `<> dcterms:title "..."@en` body.
        200 / 201 / 409 (already exists) are all acceptable."""
        body = f'<> <http://purl.org/dc/terms/title> "{title}"@en .{extra_body}'
        response = client.put(url, headers=self._headers(token), content=body)
        if response.status_code not in (200, 201, 409):
            response.raise_for_status()

    def _ensure_resource(self, client: httpx.Client, token: str,
                          url: str, title: str) -> None:
        """HEAD-first then PUT. Re-PUT of an existing resource on Flexo
        can trigger a slow update path that times out at Cloudflare
        (524); skipping it when the resource already exists is both
        faster and idempotent."""
        head = client.head(url, headers={"Authorization": f"Bearer {token}"})
        if head.status_code in (200, 204):
            return
        self._put_resource(client, token, url, title=title)

    def _ensure_org(self, client: httpx.Client, token: str) -> None:
        self._ensure_resource(client, token,
                              f"{self.url}/orgs/{self.org}", title=self.org)

    def _ensure_repo(self, client: httpx.Client, token: str) -> None:
        self._ensure_resource(client, token,
                              f"{self.url}/orgs/{self.org}/repos/{self.repo}",
                              title=self.repo)

    def _ensure_branch(self, client: httpx.Client, token: str, branch: str,
                        base: str = "master") -> None:
        """Create the branch if absent.

        Flexo creates `master` automatically with the repo, and rejects
        PUTs that don't carry an mms:ref / mms:commit. So:
          - For any branch (including master), HEAD first; if 200, skip.
          - For new branches, PUT with mms:ref pointing at `base`.
          - master itself is never recreated — only verified to exist.
        """
        branch_url = f"{self.url}/orgs/{self.org}/repos/{self.repo}/branches/{branch}"
        head = client.head(branch_url, headers={"Authorization": f"Bearer {token}"})
        if head.status_code in (200, 204):
            return  # already exists (Flexo returns 204 No Content on HEAD)
        if branch == base:
            # master should have been auto-created by the repo PUT.
            raise httpx.HTTPStatusError(
                f"branch {branch!r} missing on {self.url} (HEAD -> {head.status_code}); "
                f"repo creation should have auto-provisioned it",
                request=head.request, response=head,
            )
        extra = f'\n<> <https://mms.openmbee.org/rdf/ontology/ref> <./{base}> .'
        self._put_resource(client, token, branch_url, title=branch, extra_body=extra)

    # --- Data load -------------------------------------------------------

    def _load_graph(self, client: httpx.Client, token: str, branch: str,
                     graph: "Graph") -> None:
        """POST INSERT DATA with this graph's triples into the named branch."""
        from rdflib import Graph as _G  # local import for type hint compatibility

        # Re-serialize as N-Triples and wrap in INSERT DATA.
        # Using ntriples (line-oriented) avoids prefix-conflict issues in
        # the INSERT DATA block.
        nt = graph.serialize(format="nt")
        if isinstance(nt, bytes):
            nt = nt.decode("utf-8")

        # Empty graphs are a no-op
        nt_lines = [ln for ln in nt.splitlines() if ln.strip()]
        if not nt_lines:
            return
        body = "INSERT DATA {\n" + "\n".join(nt_lines) + "\n}"
        response = client.post(
            f"{self.url}/orgs/{self.org}/repos/{self.repo}/branches/{branch}/update",
            headers=self._headers(token, content_type="application/sparql-update"),
            content=body,
        )
        response.raise_for_status()

    # --- Public API ------------------------------------------------------

    def persist(self, ds: Dataset, output_dir: Path) -> dict:
        counts = triples_by_graph(ds)
        persisted: dict[str, int] = {}

        with httpx.Client(timeout=self.timeout) as client:
            token = self._ensure_token(client)
            self._ensure_org(client, token)
            self._ensure_repo(client, token)
            # master is auto-created with the repo — just verify it.
            self._ensure_branch(client, token, "master")

            for graph_iri, count in counts.items():
                branch = _branch_id(graph_iri, prefix=self.branch_prefix)
                self._ensure_branch(client, token, branch, base="master")
                self._load_graph(client, token, branch,
                                 ds.graph(URIRef(graph_iri)))
                persisted[graph_iri] = count

        return persisted

    def describe(self) -> str:
        token_origin = "pre-issued" if self.token else f"login@{self.auth_url}"
        return (
            f"Flexo MMS at {self.url} "
            f"(org={self.org} / repo={self.repo}; auth={token_origin})"
        )
