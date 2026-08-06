"""Minimal, zero-dependency HTTP client for the yq-nova Agent memory service.

Uses only the Python standard library (``urllib.request`` / ``json`` /
``urllib.error``) so it can run anywhere Python 3.8+ is available.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional


class NovaApiError(Exception):
    """Raised when the server returns a non-2xx response.

    Attributes:
        code: The server-provided error code (e.g. ``"validation"``,
            ``"not_found"``, ``"conflict"``), or ``"http_{status}"`` when the
            body could not be parsed as a structured error.
        message: Human-readable error message from the server.
        status: The HTTP status code (int).
        trace_id: Optional trace id returned by the server, if any.
    """

    def __init__(
        self,
        code: str,
        message: str,
        status: int,
        trace_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.trace_id = trace_id

    def __str__(self) -> str:
        return f"NovaApiError(status={self.status}, code={self.code!r}, message={self.message!r})"


class Client:
    """A thin HTTP client for a running yq-nova server.

    All requests are made against ``base_url`` (trailing ``/`` stripped).
    If an ``api_key`` is supplied it is sent as a ``Authorization: Bearer``
    header on every request, matching the server auth layer.

    Every public method returns the parsed JSON body as a Python
    ``dict`` / ``list``. Non-2xx responses raise :class:`NovaApiError`.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        base = base_url.rstrip("/")
        if not base:
            raise ValueError("base_url must not be empty")
        self.base_url = base
        self.api_key = api_key
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    # low-level request helper
    # ------------------------------------------------------------------ #
    def _request(
        self,
        method: str,
        path: str,
        body: Any = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Send a JSON request and return the parsed JSON response.

        Args:
            method: HTTP method (``GET`` / ``POST`` / ``DELETE``).
            path: Path starting with ``/``, e.g. ``/v1/health``.
            body: JSON-serializable request body (optional).
            params: Optional query-string parameters.

        Returns:
            The parsed JSON response (dict / list / scalar).

        Raises:
            NovaApiError: If the server returns a non-2xx status.
            ValueError: If ``body`` is not JSON-serializable.
        """
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            raw = e.read()
            status = e.code
            server_err = _try_parse_error(raw)
            if server_err is not None:
                code, message, trace_id = server_err
            else:
                code = f"http_{status}"
                message = raw.decode("utf-8", errors="replace") or (
                    f"HTTP {status} returned no body"
                )
                trace_id = None
            raise NovaApiError(code, message, status, trace_id) from None
        except urllib.error.URLError as e:
            raise NovaApiError(
                code="network",
                message=f"request to {url} failed: {e.reason}",
                status=0,
            ) from e

        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    # ------------------------------------------------------------------ #
    # meta endpoints
    # ------------------------------------------------------------------ #
    def health(self) -> Dict[str, Any]:
        """Return the server health info.

        Response: ``{"status": "ok", "version": ..., "git_sha": ..., "uptime_secs": ...}``
        """
        return self._request("GET", "/v1/health")

    def stats(self) -> Dict[str, Any]:
        """Return coarse runtime counters (db size, memory/graph/tag counts)."""
        return self._request("GET", "/v1/stats")

    # ------------------------------------------------------------------ #
    # memory endpoints
    # ------------------------------------------------------------------ #
    def remember(
        self,
        content: str,
        source: str = "agent",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[str] = None,
        tags: Optional[List[str]] = None,
        embed: bool = True,
        extract_graph: bool = False,
    ) -> Dict[str, Any]:
        """Store a memory record.

        Args:
            content: The raw text to remember (required, non-empty).
            source: Origin of the memory (e.g. ``"agent"``, ``"user"``).
            importance: 0.0 ~ 1.0 importance score.
            metadata: Arbitrary structured metadata (or ``None``).
            expires_at: ISO-8601 UTC expiry string, or ``None`` for never.
            tags: Optional list of user tags.
            embed: Whether to embed ``content`` for semantic search.
            extract_graph: Whether to auto-extract entities/relations.

        Returns:
            ``{"uuid": ..., "duplicate": bool, "embedding_stored": bool,
            "entities_extracted": int, "relations_extracted": int, "tags": [...]}``
        """
        if not content or not content.strip():
            raise ValueError("remember: content must be non-empty")
        body = {
            "content": content,
            "source": source,
            "importance": importance,
            "metadata": metadata,
            "expires_at": expires_at,
            "tags": tags or [],
            "embed": embed,
            "extract_graph": extract_graph,
        }
        return self._request("POST", "/v1/memory/remember", body)

    def recall(
        self,
        query: str,
        top_k: int = 20,
        mode: str = "semantic",
        score_threshold: float = 0.0,
        similarity_threshold: float = -1.0,
    ) -> Dict[str, Any]:
        """Retrieve memories matching ``query``.

        Args:
            query: The search text (required, non-empty).
            top_k: Maximum number of results to return (>= 1).
            mode: ``"semantic"``, ``"keyword"`` or ``"hybrid"``.
            score_threshold: Final-score cutoff.
            similarity_threshold: Raw vector-similarity cutoff.

        Returns:
            ``{"hits": [...], "total_candidates": int, "query": str}``
        """
        if not query or not query.strip():
            raise ValueError("recall: query must be non-empty")
        if top_k < 1:
            raise ValueError("recall: top_k must be >= 1")
        body = {
            "query": query,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "similarity_threshold": similarity_threshold,
            "mode": mode,
            "graph": {"enabled": False, "max_depth": 1, "predicate_whitelist": []},
            "hybrid_weights": None,
            "rrf_k": None,
            "rank_weights": None,
            "filter": {},
        }
        return self._request("POST", "/v1/memory/recall", body)

    def forget(
        self,
        uuid: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        mode: str = "soft",
        gc_graph: bool = False,
        batch_limit: int = 500,
    ) -> Dict[str, Any]:
        """Forget / archive memories.

        Args:
            uuid: Target a single memory by uuid.
            filter: Target memories by a filter dict (mutually exclusive with
                ``uuid``).
            mode: ``"soft"``, ``"hard"`` or ``"archive"``.
            gc_graph: Whether to cascade-clean orphan graph nodes.
            batch_limit: Max number of memories to process in one pass.

        Returns:
            ``{"affected_memories": int, "cascade_embeddings": int,
            "gc_entities": int, "gc_relations": int, "mode": str}``
        """
        if uuid is not None:
            target: Dict[str, Any] = {"type": "one", "value": uuid}
        elif filter is not None:
            target = {"type": "filter", "value": filter}
        else:
            raise ValueError("forget: either uuid or filter must be provided")
        body = {
            "target": target,
            "mode": mode,
            "gc_graph": gc_graph,
            "batch_limit": batch_limit,
        }
        return self._request("POST", "/v1/memory/forget", body)

    def get_memory(self, uuid: str) -> Dict[str, Any]:
        """Fetch a single memory record by uuid."""
        if not uuid:
            raise ValueError("get_memory: uuid must be non-empty")
        return self._request("GET", f"/v1/memory/{_quote(uuid)}")

    def delete_memory(self, uuid: str) -> Dict[str, Any]:
        """Hard-delete a single memory by uuid."""
        if not uuid:
            raise ValueError("delete_memory: uuid must be non-empty")
        return self._request("DELETE", f"/v1/memory/{_quote(uuid)}")

    # ------------------------------------------------------------------ #
    # graph endpoints
    # ------------------------------------------------------------------ #
    def upsert_entity(
        self,
        name: str,
        entity_type: str = "generic",
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create or update a graph entity keyed by ``(name, entity_type)``.

        Returns:
            ``{"outcome": {...}, "entity": {...}}``
        """
        if not name or not name.strip():
            raise ValueError("upsert_entity: name must be non-empty")
        body = {
            "name": name,
            "type": entity_type,
            "description": description,
            "metadata": metadata,
        }
        return self._request("POST", "/v1/graph/entities", body)

    def traverse(
        self,
        start: str,
        max_depth: int = 3,
        max_nodes: int = 100,
        predicate_whitelist: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """BFS-traverse the graph from ``start`` and return reachable nodes.

        Returns: A list of traverse node dicts.
        """
        if not start:
            raise ValueError("traverse: start uuid must be non-empty")
        body = {
            "start": start,
            "max_depth": max_depth,
            "max_nodes": max_nodes,
            "predicate_whitelist": predicate_whitelist or [],
            "min_confidence": min_confidence,
        }
        return self._request("POST", "/v1/graph/traverse", body)

    def extract_and_link(
        self,
        text: str,
        opts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract entities (and optionally link them) from free text.

        Args:
            text: The raw text to analyze.
            opts: Optional dict with ``enabled``, ``upsert_entities``,
                ``create_relations``, ``min_confidence`` keys.

        Returns:
            The extract-and-link result dict.
        """
        if not text or not text.strip():
            raise ValueError("extract_and_link: text must be non-empty")
        default_opts = {
            "enabled": True,
            "upsert_entities": True,
            "create_relations": False,
            "min_confidence": 0.0,
        }
        merged = dict(default_opts)
        if opts:
            merged.update(opts)
        body = {"text": text, "opts": merged}
        return self._request("POST", "/v1/graph/extract-and-link", body)


def _quote(value: str) -> str:
    """Percent-encode a path segment."""
    return urllib.parse.quote(value, safe="")


def _try_parse_error(raw: bytes):
    """Best-effort parse of a server error body.

    Returns ``(code, message, trace_id)`` or ``None`` if not parseable.
    """
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    code = data.get("code")
    message = data.get("message")
    if code is None or message is None:
        return None
    return code, str(message), data.get("trace_id")