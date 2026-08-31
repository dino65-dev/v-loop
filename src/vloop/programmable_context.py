"""Content-addressed, provenance-preserving context for advisory RLM workers.

This is deliberately a data API, not a Python execution environment.  Workers
can search and read only the handles admitted by a graph-bound request; every
derived value records its inputs and inherits the *least* trusted input's
authority ceiling.  That is stricter than the public ``<= max(inputs)``
invariant and prevents a summary from laundering untrusted instructions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable, Mapping
from urllib.parse import quote

from .canonical import digest
from .context import ContextItem, ContextPackage, ContextTrust


class ContextAuthority(IntEnum):
    """Maximum role a context value may play in a later proposal."""

    UNTRUSTED = 0
    USER = 1
    VERIFIED_MEMORY = 2
    TRUSTED = 3


_AUTHORITY_BY_TRUST = {
    ContextTrust.UNTRUSTED: ContextAuthority.UNTRUSTED,
    ContextTrust.USER: ContextAuthority.USER,
    ContextTrust.VERIFIED_MEMORY: ContextAuthority.VERIFIED_MEMORY,
    ContextTrust.TRUSTED_REPOSITORY: ContextAuthority.TRUSTED,
    ContextTrust.TRUSTED_SYSTEM: ContextAuthority.TRUSTED,
}


@dataclass(frozen=True, slots=True)
class ContextObject:
    handle: str
    content: str
    content_digest: str
    authority_ceiling: ContextAuthority
    provenance_roots: tuple[str, ...]
    transformation_id: str = "source"
    input_handles: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.handle.startswith("context://") or not self.content_digest:
            raise ValueError("context objects need a content-addressed context handle")
        if len(self.content_digest) != 64:
            raise ValueError("context object digest must be SHA-256")
        if not self.provenance_roots:
            raise ValueError("context objects need provenance roots")


@dataclass(frozen=True, slots=True)
class ContextManifest:
    contract_digest: str
    environment_digest: str
    handles: tuple[str, ...]
    object_digests: Mapping[str, str]

    @property
    def manifest_digest(self) -> str:
        return digest(
            {
                "contract_digest": self.contract_digest,
                "environment_digest": self.environment_digest,
                "handles": self.handles,
                "object_digests": dict(self.object_digests),
            }
        )


class ProgrammableContextStore:
    """In-memory immutable context store owned by the trusted worker adapter."""

    def __init__(self, *, contract_digest: str, environment_digest: str) -> None:
        if len(contract_digest) != 64 or len(environment_digest) != 64:
            raise ValueError("programmable context needs bound contract and environment digests")
        self.contract_digest = contract_digest
        self.environment_digest = environment_digest
        self._objects: dict[str, ContextObject] = {}

    @classmethod
    def from_package(cls, package: ContextPackage) -> "ProgrammableContextStore":
        store = cls(contract_digest=package.contract_digest, environment_digest=package.environment_digest)
        for item in (*package.trusted_items, *package.untrusted_items):
            store.add_item(item)
        return store

    def add_item(self, item: ContextItem, *, namespace: str = "source") -> str:
        handle = f"context://{namespace}/{quote(item.source_id, safe='._-')}/{item.content_digest}"
        self.add(
            ContextObject(
                handle=handle,
                content=item.content,
                content_digest=item.content_digest,
                authority_ceiling=_AUTHORITY_BY_TRUST[item.trust],
                provenance_roots=(item.content_digest,),
                metadata={"kind": item.kind, **dict(item.metadata)},
            )
        )
        return handle

    def add(self, value: ContextObject) -> None:
        existing = self._objects.get(value.handle)
        if existing is not None and existing != value:
            raise ValueError("context handle collision")
        self._objects[value.handle] = value

    def manifest(self, *, allowed_handles: Iterable[str] | None = None) -> ContextManifest:
        allowed = self._admitted(allowed_handles)
        return ContextManifest(
            self.contract_digest,
            self.environment_digest,
            tuple(sorted(allowed)),
            {handle: self._objects[handle].content_digest for handle in sorted(allowed)},
        )

    def search(self, query: str, *, allowed_handles: Iterable[str], limit: int = 8) -> tuple[str, ...]:
        if not query.strip() or not 1 <= limit <= 64:
            raise ValueError("search needs a query and bounded limit")
        terms = set(query.casefold().split())
        matches = []
        for handle in self._admitted(allowed_handles):
            value = self._objects[handle]
            score = len(terms.intersection(value.content.casefold().split()))
            if score:
                matches.append((-score, handle))
        return tuple(handle for _score, handle in sorted(matches)[:limit])

    def read(self, handle: str, *, allowed_handles: Iterable[str], start: int = 0, end: int | None = None) -> ContextObject:
        admitted = self._admitted(allowed_handles)
        if handle not in admitted:
            raise PermissionError("context handle was not admitted to this reasoning request")
        if start < 0 or (end is not None and end < start):
            raise ValueError("context read bounds are invalid")
        source = self._objects[handle]
        content = source.content[start:end]
        if start == 0 and end is None:
            return source
        return self._derive((source,), "slice", content, {"start": str(start), "end": "" if end is None else str(end)})

    def summarize(self, handles: Iterable[str], *, allowed_handles: Iterable[str], maximum_chars: int = 2_000) -> ContextObject:
        if not 1 <= maximum_chars <= 16_000:
            raise ValueError("summary size is outside the server-owned bounds")
        admitted = self._admitted(allowed_handles)
        selected = tuple(handles)
        if not selected or any(handle not in admitted for handle in selected):
            raise PermissionError("summary inputs were not admitted")
        inputs = tuple(self._objects[handle] for handle in selected)
        # The store makes a deterministic, inspectable packing.  A model may
        # later propose a prose summary, but cannot erase this provenance.
        content = "\n\n".join(value.content for value in inputs)[:maximum_chars]
        return self._derive(inputs, "deterministic-pack", content, {"maximum_chars": str(maximum_chars)})

    def compare(self, left: str, right: str, *, allowed_handles: Iterable[str]) -> ContextObject:
        admitted = self._admitted(allowed_handles)
        if left not in admitted or right not in admitted:
            raise PermissionError("comparison inputs were not admitted")
        a, b = self._objects[left], self._objects[right]
        content = f"equal={a.content == b.content}\nleft={a.content_digest}\nright={b.content_digest}"
        return self._derive((a, b), "compare", content, {})

    def _derive(
        self, inputs: tuple[ContextObject, ...], transformation_id: str, content: str, metadata: Mapping[str, str]
    ) -> ContextObject:
        authority = min(item.authority_ceiling for item in inputs)
        roots = tuple(sorted({root for item in inputs for root in item.provenance_roots}))
        value_digest = digest(
            {
                "content": content,
                "inputs": tuple(item.handle for item in inputs),
                "transformation_id": transformation_id,
                "metadata": dict(metadata),
            }
        )
        handle = f"context://derived/{transformation_id}/{value_digest}"
        result = ContextObject(handle, content, digest(content), authority, roots, transformation_id, tuple(item.handle for item in inputs), dict(metadata))
        self.add(result)
        return result

    def _admitted(self, allowed_handles: Iterable[str] | None) -> set[str]:
        allowed = set(self._objects) if allowed_handles is None else set(allowed_handles)
        unknown = allowed.difference(self._objects)
        if unknown:
            raise PermissionError("unknown context handles cannot be admitted")
        return allowed
