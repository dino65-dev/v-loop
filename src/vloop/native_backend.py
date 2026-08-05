"""Compatibility-preserving dispatcher for the optional Rust trusted kernel.

Only absence of the optional distribution permits an ``auto`` fallback.  A
loaded extension with the wrong ABI or an execution failure is surfaced to the
caller; silently changing security implementations after startup would make an
audit trail ambiguous.
"""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


API_VERSION = 1
BackendMode = Literal["auto", "off", "required"]


class NativeBackendUnavailable(RuntimeError):
    """The deployment explicitly required the native trusted kernel."""


@dataclass(frozen=True, slots=True)
class NativeStatus:
    mode: BackendMode
    active: bool
    implementation: str
    detail: str = ""


def _mode() -> BackendMode:
    mode = os.environ.get("VLOOP_NATIVE_BACKEND", "auto").lower()
    if mode not in {"auto", "off", "required"}:
        raise ValueError("VLOOP_NATIVE_BACKEND must be auto, off, or required")
    return mode  # type: ignore[return-value]


@lru_cache(maxsize=1)
def _native_module():
    mode = _mode()
    if mode == "off":
        return None
    try:
        from vloop_native import _core
    except ModuleNotFoundError as exc:
        # Only the optional distribution being absent is a permitted fallback.
        if exc.name not in {"vloop_native", "vloop_native._core"}:
            raise
        if mode == "required":
            raise NativeBackendUnavailable("V-Loop native core is required but not installed") from exc
        return None
    version = _core.api_version()
    if version != API_VERSION:
        raise NativeBackendUnavailable(
            f"V-Loop native core API {version} is incompatible with Python API {API_VERSION}"
        )
    return _core


def native_status() -> NativeStatus:
    module = _native_module()
    mode = _mode()
    return NativeStatus(mode, module is not None, "rust" if module is not None else "python")


def reset_native_backend_for_tests() -> None:
    """Clear only loader state after a controlled environment-variable test."""

    _native_module.cache_clear()


def sha256_hex(data: bytes) -> str:
    module = _native_module()
    return module.sha256_hex(data) if module is not None else hashlib.sha256(data).hexdigest()


def _private_key_bytes(private_key: bytes | Ed25519PrivateKey) -> bytes:
    if isinstance(private_key, bytes):
        return private_key
    return private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )


def _public_key_bytes(public_key: bytes | Ed25519PublicKey) -> bytes:
    if isinstance(public_key, bytes):
        return public_key
    return public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def ed25519_public_key(private_key: bytes | Ed25519PrivateKey) -> bytes:
    raw = _private_key_bytes(private_key)
    module = _native_module()
    if module is not None:
        return bytes(module.ed25519_public_key(raw))
    return Ed25519PrivateKey.from_private_bytes(raw).public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )


def ed25519_sign(private_key: bytes | Ed25519PrivateKey, payload: bytes) -> str:
    raw = _private_key_bytes(private_key)
    module = _native_module()
    if module is not None:
        return str(module.ed25519_sign(raw, payload))
    return base64.urlsafe_b64encode(Ed25519PrivateKey.from_private_bytes(raw).sign(payload)).decode("ascii")


def ed25519_verify(public_key: bytes | Ed25519PublicKey, payload: bytes, signature: str) -> bool:
    raw = _public_key_bytes(public_key)
    module = _native_module()
    if module is not None:
        return bool(module.ed25519_verify(raw, payload, signature))
    try:
        Ed25519PublicKey.from_public_bytes(raw).verify(base64.urlsafe_b64decode(signature.encode("ascii")), payload)
    except (InvalidSignature, ValueError):
        return False
    return True
