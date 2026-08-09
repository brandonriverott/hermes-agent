"""Canonical signed receipts for the Jobs reliability control plane."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import stat
import unicodedata
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import TypeAlias

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


JsonScalar: TypeAlias = str | int | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class CanonicalJSONError(ValueError):
    """Raised when a value is outside the receipt JSON profile."""


class ReceiptVerificationError(ValueError):
    """Raised when a receipt cannot authorize a state transition."""


class SigningKeyProtectionError(PermissionError):
    """Raised when a private signing key is not adequately protected."""


def _normalize(value: object) -> JsonValue:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        raise CanonicalJSONError("floats are not allowed in receipt JSON")
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise CanonicalJSONError("receipt object keys must be strings")
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise CanonicalJSONError("duplicate key after NFC normalization")
            normalized[normalized_key] = _normalize(item)
        return normalized
    raise CanonicalJSONError(f"unsupported receipt value: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    normalized = _normalize(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJSONError(f"duplicate receipt key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CanonicalJSONError(f"unsupported JSON number: {value}")


def loads_canonical(data: str | bytes) -> JsonValue:
    """Load receipt JSON while rejecting duplicate keys and invalid values."""

    try:
        loaded = json.loads(
            data,
            object_pairs_hook=_object_from_pairs,
            parse_constant=_reject_constant,
        )
    except CanonicalJSONError:
        raise
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalJSONError("invalid receipt JSON") from exc
    return _normalize(loaded)


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sign_receipt(
    payload: Mapping[str, JsonValue],
    *,
    receipt_id: str,
    key_id: str,
    private_key: Ed25519PrivateKey,
) -> dict[str, JsonValue]:
    """Return a canonical, identity-bindable Ed25519 receipt envelope."""

    if not isinstance(receipt_id, str) or not receipt_id:
        raise CanonicalJSONError("receipt_id must be a non-empty string")
    if not isinstance(key_id, str) or not key_id:
        raise CanonicalJSONError("key_id must be a non-empty string")
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("private_key must be an Ed25519PrivateKey")

    normalized_payload = _normalize(dict(payload))
    if not isinstance(normalized_payload, dict):
        raise CanonicalJSONError("receipt payload must be an object")
    payload_bytes = canonical_json_bytes(normalized_payload)
    signature = private_key.sign(payload_bytes)
    return {
        "schema_version": 1,
        "receipt_id": receipt_id,
        "payload": normalized_payload,
        "payload_digest": digest_bytes(payload_bytes),
        "signing": {
            "algorithm": "Ed25519",
            "key_id": key_id,
            "signature": "base64:" + base64.b64encode(signature).decode("ascii"),
        },
    }


def _verification_error(detail: str) -> ReceiptVerificationError:
    return ReceiptVerificationError(detail)


def verify_receipt(
    envelope: Mapping[str, JsonValue],
    *,
    trusted_keys: Mapping[str, Ed25519PublicKey],
    expected: Mapping[str, str],
    revoked_key_ids: Collection[str] = (),
) -> None:
    """Verify schema, canonical payload, signature, and expected identity."""

    try:
        if set(envelope) != {
            "schema_version",
            "receipt_id",
            "payload",
            "payload_digest",
            "signing",
        }:
            raise _verification_error("receipt envelope fields are invalid")
        if envelope["schema_version"] != 1:
            raise _verification_error("unsupported receipt schema_version")
        if not isinstance(envelope["receipt_id"], str) or not envelope["receipt_id"]:
            raise _verification_error("receipt_id is invalid")

        payload = envelope["payload"]
        normalized_payload = _normalize(payload)
        if not isinstance(payload, dict) or not isinstance(normalized_payload, dict):
            raise _verification_error("receipt payload must be an object")
        if normalized_payload != payload:
            raise _verification_error("receipt payload is not canonical")
        payload_bytes = canonical_json_bytes(payload)
        if envelope["payload_digest"] != digest_bytes(payload_bytes):
            raise _verification_error("receipt payload digest mismatch")

        signing = envelope["signing"]
        if not isinstance(signing, dict) or set(signing) != {
            "algorithm",
            "key_id",
            "signature",
        }:
            raise _verification_error("receipt signing fields are invalid")
        if signing["algorithm"] != "Ed25519":
            raise _verification_error("unsupported receipt signing algorithm")
        key_id = signing["key_id"]
        if not isinstance(key_id, str) or not key_id:
            raise _verification_error("receipt key_id is invalid")
        if key_id in revoked_key_ids:
            raise _verification_error(f"receipt key is revoked: {key_id}")
        public_key = trusted_keys.get(key_id)
        if public_key is None:
            raise _verification_error(f"unknown receipt key: {key_id}")
        if not isinstance(public_key, Ed25519PublicKey):
            raise _verification_error(f"invalid trusted receipt key: {key_id}")

        encoded_signature = signing["signature"]
        if not isinstance(encoded_signature, str) or not encoded_signature.startswith(
            "base64:"
        ):
            raise _verification_error("receipt signature encoding is invalid")
        try:
            signature = base64.b64decode(
                encoded_signature.removeprefix("base64:"), validate=True
            )
        except (binascii.Error, ValueError) as exc:
            raise _verification_error("receipt signature encoding is invalid") from exc
        try:
            public_key.verify(signature, payload_bytes)
        except (InvalidSignature, ValueError) as exc:
            raise _verification_error("receipt signature is invalid") from exc

        for field, expected_value in expected.items():
            if payload.get(field) != expected_value:
                raise _verification_error(f"receipt {field} identity mismatch")
    except ReceiptVerificationError:
        raise
    except (CanonicalJSONError, KeyError, TypeError, ValueError) as exc:
        raise _verification_error("invalid receipt envelope") from exc


def private_key_pem(private_key: Ed25519PrivateKey) -> bytes:
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("private_key must be an Ed25519PrivateKey")
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def load_private_key(path: Path) -> Ed25519PrivateKey:
    """Load a PEM key only when POSIX directory and file modes are exact."""

    path = Path(path)
    if os.name != "posix":
        raise SigningKeyProtectionError("SIGNING_KEY_UNPROTECTED")
    try:
        parent_stat = path.parent.stat()
        file_stat = path.lstat()
    except OSError as exc:
        raise SigningKeyProtectionError("SIGNING_KEY_UNPROTECTED") from exc
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
        or not stat.S_ISREG(file_stat.st_mode)
        or path.is_symlink()
        or stat.S_IMODE(file_stat.st_mode) != 0o600
    ):
        raise SigningKeyProtectionError("SIGNING_KEY_UNPROTECTED")
    try:
        loaded = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("invalid Ed25519 private key PEM") from exc
    if not isinstance(loaded, Ed25519PrivateKey):
        raise ValueError("private key is not Ed25519")
    return loaded


__all__ = [
    "CanonicalJSONError",
    "JsonScalar",
    "JsonValue",
    "ReceiptVerificationError",
    "SigningKeyProtectionError",
    "canonical_json_bytes",
    "digest_bytes",
    "load_private_key",
    "loads_canonical",
    "private_key_pem",
    "sign_receipt",
    "verify_receipt",
]
