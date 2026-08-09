import copy

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_cli import jobs_receipts as receipts


def test_canonical_json_vector_normalizes_strings_and_sorts_keys():
    value = {"z": [2, 1], "text": "e\u0301", "a": {"ok": True}}
    assert receipts.canonical_json_bytes(value) == (
        '{"a":{"ok":true},"text":"é","z":[2,1]}'.encode("utf-8")
    )


@pytest.mark.parametrize("value", [1.5, float("nan"), float("inf"), {1: "bad"}])
def test_canonical_json_rejects_unsupported_values(value):
    with pytest.raises(receipts.CanonicalJSONError):
        receipts.canonical_json_bytes(value)


def test_duplicate_keys_are_rejected_on_load():
    with pytest.raises(receipts.CanonicalJSONError):
        receipts.loads_canonical('{"a":1,"a":2}')


def test_normalized_duplicate_keys_are_rejected():
    with pytest.raises(receipts.CanonicalJSONError):
        receipts.canonical_json_bytes({"e\u0301": 1, "é": 2})


def _payload():
    return {
        "job_id": "j_1",
        "attempt_id": "a_1",
        "commit": "c" * 40,
        "state": "VERIFIED",
        "evidence": {},
        "timestamp": "2026-08-09T00:00:00Z",
        "transitioned_by": "jobs_graph/1",
    }


def _signed():
    private_key = Ed25519PrivateKey.generate()
    envelope = receipts.sign_receipt(
        _payload(),
        receipt_id="r_1",
        key_id="lane:test:v1",
        private_key=private_key,
    )
    return private_key, envelope


def test_signed_receipt_verifies_and_binds_identity():
    private_key, envelope = _signed()
    receipts.verify_receipt(
        envelope,
        trusted_keys={"lane:test:v1": private_key.public_key()},
        expected={"job_id": "j_1", "attempt_id": "a_1", "commit": "c" * 40},
    )
    assert envelope["payload_digest"].startswith("sha256:")
    assert envelope["signing"]["signature"].startswith("base64:")

    with pytest.raises(receipts.ReceiptVerificationError, match="attempt_id"):
        receipts.verify_receipt(
            envelope,
            trusted_keys={"lane:test:v1": private_key.public_key()},
            expected={"attempt_id": "a_other"},
        )


@pytest.mark.parametrize("mutation", ["schema", "digest", "signature", "extra"])
def test_receipt_verification_fails_closed_for_tampering(mutation):
    private_key, original = _signed()
    envelope = copy.deepcopy(original)
    if mutation == "schema":
        envelope["schema_version"] = 2
    elif mutation == "digest":
        envelope["payload_digest"] = "sha256:" + "0" * 64
    elif mutation == "signature":
        envelope["signing"]["signature"] = "base64:AAAA"
    else:
        envelope["unexpected"] = True

    with pytest.raises(receipts.ReceiptVerificationError):
        receipts.verify_receipt(
            envelope,
            trusted_keys={"lane:test:v1": private_key.public_key()},
            expected={"job_id": "j_1"},
        )


def test_unknown_and_revoked_receipt_keys_fail_closed():
    private_key, envelope = _signed()
    with pytest.raises(receipts.ReceiptVerificationError, match="unknown"):
        receipts.verify_receipt(envelope, trusted_keys={}, expected={})
    with pytest.raises(receipts.ReceiptVerificationError, match="revoked"):
        receipts.verify_receipt(
            envelope,
            trusted_keys={"lane:test:v1": private_key.public_key()},
            expected={},
            revoked_key_ids={"lane:test:v1"},
        )


def test_receipt_payload_must_already_be_canonical():
    private_key = Ed25519PrivateKey.generate()
    payload = {**_payload(), "label": "é"}
    envelope = receipts.sign_receipt(
        payload,
        receipt_id="r_1",
        key_id="lane:test:v1",
        private_key=private_key,
    )
    envelope["payload"]["label"] = "e\u0301"
    with pytest.raises(receipts.ReceiptVerificationError, match="canonical"):
        receipts.verify_receipt(
            envelope,
            trusted_keys={"lane:test:v1": private_key.public_key()},
            expected={},
        )


def test_private_key_requires_protected_posix_modes(tmp_path):
    auth = tmp_path / "auth"
    auth.mkdir(mode=0o700)
    key_path = auth / "receipt-signing-key.pem"
    key_path.write_bytes(receipts.private_key_pem(Ed25519PrivateKey.generate()))
    key_path.chmod(0o644)
    with pytest.raises(receipts.SigningKeyProtectionError):
        receipts.load_private_key(key_path)


def test_private_key_loads_from_protected_posix_path(tmp_path):
    auth = tmp_path / "auth"
    auth.mkdir(mode=0o700)
    key_path = auth / "receipt-signing-key.pem"
    original = Ed25519PrivateKey.generate()
    key_path.write_bytes(receipts.private_key_pem(original))
    key_path.chmod(0o600)

    loaded = receipts.load_private_key(key_path)

    assert receipts.private_key_pem(loaded) == receipts.private_key_pem(original)
