"""Contract tests for the VMX KeySafe parser.

Two properties are pinned here:

1. Structure. Escaped (percent-encoded) commas, slashes and parentheses inside
   identifiers and values must be treated as data, never as list/pair/phrase
   structure. The expected parse trees in this file are written out by hand; the
   test never asks the parser what to expect.
2. Failure classification. Corrupt syntax, invalid fields, wrong passphrase and
   a failing MAC must surface as four distinct error types rather than one
   generic "bad password" failure.

The cryptographic fixtures live in
``tests/_data/descriptor/vmx/keysafe_contract_fixtures.json`` and are produced by
``tests/_tools/descriptor/generate_keysafe_fixtures.py``, an independent script
that does not import the parser. All key material is generated for tests only;
no real virtual machine credentials are read.
"""

from __future__ import annotations

import base64
import json

import pytest

from dissect.hypervisor.descriptor.vmx import (
    HAS_PYCRYPTODOME,
    HAS_PYSTANDALONE,
    VMX,
    InvalidMacError,
    InvalidPassphraseError,
    KeySafe,
    KeySafeFieldError,
    KeySafeSyntaxError,
    Pair,
    Phrase,
    _parse_key_locator,
    _split_list,
)
from tests._util import absolute_path

CRYPTO_AVAILABLE = HAS_PYCRYPTODOME or HAS_PYSTANDALONE
needs_crypto = pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="No crypto module available")

FIXTURES = json.loads(absolute_path("_data/descriptor/vmx/keysafe_contract_fixtures.json").read_text())


def crypto_dict(**fields: str) -> str:
    """Build an (already percent-encoded) phrase crypto dict from explicit fields."""
    defaults = {"pass2key": "PBKDF2-HMAC-SHA-1", "cipher": "AES-256", "rounds": "1000", "salt": "AA=="}
    defaults.update(fields)
    return ":".join(f"{key}={value}" for key, value in defaults.items())


def phrase(phrase_id: str, **fields: object) -> str:
    return f"phrase/{phrase_id}/{crypto_dict(**fields)}"


# A valid, minimal pair. Decodes to 54 bytes: IV(16) + one AES block(16) + HMAC-SHA-1(20) + spare(2).
VALID_DATA = "A" * 72
PAIR = f"pair/({phrase('p')},HMAC-SHA-1,{VALID_DATA})"


def tree(node: object) -> object:
    """Hand-comparable structural snapshot of a parsed locator tree."""
    if isinstance(node, list):
        return ["list", [tree(member) for member in node]]
    if isinstance(node, Pair):
        return ["pair", tree(node.wrapped_key), node.mac, base64.b64encode(node.data).decode()]
    if isinstance(node, Phrase):
        return ["phrase", node.id, node.pass2key, node.cipher, node.rounds, base64.b64encode(node.salt).decode()]
    return node


# ---------------------------------------------------------------------------
# Structure: escaped separators are data, not structure
# ---------------------------------------------------------------------------


def test_split_list_only_splits_unescaped_commas() -> None:
    # The percent-encoded comma stays inside the single member; only the raw
    # comma separates the two members.
    assert _split_list("(a%2Cb,c)") == ["a%2Cb", "c"]
    assert _split_list("(one,(nested,deep),three)") == ["one", "(nested,deep)", "three"]


def test_escaped_separators_in_phrase_id_are_data() -> None:
    locator = _parse_key_locator(f"pair/(phrase/a%2Cb%2Fc%28d%29e/{crypto_dict()},HMAC-SHA-1,{VALID_DATA})")

    assert isinstance(locator, Pair)
    # Every escaped separator is decoded back into the identifier verbatim.
    assert locator.wrapped_key.id == "a,b/c(d)e"
    assert tree(locator) == [
        "pair",
        ["phrase", "a,b/c(d)e", "PBKDF2-HMAC-SHA-1", "AES-256", 1000, "AA=="],
        "HMAC-SHA-1",
        VALID_DATA,
    ]


def test_fixture_phrase_id_keeps_escaped_separators() -> None:
    safe = KeySafe.from_text(FIXTURES["keysafe_single"])
    assert len(safe.locators) == 1
    parsed_id = safe.locators[0].wrapped_key.id
    assert parsed_id == FIXTURES["phrase_id"] == "id,with/slash(and)paren"


def test_multilevel_nested_list_parse_tree() -> None:
    nested = f"list/(list/({PAIR}),{PAIR})"
    expected = [
        "list",
        [
            ["list", [tree(_parse_key_locator(PAIR))]],
            tree(_parse_key_locator(PAIR)),
        ],
    ]
    assert tree(_parse_key_locator(nested)) == expected


def test_keysafe_rejects_nested_or_non_pair_locators() -> None:
    # A KeySafe must be a flat list of pairs; a nested list is not a pair.
    with pytest.raises(KeySafeSyntaxError, match="not a list of pairs"):
        KeySafe.from_text(f"vmware:key/list/(list/({PAIR}))")
    # A bare phrase/pair locator is not wrapped in a list, so it is rejected too.
    with pytest.raises(KeySafeSyntaxError, match="not a list of pairs"):
        KeySafe.from_text(f"vmware:key/{phrase('p')}")


# ---------------------------------------------------------------------------
# Structure: minimal ambiguous inputs and the strings that must be rejected
# ---------------------------------------------------------------------------

SYNTAX_CASES = [
    pytest.param("list/(,x)", KeySafeSyntaxError, "empty list member", id="empty-member-leading"),
    pytest.param("list/(x,)", KeySafeSyntaxError, "empty list member", id="empty-member-trailing"),
    pytest.param("list/(x,,y)", KeySafeSyntaxError, "empty list member", id="empty-member-middle"),
    pytest.param("list/(x))junk", KeySafeSyntaxError, "trailing fragment", id="trailing-after-close"),
    pytest.param("list/abc", KeySafeSyntaxError, "missing opening brace", id="no-braces"),
    pytest.param("list/((x)", KeySafeSyntaxError, "unbalanced opening brace", id="unbalanced-open"),
    pytest.param("list/(x))", KeySafeSyntaxError, "trailing fragment", id="unbalanced-close"),
    pytest.param("pair/(x,y)", KeySafeSyntaxError, "exactly 3 members", id="pair-arity-2"),
    pytest.param("pair/(x,y,z,w)", KeySafeSyntaxError, "exactly 3 members", id="pair-arity-4"),
    pytest.param(
        f"pair/(list/({phrase('p')}),HMAC-SHA-1,{VALID_DATA})",
        KeySafeSyntaxError,
        "must wrap",
        id="pair-non-phrase",
    ),
    pytest.param("phrase//" + crypto_dict(), KeySafeSyntaxError, "non-empty identifier", id="empty-phrase-id"),
    pytest.param(
        f"pair/({phrase('p')},,{VALID_DATA})", KeySafeSyntaxError, "empty list member", id="pair-empty-mac-slot"
    ),
]


@pytest.mark.parametrize("locator,error,message", SYNTAX_CASES)
def test_malformed_syntax_is_rejected(locator: str, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        _parse_key_locator(locator)


def test_empty_list_parses_as_empty_but_keysafe_rejects_it() -> None:
    # At the locator level an empty list is structurally valid...
    assert _parse_key_locator("list/()") == []
    # ...but a KeySafe needs at least one pair.
    with pytest.raises(KeySafeSyntaxError, match="not a list of pairs"):
        KeySafe.from_text("vmware:key/list/()")


def test_keysafe_identifier_and_trailing_fragment_rejected() -> None:
    with pytest.raises(KeySafeSyntaxError, match="wrong identifier"):
        KeySafe.from_text("vmware:nope/list/(" + PAIR + ")")
    with pytest.raises(KeySafeSyntaxError, match="wrong identifier"):
        KeySafe.from_text("list/(" + PAIR + ")")
    with pytest.raises(KeySafeSyntaxError, match="trailing fragment"):
        KeySafe.from_text("vmware:key/" + PAIR + ")JUNK")
    with pytest.raises(KeySafeSyntaxError, match="not a list of pairs"):
        KeySafe.from_text("vmware:key/list/()")


# ---------------------------------------------------------------------------
# Field validation: structurally fine but illegal field values
# ---------------------------------------------------------------------------

FIELD_CASES = [
    pytest.param(phrase("p", pass2key="NOPE"), "pass2key", id="unknown-pass2key"),
    pytest.param(phrase("p", cipher="AES-512"), "cipher", id="unknown-cipher"),
    pytest.param(phrase("p", rounds="NaN"), "invalid field", id="non-integer-rounds"),
    pytest.param(phrase("p", rounds="0"), "positive integer", id="zero-rounds"),
    pytest.param(phrase("p", rounds="-5"), "positive integer", id="negative-rounds"),
    pytest.param(phrase("p", salt="AAA"), "invalid field", id="non-base64-salt"),
]


@pytest.mark.parametrize("locator_string,message", FIELD_CASES)
def test_illegal_phrase_fields_are_field_errors(locator_string: str, message: str) -> None:
    with pytest.raises(KeySafeFieldError, match=message):
        _parse_key_locator(locator_string)


def test_illegal_pair_fields_are_field_errors() -> None:
    with pytest.raises(KeySafeFieldError, match="HMAC"):
        _parse_key_locator(f"pair/({phrase('p')},HMAC-MD5,{VALID_DATA})")
    with pytest.raises(KeySafeFieldError, match="base64"):
        _parse_key_locator(f"pair/({phrase('p')},HMAC-SHA-1,not base64!)")
    with pytest.raises(KeySafeFieldError, match="too short"):
        # Valid base64, but only 16 bytes: no room for a ciphertext block and HMAC.
        _parse_key_locator(f"pair/({phrase('p')},HMAC-SHA-1,AAAAAAAAAAAAAAAAAAAAAA==)")


def test_syntax_and_field_errors_are_distinct() -> None:
    assert issubclass(KeySafeSyntaxError, ValueError)
    assert issubclass(KeySafeFieldError, ValueError)
    assert not issubclass(KeySafeSyntaxError, KeySafeFieldError)


# ---------------------------------------------------------------------------
# Sealing: correct passphrase, wrong passphrase and MAC mismatch are distinct
# ---------------------------------------------------------------------------


@needs_crypto
def test_unseal_reaches_independently_specified_key() -> None:
    safe = KeySafe.from_text(FIXTURES["keysafe_single"])
    key, mac = safe.unseal_with_phrase(FIXTURES["passphrase"])

    assert base64.b64encode(key).decode() == FIXTURES["expected_key_b64"]
    assert mac == "HMAC-SHA-1"
    assert safe.locators[0].wrapped_key.rounds == 1000
    assert base64.b64encode(safe.locators[0].wrapped_key.salt).decode() == FIXTURES["salt_b64"]


@needs_crypto
def test_unseal_picks_matching_pair_in_multi_locator_safe() -> None:
    safe = KeySafe.from_text(FIXTURES["keysafe_multi"])
    assert [locator.wrapped_key.id for locator in safe.locators] == [
        FIXTURES["phrase_id"],
        FIXTURES["second_id"],
    ]

    first_key, _ = safe.unseal_with_phrase(FIXTURES["passphrase"])
    second_key, _ = safe.unseal_with_phrase(FIXTURES["second_passphrase"])

    assert base64.b64encode(first_key).decode() == FIXTURES["expected_key_b64"]
    assert base64.b64encode(second_key).decode() == FIXTURES["second_expected_key_b64"]


@needs_crypto
def test_wrong_passphrase_is_passphrase_error_not_mac_error() -> None:
    safe = KeySafe.from_text(FIXTURES["keysafe_single"])
    with pytest.raises(InvalidPassphraseError):
        safe.unseal_with_phrase(FIXTURES["wrong_passphrase"])


@needs_crypto
def test_tampered_payload_is_mac_error_with_correct_passphrase() -> None:
    safe = KeySafe.from_text(FIXTURES["keysafe_tampered_mac"])
    # The correct passphrase derives the correct wrapping key, so reaching the
    # HMAC check and failing there isolates the mismatch from a bad passphrase.
    with pytest.raises(InvalidMacError):
        safe.locators[0].unlock_with_phrase(FIXTURES["passphrase"])


@needs_crypto
def test_failure_classification_is_four_way() -> None:
    for error in (KeySafeSyntaxError, KeySafeFieldError, InvalidPassphraseError, InvalidMacError):
        assert issubclass(error, ValueError)
    assert InvalidPassphraseError is not InvalidMacError
    assert not issubclass(InvalidMacError, InvalidPassphraseError)
    assert not issubclass(InvalidPassphraseError, InvalidMacError)


@needs_crypto
def test_vmx_end_to_end_unseal_uses_test_only_material() -> None:
    vmx = VMX(
        {
            "encryption.keysafe": FIXTURES["keysafe_single"],
            "encryption.data": FIXTURES["encryption_data_b64"],
        }
    )
    vmx.unlock_with_phrase(FIXTURES["passphrase"])

    assert vmx.attr["displayname"] == FIXTURES["expected_vmx_dictionary"]["displayname"]
    assert vmx.attr["datafilekey"] == FIXTURES["expected_vmx_dictionary"]["datafilekey"]
