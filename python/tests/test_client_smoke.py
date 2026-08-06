"""Tiny smoke test for the yq-nova Python client (no live server required)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yq_nova import Client, NovaApiError  # noqa: E402


def test_imports_and_construction():
    client = Client("http://127.0.0.1:7999")
    assert client.base_url == "http://127.0.0.1:7999"
    # trailing slash is stripped
    client2 = Client("http://127.0.0.1:7999/")
    assert client2.base_url == "http://127.0.0.1:7999"
    # empty base raises
    try:
        Client("")
        assert False, "expected ValueError for empty base_url"
    except ValueError:
        pass


def test_error_type_attributes():
    err = NovaApiError("not_found", "boom", 404, trace_id="abc")
    assert err.code == "not_found"
    assert err.message == "boom"
    assert err.status == 404
    assert err.trace_id == "abc"


def test_client_side_validation():
    client = Client("http://127.0.0.1:7999")
    # These must raise without touching the network.
    for call in (
        lambda: client.remember(""),
        lambda: client.recall(""),
        lambda: client.recall("q", top_k=0),
        lambda: client.forget(),
        lambda: client.get_memory(""),
        lambda: client.upsert_entity(""),
        lambda: client.traverse(""),
        lambda: client.extract_and_link(""),
    ):
        try:
            call()
            assert False, f"expected ValueError for {call}"
        except ValueError:
            pass


if __name__ == "__main__":
    test_imports_and_construction()
    test_error_type_attributes()
    test_client_side_validation()
    print("all smoke tests passed")