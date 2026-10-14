"""Remote Kindex MCP: Cloudflare Access assertion checks and the tool allowlist."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("mcp")
jwt = pytest.importorskip("jwt")
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402

from kindex import mcp_remote  # noqa: E402

TEAM = "team.example.cloudflareaccess.com"
AUD = "aud-tag-123"
HOST = "app.example.com"


class _Keys:
    """Stands in for PyJWKClient: one signing key, looked up by any token."""

    def __init__(self, key):
        self.key = key

    def get_signing_key_from_jwt(self, token):
        return self


@pytest.fixture(scope="module")
def keys():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token(private, **overrides):
    now = int(time.time())
    claims = {"iss": f"https://{TEAM}", "aud": [AUD], "iat": now, "exp": now + 300,
              "sub": "user-1", "email": "owner@example.com", **overrides}
    return jwt.encode(claims, private, algorithm="RS256")


def _verifier(keys):
    return mcp_remote.AccessVerifier(TEAM, AUD, jwks_client=_Keys(keys.public_key()))


def _client(keys):
    from starlette.testclient import TestClient

    app = mcp_remote.create_app(public_host=HOST, port=8767, verifier=_verifier(keys))
    return TestClient(app, base_url=f"https://{HOST}")


def _rpc(client, token, method, params=None, *, rpc_id=1):
    headers = {"accept": "application/json, text/event-stream"}
    if token is not None:
        headers["cf-access-jwt-assertion"] = token
    body = {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params or {}}
    return client.post("/mcp", json=body, headers=headers)


def test_allowlist_excludes_host_reaching_tools():
    for name in ("remind_create", "remind_exec", "ingest", "task_execute", "graph_merge",
                 "mode_import", "kinbase_submit", "stale_check", "dream"):
        assert name not in mcp_remote.REMOTE_TOOLS


def test_served_tools_are_exactly_the_allowlist(keys):
    with _client(keys) as client:
        init = {"protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"}}
        assert _rpc(client, _token(keys), "initialize", init).status_code == 200
        listed = _rpc(client, _token(keys), "tools/list", rpc_id=2)
    assert listed.status_code == 200
    names = {tool["name"] for tool in listed.json()["result"]["tools"]}
    assert names == set(mcp_remote.REMOTE_TOOLS)


@pytest.mark.parametrize("case", ["missing", "wrong_aud", "wrong_issuer", "expired", "other_key", "garbage"])
def test_requests_without_a_valid_access_assertion_are_refused(keys, case):
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = {
        "missing": None,
        "wrong_aud": _token(keys, aud=["someone-else"]),
        "wrong_issuer": _token(keys, iss="https://evil.cloudflareaccess.com"),
        "expired": _token(keys, iat=int(time.time()) - 900, exp=int(time.time()) - 600),
        "other_key": _token(other),
        "garbage": "not.a.jwt",
    }[case]
    with _client(keys) as client:
        response = _rpc(client, token, "tools/list")
    assert response.status_code == (401 if case == "missing" else 403)
    assert "result" not in response.text


def test_foreign_host_header_is_refused(keys):
    from starlette.testclient import TestClient

    app = mcp_remote.create_app(public_host=HOST, port=8767, verifier=_verifier(keys))
    with TestClient(app, base_url="https://attacker.example") as client:
        response = _rpc(client, _token(keys), "tools/list")
    assert response.status_code in (400, 403, 421)


def test_add_refuses_file_referents_but_keeps_url_referents(monkeypatch):
    calls = []

    def add(text, node_type="concept", tags="", domains="", audience="private",
            referent="", referent_digest="", referent_scope="", **rest):
        calls.append(referent)
        return "ok"

    guarded = mcp_remote._guard("add", add)
    assert guarded("x", referent="/etc/passwd").startswith("Error:")
    assert guarded("x", referent="https://example.com/a", referent_digest="ab" * 32) == "ok"
    assert guarded("x", referent="https://example.com/a", referent_scope="file").startswith("Error:")
    assert calls == ["https://example.com/a"]


def test_show_never_resolves_host_sources():
    seen = []
    guarded = mcp_remote._guard("show", lambda node_id, resolve_sources=False: seen.append(resolve_sources))
    guarded("n1", resolve_sources=True)
    assert seen == [False]


def test_cli_refuses_a_non_loopback_listener(monkeypatch):
    monkeypatch.setattr("sys.argv", ["kindex-remote", "--access-team-domain", TEAM,
                                     "--access-aud", AUD, "--public-host", HOST, "--host", "0.0.0.0"])
    with pytest.raises(SystemExit) as exc:
        mcp_remote.main()
    assert exc.value.code == 2
