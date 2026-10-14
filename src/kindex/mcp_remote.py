"""Kindex MCP over streamable HTTP for remote agents behind Cloudflare Access.

The server listens on loopback only; a Cloudflare Tunnel is the way in.
Cloudflare Access authenticates every caller (Managed OAuth for a person's
agents, a service token for headless ones), and this server verifies the
Access JWT on every request as well, so a local process or a misrouted tunnel
cannot skip the edge. Only an explicit allowlist of graph tools is served:
nothing that runs a command, reads a host path or ingests from the machine.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import functools
import inspect
import ipaddress
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from . import mcp_lite, mcp_server

log = logging.getLogger("kindex.remote")

# Opt-in, like kindex-lite. Reminders (shell actions), ingestion, modes,
# maintenance, merges, Kinbase (a local executable) and task_execute stay off.
REMOTE_TOOLS = mcp_lite.LOCAL_TOOLS + (
    "ask", "suggest", "graph_stats", "watch_add", "watch_list", "watch_resolve",
)


class AccessVerifier:
    """Checks a Cloudflare Access application token (``Cf-Access-Jwt-Assertion``)."""

    def __init__(self, team_domain: str, audience: str, *, jwks_client=None):
        import jwt

        self.issuer = f"https://{team_domain}"
        self.audience = audience
        self.jwks = jwks_client or jwt.PyJWKClient(
            f"{self.issuer}/cdn-cgi/access/certs", cache_keys=True, lifespan=3600)

    def verify(self, token: str) -> dict:
        import jwt

        key = self.jwks.get_signing_key_from_jwt(token).key
        return jwt.decode(token, key, algorithms=["RS256"], audience=self.audience,
                          issuer=self.issuer, leeway=30,
                          options={"require": ["exp", "iat", "aud", "iss"]})


class AccessMiddleware:
    """ASGI guard: no valid Access JWT, no MCP. Only lifespan events pass unchecked."""

    def __init__(self, app, verifier: AccessVerifier):
        self.app = app
        self.verifier = verifier

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            return await self.app(scope, receive, send)
        if scope["type"] != "http":
            # Nothing but HTTP is served; refuse websockets before they reach a route.
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            return None
        import anyio
        import jwt

        token = dict(scope.get("headers") or ()).get(b"cf-access-jwt-assertion")
        if not token:
            return await _deny(send, 401, "missing Cloudflare Access assertion")
        try:
            # Fetching signing keys is blocking I/O; keep it off the event loop.
            claims = await anyio.to_thread.run_sync(self.verifier.verify, token.decode("ascii"))
        except (jwt.PyJWTError, UnicodeDecodeError) as error:
            log.warning("refused Access assertion: %s", type(error).__name__)
            return await _deny(send, 403, "invalid Cloudflare Access assertion")
        log.info("request %s %s by %s", scope.get("method"), scope.get("path"),
                 claims.get("email") or claims.get("common_name") or claims.get("sub"))
        return await self.app(scope, receive, send)


async def _deny(send, status: int, message: str) -> None:
    body = json.dumps({"error": message}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


# One worker: tool handlers share a single SQLite connection, which must stay on
# the thread that opened it, and serial calls keep writes ordered. The event
# loop stays free, so Access checks and other requests are never stalled by a
# slow LLM call or a busy database.
_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kindex-remote")


def _off_loop(handler):
    @functools.wraps(handler)
    async def run(*args, **kwargs):
        call = functools.partial(contextvars.copy_context().run, handler, *args, **kwargs)
        return await asyncio.get_running_loop().run_in_executor(_WORKER, call)
    return run


def _guard(name: str, handler):
    """Remote limits on the few allowlisted tools that can touch host paths."""
    if name in ("task_add", "task_list"):
        @functools.wraps(handler)
        def tasks(*args, **kwargs):
            values = inspect.signature(handler).bind(*args, **kwargs).arguments
            if values.get("project_path"):
                return "Error: remote task tools do not accept project_path; it names this host's paths"
            return handler(*args, **kwargs)
        return tasks
    if name == "add":
        @functools.wraps(handler)
        def add(*args, **kwargs):
            values = inspect.signature(handler).bind(*args, **kwargs).arguments
            referent = values.get("referent") or ""
            scope = values.get("referent_scope") or ("url" if "://" in referent else "file")
            if referent and scope != "url":
                return "Error: remote add accepts only URL referents; file referents name this host's paths"
            return handler(*args, **kwargs)
        return add
    if name == "show":
        # Not wrapped: the remote schema has no resolve_sources, because
        # resolving saved sources opens other databases on this host.
        def show(node_id: str) -> str:
            """Show full details of a node including edges and provenance.

            Saved source references are listed but not resolved remotely.

            Args:
                node_id: Node ID or title to look up.
            """
            return handler(node_id, resolve_sources=False)
        return show
    return handler


def create_server(*, public_host: str, port: int):
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings

    server = FastMCP(
        "kindex",
        instructions=(
            "Kindex knowledge graph, served remotely. Search before adding; use "
            "tag_start/tag_resume and tag_update to frame sessions. Tools that run "
            "commands, read host paths or ingest from the host are not available here."
        ),
        host="127.0.0.1", port=port, stateless_http=True, json_response=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[public_host, f"127.0.0.1:{port}", f"localhost:{port}"],
            allowed_origins=[f"https://{public_host}"]),
    )
    for name in REMOTE_TOOLS:
        # Keep the full server's guard (errors, redaction, health) on each tool.
        server.add_tool(_off_loop(_guard(name, getattr(mcp_server, name))), name=name)
    return server


def create_app(*, public_host: str, port: int, verifier: AccessVerifier):
    return AccessMiddleware(create_server(public_host=public_host, port=port).streamable_http_app(),
                            verifier)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the Kindex MCP over HTTP on loopback, behind Cloudflare Access")
    parser.add_argument("--access-team-domain", required=True,
                        help="Zero Trust team domain, e.g. example.cloudflareaccess.com")
    parser.add_argument("--access-aud", required=True, help="Access application audience (AUD) tag")
    parser.add_argument("--public-host", required=True, help="Public hostname, e.g. app.example.com")
    parser.add_argument("--host", default="127.0.0.1", help="Loopback address to listen on")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--profile", help="Kindex profile to serve (default: normal resolution)")
    args = parser.parse_args()
    try:
        loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        parser.error("--host must be a loopback IP address; publish through the tunnel")
    if args.profile:
        os.environ["KIN_PROFILE"] = args.profile
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    import uvicorn

    app = create_app(public_host=args.public_host, port=args.port,
                     verifier=AccessVerifier(args.access_team_domain, args.access_aud))
    uvicorn.run(app, host=args.host, port=args.port, proxy_headers=False, log_level="warning")


if __name__ == "__main__":
    main()
