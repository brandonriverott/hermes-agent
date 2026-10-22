"""Large MCP catalogs must obey Hermes' existing wire cap, not a smaller SDK default."""

import asyncio
import json

import pytest

from tools import mcp_tool
from tools.mcp_tool_errors import _MCP_HTTP_MAX_BODY_BYTES


@pytest.mark.asyncio
async def test_catalog_above_sdk_default_fits_hermes_wire_cap(monkeypatch):
    assert mcp_tool._ensure_mcp_sdk()
    httpx = mcp_tool.sdk_httpx()
    description = "x" * (1024 * 1024 + 128)

    async def handler(request):
        if request.method != "POST":
            return httpx.Response(405)
        message = json.loads(request.content)
        if "id" not in message:
            return httpx.Response(202)
        if message["method"] == "initialize":
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": message["id"], "result": {
                    "protocolVersion": message["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "catalog-fixture", "version": "1"},
                },
            })
        assert message["method"] == "tools/list"
        payload = {"jsonrpc": "2.0", "id": message["id"], "result": {"tools": [{
            "name": "fixture", "description": description,
            "inputSchema": {"type": "object", "properties": {}},
        }]}}
        body = ("event: message\ndata: " + json.dumps(payload) + "\n\n").encode()
        assert len(body) < _MCP_HTTP_MAX_BODY_BYTES
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(handler))
    server = mcp_tool.MCPServerTask("catalog-fixture")
    server._http_rejection = {}

    async def list_tools():
        async with server._streamable_http_transport(
            "http://127.0.0.1/mcp", {}, 5, True, None, None, False, set(),
        ) as streams:
            async with mcp_tool.ClientSession(*streams[:2]) as session:
                await session.initialize()
                result = await session.list_tools()
                assert result.tools[0].description == description

    await asyncio.wait_for(list_tools(), timeout=5)


def test_legacy_sse_receives_same_cap_when_supported(monkeypatch):
    assert mcp_tool._ensure_mcp_sdk()
    captured = {}

    def sse_client(*, max_sse_event_size=1024 * 1024, **kwargs):
        captured["limit"] = max_sse_event_size
        return object()

    monkeypatch.setattr(mcp_tool, "sse_client", sse_client)
    mcp_tool.MCPServerTask("fixture")._sse_transport("http://127.0.0.1/sse", {}, 5, True, None, None, False)
    assert captured["limit"] == _MCP_HTTP_MAX_BODY_BYTES
