"""
graphiti_pro_core/mcp_server/resources.py — Ladybug-adapted version for zlnewma
=============================================================================
Difference from upstream:

  • ``get_status()`` no longer calls ``driver.client.verify_connectivity()``
    (Neo4j-only API).  Instead it runs a trivial ``RETURN 1`` Cypher query which
    works on both Ladybug and Neo4j drivers.

Apply this file over ``graphiti_pro_core/mcp_server/resources.py`` in upstream.
"""
from __future__ import annotations

from typing import cast

from mcp.server.fastmcp.resources import FunctionResource

from graphiti_core import Graphiti

from .types import StatusResponse
from ..clients import get_graphiti_client
from utils import logger


async def get_status() -> StatusResponse:
    """Get server status — works for both Ladybug and Neo4j backends."""
    graphiti_client = get_graphiti_client()

    if graphiti_client is None:
        return StatusResponse(status="error", message="Graphiti client not initialized")

    try:
        client = cast(Graphiti, graphiti_client)

        # ``RETURN 1`` is valid Cypher on both Ladybug and Neo4j.
        # Ladybug: uses LadybugDriver.execute_query()
        # Neo4j: uses the bolt driver — same interface
        await client.driver.execute_query("RETURN 1")

        return StatusResponse(
            status="ok",
            message="✅ Graphiti MCP server is running and graph DB is reachable",
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Graph DB health check failed: {error_msg}")
        return StatusResponse(
            status="error",
            message=f"❌ Graphiti MCP server is running but graph DB unreachable: {error_msg}",
        )


get_status_resource = FunctionResource.from_function(
    fn=get_status, uri="http://graphiti/status"
)
