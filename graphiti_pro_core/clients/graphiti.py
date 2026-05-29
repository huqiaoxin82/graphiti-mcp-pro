"""
graphiti_pro_core/clients/graphiti.py — Ladybug-adapted version for zlnewma
=========================================================================
Differences from upstream:

  • ``GraphitiClient.initialize()`` branches on ``graphiti_config.graph_backend``:
    - ``ladybug``  → instantiate ``LadybugDriver`` + ``Graphiti(graph_driver=driver, ...)``
    - ``neo4j`` → original Neo4j path (unchanged)

  • ``GraphitiClient.cleanup()`` calls ``driver.close()`` generically (works for both)

Apply this file over ``graphiti_pro_core/clients/graphiti.py`` in upstream.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphiti_core import Graphiti

from utils import logger


class GraphitiClient:
    """Graphiti client management — supports Ladybug and Neo4j backends."""

    @staticmethod
    async def initialize() -> "Graphiti":
        try:
            from config import GraphitiCompatConfig, config_manager
            from config.models import GraphBackend

            await config_manager.refresh_cache()
            graphiti_config = await GraphitiCompatConfig.acquire()

            # --- LLM client ---
            from .llm import create_llm_client

            llm_client = create_llm_client(graphiti_config.llm, graphiti_config.small_llm)
            if not llm_client:
                raise ValueError("LLM_BASE_URL and LLM_API_KEY must be set")

            # --- Embedder ---
            from .embedder import create_embedder_client

            embedder_client = create_embedder_client(graphiti_config.embedder)
            if embedder_client is None:
                logger.error("Embedder client is None — embedding will be skipped")

            # --- Cross-encoder / reranker ---
            from .reranker import create_reranker_client

            cross_encoder_client = create_reranker_client(graphiti_config.small_llm)
            if cross_encoder_client is None:
                logger.error("Cross encoder client is None — ranking will be skipped")

            # --- Graph driver + Graphiti init ---
            from graphiti_core import Graphiti

            if graphiti_config.graph_backend == GraphBackend.LADYBUG:
                # ── Ladybug embedded driver ──────────────────────────────────
                import sys
                import ladybug
                sys.modules['kuzu'] = ladybug
                from graphiti_core.driver.kuzu_driver import KuzuDriver
                
                # Monkey patch KuzuDriver.execute_query to bypass FTS missing issue & fix missing parameters bug
                import re
                import logging
                logger_kuzu = logging.getLogger(__name__)

                original_execute_query = KuzuDriver.execute_query
                async def patched_execute_query(self, cypher_query_: str, **kwargs):
                    if 'QUERY_FTS_INDEX' in cypher_query_:
                        return [], None, None
                    
                    params_in_query = set(re.findall(r'\$(\w+)', cypher_query_))
                    for param in params_in_query:
                        if param not in kwargs:
                            kwargs[param] = None

                    try:
                        results = await self.client.execute(cypher_query_, parameters=kwargs)
                    except Exception as e:
                        logger_kuzu.error(f'Error executing Kuzu query: {e}\n{cypher_query_}\n{kwargs}')
                        raise

                    if not results:
                        return [], None, None

                    if isinstance(results, list):
                        dict_results = [list(result.rows_as_dict()) for result in results]
                    else:
                        dict_results = list(results.rows_as_dict())
                    return dict_results, None, None

                KuzuDriver.execute_query = patched_execute_query
                # Monkey patch missing _database attribute on KuzuDriver which causes crash in graphiti_core
                KuzuDriver._database = property(lambda self: None)

                ladybug_config = graphiti_config.ladybug
                if not ladybug_config:
                    raise ValueError("Ladybug configuration is required when graph_backend=ladybug")

                # KuzuDriver expects 'db', so we pass ladybug's path
                driver = KuzuDriver(db=ladybug_config.database_path)

                logger.info("Initializing Graphiti with LadybugDB (via KuzuDriver) backend")
                graphiti_client: Graphiti = Graphiti(
                    graph_driver=driver,
                    llm_client=llm_client,
                    embedder=embedder_client,
                    cross_encoder=cross_encoder_client,
                    max_coroutines=graphiti_config.semaphore_limit,
                )

            else:
                # ── Neo4j driver (original path) ──────────────────────────
                if not (
                    graphiti_config.neo4j
                    and graphiti_config.neo4j.uri
                    and graphiti_config.neo4j.user
                    and graphiti_config.neo4j.password
                ):
                    raise ValueError(
                        "graph_backend=neo4j but NEO4J_URI/USER/PASSWORD missing"
                    )
                graphiti_client = Graphiti(
                    uri=graphiti_config.neo4j.uri,
                    user=graphiti_config.neo4j.user,
                    password=graphiti_config.neo4j.password,
                    llm_client=llm_client,
                    embedder=embedder_client,
                    cross_encoder=cross_encoder_client,
                    max_coroutines=graphiti_config.semaphore_limit,
                )
                logger.info(f"Using Neo4j driver at: {graphiti_config.neo4j.uri}")

            # --- Store in module state ---
            from .__state__ import set_graphiti_client

            set_graphiti_client(graphiti_client)

            # --- Build indices ---
            await graphiti_client.build_indices_and_constraints()
            logger.info("✅ Graphiti client initialized successfully")

            logger.info(f"LLM model: {graphiti_config.llm.model}")
            logger.info(f"Embedding model: {graphiti_config.embedder.model}")
            logger.info(f"Concurrency limit: {graphiti_config.semaphore_limit}")

            return graphiti_client

        except Exception as e:
            logger.error(f"❌ Failed to initialize Graphiti: {e}")
            raise

    @staticmethod
    async def cleanup() -> None:
        from .__state__ import get_graphiti_client, set_graphiti_client

        graphiti_client = get_graphiti_client()
        if graphiti_client is not None:
            try:
                # Works for both Ladybug and Neo4j drivers
                if hasattr(graphiti_client, "driver") and graphiti_client.driver:
                    driver = graphiti_client.driver
                    if hasattr(driver, "close"):
                        await driver.close()
                logger.info("✅ Graphiti client cleaned up")
            except Exception as e:
                logger.error(f"❌ Error cleaning up Graphiti client: {e}")
            finally:
                set_graphiti_client(None)


# Backward-compat aliases
async def initialize_graphiti_client() -> "Graphiti":
    return await GraphitiClient.initialize()


async def cleanup_graphiti_client() -> None:
    return await GraphitiClient.cleanup()
