"""
config/models.py — Ladybug-adapted version for zlnewma
====================================================
Differences from upstream graphiti-mcp-pro/config/models.py:

  1. Added ``GraphBackend`` enum: ``neo4j`` | ``ladybug``
  2. Added ``LadybugConfig``: ``database_path`` field
  3. ``Neo4jConfig`` fields are now Optional (not required when backend=ladybug)
  4. ``GraphitiCompatConfig``:
     - New ``graph_backend`` field (default: ladybug for desktop use)
     - New Optional ``ladybug`` field
     - ``neo4j`` field made Optional
     - ``acquire()`` branches on ``graph_backend``

Apply this file over ``config/models.py`` in a fresh graphiti-mcp-pro clone.
"""

from __future__ import annotations

from abc import ABC
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Set, Type, TypeVar

from pydantic import BaseModel, Field

from utils import logger

T = TypeVar("T", bound="BaseConfig")


# ---------------------------------------------------------------------------
# New: graph backend selector
# ---------------------------------------------------------------------------


class GraphBackend(str, Enum):
    NEO4J = "neo4j"
    LADYBUG = "ladybug"


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BaseConfig(BaseModel, ABC):
    """Base configuration class with common functionality."""

    @classmethod
    def get_config_keys(cls) -> Set[str]:
        keys: Set[str] = set()
        for field_name, field_info in cls.model_fields.items():
            key = field_info.alias or field_name
            keys.add(key)
        return keys

    @classmethod
    async def acquire(cls: Type[T]) -> T:
        from .manager import config_manager
        from .exceptions import ConfigValidationError

        keys = list(cls.get_config_keys())
        config_data = await config_manager.get_config(keys)
        try:
            instance = cls.model_validate(config_data)
            logger.debug(f"Created {cls.__name__} instance")
            return instance
        except Exception as e:
            logger.error(f"Failed to create {cls.__name__}: {e}")
            raise ConfigValidationError(
                f"Configuration validation failed for {cls.__name__}: {e}"
            )


# ---------------------------------------------------------------------------
# Graph DB configs
# ---------------------------------------------------------------------------


class Neo4jConfig(BaseConfig):
    """Neo4j database configuration (optional — only needed when graph_backend=neo4j)."""

    uri: Optional[str] = Field(
        default=None, alias="neo4j_uri", description="Neo4j database URI"
    )
    user: Optional[str] = Field(
        default=None, alias="neo4j_user", description="Neo4j username"
    )
    password: Optional[str] = Field(
        default=None, alias="neo4j_password", description="Neo4j password"
    )


class LadybugConfig(BaseConfig):
    """Ladybug embedded graph DB configuration (used when graph_backend=ladybug)."""

    database_path: str = Field(
        alias="ladybug_database_path",
        description="Absolute path to the Ladybug .ladybug database directory",
    )


# ---------------------------------------------------------------------------
# LLM / embedder / reranker configs (unchanged from upstream)
# ---------------------------------------------------------------------------


class LLMCompatConfig(BaseConfig):
    api_key: Optional[str] = Field(default=None, alias="llm_api_key")
    base_url: str = Field(alias="llm_base_url")
    model: str = Field(alias="llm_model_name")
    temperature: float = Field(alias="llm_temperature")


class EmbedderCompatConfig(BaseConfig):
    model: str = Field(alias="embedding_model_name")
    api_key: Optional[str] = Field(default=None, alias="embedding_api_key")
    base_url: str = Field(alias="embedding_base_url")


class SmallLLMCompatConfig(BaseConfig):
    api_key: Optional[str] = Field(default=None, alias="small_llm_api_key")
    base_url: str = Field(alias="small_llm_base_url")
    model: str = Field(alias="small_llm_model_name")

    def is_same_as_llm(self, llm_config: LLMCompatConfig) -> bool:
        return (
            self.api_key == llm_config.api_key
            and self.base_url == llm_config.base_url
            and self.model == llm_config.model
        )


class MCPConfig(BaseConfig):
    enable_sync_return: bool = Field(alias="enable_sync_return")


class LogSetting(BaseConfig):
    log_save_days: int = Field(alias="log_save_days")


# ---------------------------------------------------------------------------
# Top-level config — now supports both Neo4j and Ladybug
# ---------------------------------------------------------------------------


class GraphitiCompatConfig(BaseConfig):
    """Main Graphiti configuration — supports neo4j and ladybug backends."""

    graph_backend: GraphBackend = Field(
        alias="graph_backend",
        default=GraphBackend.LADYBUG,
        description="Graph DB backend: 'ladybug' (default, embedded) or 'neo4j' (legacy)",
    )
    # Ladybug fields — required when graph_backend=ladybug
    ladybug: Optional[LadybugConfig] = None
    # Neo4j fields — required when graph_backend=neo4j
    neo4j: Optional[Neo4jConfig] = None

    llm: LLMCompatConfig
    embedder: EmbedderCompatConfig
    small_llm: SmallLLMCompatConfig
    semaphore_limit: int = Field(alias="semaphore_limit", default=10)

    @classmethod
    async def acquire(cls) -> "GraphitiCompatConfig":  # type: ignore[override]
        from .manager import config_manager
        from .exceptions import ConfigValidationError

        try:
            await config_manager.refresh_cache()

            # Common configs
            llm_config = await LLMCompatConfig.acquire()
            embedder_config = await EmbedderCompatConfig.acquire()
            small_llm_config = await SmallLLMCompatConfig.acquire()
            semaphore_cfg = await config_manager.get_config(["semaphore_limit"])
            graph_backend_cfg = await config_manager.get_config(["graph_backend"])

            graph_backend = GraphBackend(
                graph_backend_cfg.get("graph_backend", GraphBackend.LADYBUG.value)
            )

            ladybug_config: Optional[LadybugConfig] = None
            neo4j_config: Optional[Neo4jConfig] = None

            if graph_backend == GraphBackend.LADYBUG:
                ladybug_config = await LadybugConfig.acquire()
                logger.info(
                    f"Using Ladybug backend: {ladybug_config.database_path}"
                )
            else:
                neo4j_config = await Neo4jConfig.acquire()
                logger.info(
                    f"Using Neo4j backend: {neo4j_config.uri}"
                )

            instance = cls(
                graph_backend=graph_backend,
                ladybug=ladybug_config,
                neo4j=neo4j_config,
                llm=llm_config,
                embedder=embedder_config,
                small_llm=small_llm_config,
                semaphore_limit=semaphore_cfg.get("semaphore_limit", 10),
            )
            logger.info("✅ GraphitiCompatConfig created")
            return instance

        except Exception as e:
            logger.error(f"Failed to create GraphitiCompatConfig: {e}")
            from .exceptions import ConfigValidationError
            raise ConfigValidationError(
                f"GraphitiCompatConfig creation failed: {e}"
            )

    @classmethod
    def get_config_keys(cls) -> Set[str]:
        keys: Set[str] = set()
        keys.update(LLMCompatConfig.get_config_keys())
        keys.update(EmbedderCompatConfig.get_config_keys())
        keys.update(SmallLLMCompatConfig.get_config_keys())
        keys.update(Neo4jConfig.get_config_keys())
        keys.update(LadybugConfig.get_config_keys())
        keys.add("semaphore_limit")
        keys.add("graph_backend")
        return keys
