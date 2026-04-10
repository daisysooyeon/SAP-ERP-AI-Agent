"""
src/config.py
Central configuration loader for SAP ERP AI Agent.

Reads configs.yaml from the project root and exposes a typed AppConfig object.
Sensitive secrets (API keys, etc.) remain in .env and are accessed via os.getenv.

Usage:
    from src.config import get_config
    cfg = get_config()
    print(cfg.models.router.name)   # "qwen3:4b"
    print(cfg.paths.erp_db)         # "data/sap_erp.db"
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Typed schema (dataclasses)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str
    temperature: float = 0.0


@dataclass
class ModelsConfig:
    router:       ModelConfig = field(default_factory=lambda: ModelConfig("qwen3:4b"))
    worker_a:     ModelConfig = field(default_factory=lambda: ModelConfig("qwen3:4b"))
    worker_a_sql: ModelConfig = field(default_factory=lambda: ModelConfig("qwen2.5-coder:3b"))
    worker_b:     ModelConfig = field(default_factory=lambda: ModelConfig("qwen3:4b", 0.1))
    synthesizer:  ModelConfig = field(default_factory=lambda: ModelConfig("gpt-4o", 0.3))


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"


@dataclass
class PathsConfig:
    erp_db:          str = "data/sap_erp.db"
    checkpoint_db:   str = "data/checkpoints.db"
    chroma_db:       str = "./chroma_db"
    docs_dir:        str = "data/docs"
    eval_test_cases: str = "data/eval/router_test_cases.json"
    reports_dir:     str = "reports"


@dataclass
class LoggingConfig:
    level:         str = "DEBUG"
    console_level: str = "INFO"
    dir:           str = "logs"
    max_bytes:     int = 10 * 1024 * 1024   # 10 MB
    backup_count:  int = 5


@dataclass
class RagConfig:
    chunk_size:       int   = 512
    chunk_overlap:    int   = 64
    collection_name:  str   = "sap_manuals"
    dense_weight:     float = 0.6
    sparse_weight:    float = 0.4
    top_k_retrieval:  int   = 10
    top_k_rerank:     int   = 3


@dataclass
class FeatureFlagsConfig:
    human_in_the_loop: bool = True


@dataclass
class AppConfig:
    ollama:        OllamaConfig       = field(default_factory=OllamaConfig)
    models:        ModelsConfig       = field(default_factory=ModelsConfig)
    paths:         PathsConfig        = field(default_factory=PathsConfig)
    logging:       LoggingConfig      = field(default_factory=LoggingConfig)
    rag:           RagConfig          = field(default_factory=RagConfig)
    feature_flags: FeatureFlagsConfig = field(default_factory=FeatureFlagsConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs.yaml"


def _parse_model(raw: dict) -> ModelConfig:
    return ModelConfig(
        name=raw.get("name", "qwen3:4b"),
        temperature=float(raw.get("temperature", 0.0)),
    )


def _load_from_yaml(path: Path) -> AppConfig:
    """Parse configs.yaml and build a fully typed AppConfig."""
    with open(path, encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f) or {}

    # ── ollama ──────────────────────────────────────────────────────────────
    o = raw.get("ollama", {})
    ollama = OllamaConfig(
        base_url=o.get("base_url", "http://localhost:11434"),
    )

    # ── models ──────────────────────────────────────────────────────────────
    m = raw.get("models", {})
    models = ModelsConfig(
        router=      _parse_model(m.get("router",       {"name": "qwen3:4b"})),
        worker_a=    _parse_model(m.get("worker_a",     {"name": "qwen3:4b"})),
        worker_a_sql=_parse_model(m.get("worker_a_sql", {"name": "qwen2.5-coder:3b"})),
        worker_b=    _parse_model(m.get("worker_b",     {"name": "qwen3:4b", "temperature": 0.1})),
        synthesizer= _parse_model(m.get("synthesizer",  {"name": "gpt-4o",   "temperature": 0.3})),
    )

    # ── paths ────────────────────────────────────────────────────────────────
    p = raw.get("paths", {})
    paths = PathsConfig(
        erp_db=          p.get("erp_db",          "data/sap_erp.db"),
        checkpoint_db=   p.get("checkpoint_db",   "data/checkpoints.db"),
        chroma_db=       p.get("chroma_db",        "./chroma_db"),
        docs_dir=        p.get("docs_dir",         "data/docs"),
        eval_test_cases= p.get("eval_test_cases",  "data/eval/router_test_cases.json"),
        reports_dir=     p.get("reports_dir",      "reports"),
    )

    # ── logging ─────────────────────────────────────────────────────────────
    lg = raw.get("logging", {})
    logging = LoggingConfig(
        level=         lg.get("level",         "DEBUG"),
        console_level= lg.get("console_level", "INFO"),
        dir=           lg.get("dir",           "logs"),
        max_bytes=     int(lg.get("max_bytes",     10 * 1024 * 1024)),
        backup_count=  int(lg.get("backup_count",  5)),
    )

    # ── rag ─────────────────────────────────────────────────────────────────
    r = raw.get("rag", {})
    rag = RagConfig(
        chunk_size=      int(r.get("chunk_size",      512)),
        chunk_overlap=   int(r.get("chunk_overlap",   64)),
        collection_name= r.get("collection_name",     "sap_manuals"),
        dense_weight=    float(r.get("dense_weight",  0.6)),
        sparse_weight=   float(r.get("sparse_weight", 0.4)),
        top_k_retrieval= int(r.get("top_k_retrieval", 10)),
        top_k_rerank=    int(r.get("top_k_rerank",    3)),
    )

    # ── feature flags ────────────────────────────────────────────────────────
    ff = raw.get("feature_flags", {})
    feature_flags = FeatureFlagsConfig(
        human_in_the_loop=bool(ff.get("human_in_the_loop", True)),
    )

    return AppConfig(
        ollama=ollama,
        models=models,
        paths=paths,
        logging=logging,
        rag=rag,
        feature_flags=feature_flags,
    )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    Load and return the AppConfig singleton (cached after first call).

    The config path can be overridden via the CONFIG_PATH env variable:
        CONFIG_PATH=/custom/path/configs.yaml python -m src.main
    """
    config_path = Path(os.getenv("CONFIG_PATH", str(_CONFIG_PATH)))
    if not config_path.exists():
        raise FileNotFoundError(
            f"configs.yaml not found at {config_path}. "
            "Make sure configs.yaml exists in the project root."
        )
    return _load_from_yaml(config_path)
