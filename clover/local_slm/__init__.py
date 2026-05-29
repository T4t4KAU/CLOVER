"""SLM client utilities for local NodeAgents."""

from clover.local_slm.client import (
    DEFAULT_SLM_CONFIG_PATH,
    create_slm_client,
    generate_slm_text,
    load_slm_config,
)

__all__ = [
    "DEFAULT_SLM_CONFIG_PATH",
    "create_slm_client",
    "generate_slm_text",
    "load_slm_config",
]
