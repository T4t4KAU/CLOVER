"""Runtime feature flags used by CLOVER ablation experiments."""

from __future__ import annotations

from typing import Any

ENABLE_EDGE_AGENT = "enable_edge_agent"
ENABLE_EDGE_REPAIR = "enable_edge_repair"
ENABLE_TERMINAL_EDGE_REVIEW = "enable_terminal_edge_review"
ENABLE_CONTRACT_GATE = "enable_contract_gate"
ENABLE_NODE_REVIEW = "enable_node_review"
ENABLE_CLOUD_RECOVERY = "enable_cloud_recovery"
ENABLE_CLOUD_REPLAN = "enable_cloud_replan"
ENABLE_CLOUD_SYNTHESIS = "enable_cloud_synthesis"
ENABLE_STATIC_FAST_PATH = "enable_static_fast_path"
ENABLE_STATIC_FINALIZATION = "enable_static_finalization"

RUNTIME_FEATURE_FLAGS = (
    ENABLE_EDGE_AGENT,
    ENABLE_EDGE_REPAIR,
    ENABLE_TERMINAL_EDGE_REVIEW,
    ENABLE_CONTRACT_GATE,
    ENABLE_NODE_REVIEW,
    ENABLE_CLOUD_RECOVERY,
    ENABLE_CLOUD_REPLAN,
    ENABLE_CLOUD_SYNTHESIS,
    ENABLE_STATIC_FAST_PATH,
    ENABLE_STATIC_FINALIZATION,
)

_LEGACY_FEATURE_FALLBACKS = {
    ENABLE_EDGE_REPAIR: ENABLE_EDGE_AGENT,
    ENABLE_TERMINAL_EDGE_REVIEW: ENABLE_EDGE_AGENT,
    ENABLE_CLOUD_REPLAN: ENABLE_CLOUD_RECOVERY,
    ENABLE_CLOUD_SYNTHESIS: ENABLE_CLOUD_RECOVERY,
}


def runtime_feature_enabled(
    config: dict[str, Any] | None,
    feature: str,
) -> bool:
    """Return one feature flag, defaulting to the full CLOVER behavior."""

    if feature not in RUNTIME_FEATURE_FLAGS:
        raise ValueError(f"Unknown CLOVER runtime feature: {feature}")
    if not isinstance(config, dict):
        return True
    selected_feature = feature
    if feature not in config:
        selected_feature = _LEGACY_FEATURE_FALLBACKS.get(feature, feature)
    if selected_feature not in config:
        return True
    value = config[selected_feature]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"{feature} must be a boolean, got {value!r}")


def runtime_feature_flags(config: dict[str, Any] | None) -> dict[str, bool]:
    """Return all runtime feature flags in a stable order."""

    return {
        feature: runtime_feature_enabled(config, feature)
        for feature in RUNTIME_FEATURE_FLAGS
    }
