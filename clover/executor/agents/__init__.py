"""Task-specific NodeAgent implementations."""

from clover.executor.agents.base import BaseNodeAgent, FastPathDecision
from clover.executor.agents.document_reasoning import DocumentReasoningNodeAgent
from clover.executor.agents.registry import (
    NODE_AGENT_REGISTRY_BY_KIND,
    build_node_agent,
    node_agent_class_for_task,
)
from clover.executor.agents.table_reasoning import TableReasoningNodeAgent

__all__ = [
    "BaseNodeAgent",
    "DocumentReasoningNodeAgent",
    "FastPathDecision",
    "NODE_AGENT_REGISTRY_BY_KIND",
    "TableReasoningNodeAgent",
    "build_node_agent",
    "node_agent_class_for_task",
]
