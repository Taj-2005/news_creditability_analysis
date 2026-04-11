"""
Agent orchestration package (Milestone 2).

Provides state schema, graph wiring, and node stubs. The existing ML pipeline
in ``src.features``, ``src.models``, and ``src.app.core`` remains unchanged;
nodes will integrate via explicit calls in a later implementation phase.
"""

from src.agent.state import AgentState

__all__ = ["AgentState"]
