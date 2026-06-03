"""
HayekMAS package.

The `HayekMAS` engine plus domain adapters for CloudCast, researchworld,
and accelerator design-space exploration.

Based on:
- Baum, E. "Toward a Model of Mind as a Laissez-Faire Economy of Idiots" (1996)
- Baum, E. "Toward a Model of Intelligence as an Economy of Agents" (1999)
"""

__version__ = "0.1.0"

from hayekmas.base import (
    HayekMAS,
    TerminationReason,
    BaseAction,
    BaseAgent,
    BaseEnv,
    AgentStatus,
    Trainer,
    Evaluator,
)
