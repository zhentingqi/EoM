"""
Core building blocks used by `HayekMAS`.

The base package now exposes four primary concepts:
- `contracts`: adapter-facing action / agent / environment interfaces
- `pipeline`: generic train/eval workflow scaffolding
- `mas`: the core Hayek execution engine
- `population`: the shared population data structure
"""

from hayekmas.base.agent import BaseAction, BaseAgent, AgentStatus, set_agent_id_counter
from hayekmas.base.env import BaseEnv
from hayekmas.base.pipeline import Trainer, Evaluator
from hayekmas.base.mas import HayekMAS, TerminationReason
from hayekmas.base.population import Population
