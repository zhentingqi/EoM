"""
Population management for the Hayek agent economy.

This file defines the shared `Population` data structure used by the core
engine. It stores agent membership, role indexes, wakeup matching, and parent
selection for mutation.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Set, List, Optional, Callable, Any
import time

from pydantic import BaseModel, Field, PrivateAttr

from hayekmas.base.agent import BaseAgent
from hayekmas.utils.logger import logger


def _default_role_extractor(agent: BaseAgent) -> str:
    """Extract a default role name for an agent.

    Returns ``agent.role`` directly — every ``BaseAgent`` provides one (either
    from the class-level ``ROLE`` constant or derived from the name prefix).
    """
    return agent.role


class Population(BaseModel):
    """Runtime population store for agents in the Hayek economy.

    The population keeps a fast lookup by agent id and a secondary index by
    role so the engine can activate, inspect, and mutate agents efficiently.
    """

    # Role -> set of agent IDs (hashable). This is the "dictionary of sets".
    by_role: Dict[str, Set[int]] = Field(default_factory=dict)

    # Non-serialized: id -> agent instance (populated at runtime).
    _agents: Dict[int, BaseAgent] = PrivateAttr(default_factory=dict)
    # Optional: custom role extractor (agent -> role name). Not in schema.
    _role_extractor: Optional[Callable[[BaseAgent], str]] = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, role_extractor: Optional[Callable[[BaseAgent], str]] = None, **data):
        """Initialize the population container.

        Args:
            role_extractor: Optional function used to derive a role name for an
                agent.
            **data: Initial pydantic model data.
        """
        super().__init__(**data)
        self._agents = {}
        self._role_extractor = role_extractor

    def _role_for(self, agent: BaseAgent) -> str:
        """Return the role name for an agent."""
        if self._role_extractor is not None:
            return self._role_extractor(agent)
        return _default_role_extractor(agent)

    def add_agent(self, agent: BaseAgent) -> None:
        """Add an agent to the population and index it by role."""
        aid = agent.id
        self._agents[aid] = agent
        role = self._role_for(agent)
        if role not in self.by_role:
            self.by_role[role] = set()
        self.by_role[role].add(aid)

    def remove_agent(self, agent: BaseAgent) -> None:
        """Remove an agent from the population and from its role set."""
        aid = agent.id
        self._agents.pop(aid, None)
        role = self._role_for(agent)
        if role in self.by_role:
            self.by_role[role].discard(aid)
            if not self.by_role[role]:
                del self.by_role[role]

    def __len__(self) -> int:
        """Return the current number of agents in the population."""
        return len(self._agents)

    def __iter__(self):
        """Iterate over all agents (order undefined)."""
        return iter(self._agents.values())

    def get_all(self) -> List[BaseAgent]:
        """Return all agents as a list."""
        return list(self._agents.values())

    def get_by_role(self, role: str) -> List[BaseAgent]:
        """Return all agents in the given role."""
        ids = self.by_role.get(role, set())
        return [self._agents[aid] for aid in ids if aid in self._agents]

    def get_best_agents(
        self,
        n: Optional[int] = None,
        *,
        key: Optional[Callable[[BaseAgent], Any]] = None,
        role: Optional[str] = None,
    ) -> List[BaseAgent]:
        """
        Return the best agents, optionally limited to a role and/or top n.
        Default sort key is wealth (highest first). Best agents are first.
        """
        if key is None:
            key = lambda a: a.wealth
        if role is not None:
            agents = self.get_by_role(role)
        else:
            agents = self.get_all()
        sorted_agents = sorted(agents, key=key, reverse=True)
        if n is not None:
            sorted_agents = sorted_agents[:n]
        return sorted_agents

    def get_richest_agent(self) -> Optional[BaseAgent]:
        """Return the currently richest living agent."""
        agents = self.get_all()
        if not agents:
            return None
        return max(agents, key=lambda agent: agent.wealth)

    def get_poorest_agent(self) -> Optional[BaseAgent]:
        """Return the currently poorest living agent."""
        agents = self.get_all()
        if not agents:
            return None
        return min(agents, key=lambda agent: agent.wealth)

    def _match_with_retry(
        self,
        agent: BaseAgent,
        env: Any,
        *,
        retry_attempts: int,
        retry_backoff_seconds: float,
        fail_open: bool,
        log_wakeup: bool = True,
    ) -> bool:
        """Evaluate an agent wakeup condition with retry handling.

        Args:
            agent: Agent whose wakeup rule should be evaluated.
            env: Active environment passed to the wakeup check.
            retry_attempts: Additional retry attempts after the initial try.
            retry_backoff_seconds: Base exponential backoff between retries.
            fail_open: Whether to treat repeated failures as active.

        Returns:
            `True` when the agent should be considered active for the step.
        """
        total_attempts = max(1, retry_attempts + 1)
        for attempt in range(total_attempts):
            try:
                return agent.match_wakeup_condition(env)
            except Exception as exc:
                is_last_attempt = attempt == total_attempts - 1
                if is_last_attempt:
                    if log_wakeup:
                        logger.log(
                            f"⚠️  Wakeup failed for {agent.name} after {total_attempts} attempt(s): {exc}",
                            indent=2,
                        )
                    return fail_open

                delay = max(0.0, retry_backoff_seconds) * (2 ** attempt)
                if log_wakeup:
                    logger.log(
                        f"⚠️  Wakeup error for {agent.name}: {exc} | retrying in {delay:.1f}s",
                        indent=2,
                    )
                if delay > 0:
                    time.sleep(delay)

        return fail_open

    def get_active(
        self,
        env: Any,
        *,
        parallel: bool = False,
        max_workers: int = 1,
        retry_attempts: int = 0,
        retry_backoff_seconds: float = 0.0,
        fail_open: bool = False,
        log_wakeup: bool = True,
    ) -> List[BaseAgent]:
        """Return agents whose wakeup condition matches the current environment.

        Args:
            env: Active environment passed to each agent wakeup check.
            parallel: Whether to evaluate wakeup conditions concurrently.
            max_workers: Worker count for concurrent wakeup evaluation.
            retry_attempts: Additional retry attempts for wakeup failures.
            retry_backoff_seconds: Base exponential backoff between retries.
            fail_open: Whether repeated wakeup failures should activate the agent.

        Returns:
            The list of agents eligible to bid in the current step.
        """
        agents = list(self._agents.values())
        if not parallel or max_workers <= 1 or len(agents) <= 1:
            return [
                agent
                for agent in agents
                if self._match_with_retry(
                    agent,
                    env,
                    retry_attempts=retry_attempts,
                    retry_backoff_seconds=retry_backoff_seconds,
                    fail_open=fail_open,
                    log_wakeup=log_wakeup,
                )
            ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            active_flags = list(
                executor.map(
                    lambda agent: self._match_with_retry(
                        agent,
                        env,
                        retry_attempts=retry_attempts,
                        retry_backoff_seconds=retry_backoff_seconds,
                        fail_open=fail_open,
                        log_wakeup=log_wakeup,
                    ),
                    agents,
                )
            )
        return [agent for agent, is_active in zip(agents, active_flags) if is_active]

    def get_agent_ids(self) -> Set[int]:
        """Return the set of all agent IDs (e.g. for max_id)."""
        return set(self._agents.keys())
