"""
Core agent contracts used by the HayekMAS engine.

These are the stable adapter-facing interfaces: actions, agents, and
agent status. Adapters should subclass these contracts; the core engine
only depends on the abstractions defined here.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional


class BaseAction(ABC):
    """Abstract action interface used by environments.

    Subclasses represent one environment step emitted by an agent.
    """

    def __repr__(self) -> str:
        """Return a compact debug representation for the action."""
        return f"{self.__class__.__name__}()"


class AgentStatus(Enum):
    """Lifecycle status for an agent in the Hayek economy."""

    NOVICE = "novice"
    VETERAN = "veteran"
    TYCOON = "tycoon"


class BaseAgent(ABC):
    """Abstract base class for all Hayek agents.

    Every agent carries two prompt layers:
    - ``FROZEN_SYSTEM_PROMPT`` (class-level): immutable identity and hard
      constraints that are never mutated.
    - ``TRAINABLE_SYSTEM_PROMPT`` (class-level default): strategy text that the
      Hayek mechanism is free to evolve via mutation.  Copied to the
      instance attribute ``trainable_system_prompt`` at init time.

    Use ``get_system_prompt()`` to obtain the combined prompt for LLM calls.
    Only access ``trainable_system_prompt`` directly when mutating/evolving.

    Args:
        name: Human-readable agent name.
        initial_bid: Optional starting bid before the auction loop assigns one.
        initial_wealth: Starting wealth before the agent enters the economy.
    """

    _id_counter = 0
    AGENT_TAGS: tuple[str, ...] = ()
    ROLE: str = ""
    FROZEN_SYSTEM_PROMPT: str = ""
    TRAINABLE_SYSTEM_PROMPT: str = ""

    def __init__(
        self,
        name: str = "",
        initial_bid: Optional[float] = None,
        initial_wealth: float = 0.0,
    ):
        BaseAgent._id_counter += 1
        self.id = BaseAgent._id_counter
        self.name = name or f"Agent-{self.id}"

        self.frozen_system_prompt: str = self.FROZEN_SYSTEM_PROMPT
        self.trainable_system_prompt: str = self.TRAINABLE_SYSTEM_PROMPT
        self.wealth: float = initial_wealth
        self.capability_score: float = 0.0
        self._bid: Optional[float] = initial_bid
        self._status: AgentStatus = AgentStatus.NOVICE
        self.agent_tags: tuple[str, ...] = tuple(getattr(self.__class__, "AGENT_TAGS", ()))

        # for statistics: lineage tracking
        self.root_ancestor_class: str = self.__class__.__name__
        self.father_agent_id: int = self.id
        self.father_agent_name: str = self.name
        self.parent_agent_id: Optional[int] = None
        self.parent_agent_name: Optional[str] = None
        self.spawn_method: str = "initial"
        self.tasks_lived: int = 0  # for statistics
        self.bankruptcy_episode: Optional[int] = None  # for statistics
        self._death_trace: str = ""  # recorded at bankruptcy time
        self.recent_failure_trace: str = ""
        self.recent_failure_task: str = ""
        self.recent_failure_answer: str = ""

    def __repr__(self) -> str:
        """Return a compact debug representation for the agent."""
        return f"{self.__class__.__name__}(id={self.id}, name={self.name!r}, wealth={self.wealth:.2f})"

    @property
    def role(self) -> str:
        """Return the agent's role used for population indexing and family
        membership checks. Defaults to the class-level ``ROLE`` constant; falls
        back to the name prefix (before the first ``-``) when ``ROLE`` is
        unset, so legacy unnamed agents still produce a stable key.
        """
        if self.ROLE:
            return self.ROLE
        return self.name.split("-")[0].lower() if "-" in self.name else self.name.lower()

    def get_system_prompt(self) -> str:
        """Return the combined system prompt (frozen + trainable)."""
        if self.frozen_system_prompt:
            return self.frozen_system_prompt + "\n\n" + self.trainable_system_prompt
        return self.trainable_system_prompt

    def gain_money(self, amount: float):
        """Increase the agent's wealth.

        Args:
            amount: Wealth increment to add.
        """
        self.wealth += amount

    def gain_capability(self, amount: float):
        """Increase the agent's capability score.

        Args:
            amount: Capability increment to add.
        """
        self.capability_score += amount

    def lose_money(self, amount: float):
        """Decrease the agent's wealth.

        Args:
            amount: Wealth decrement to subtract.
        """
        self.wealth -= amount

    def initialize(self, initial_wealth: float = 0.0):
        """Reset the agent to its initial state for entering the economy.

        Args:
            initial_wealth: Starting wealth to assign.
        """
        self.wealth = initial_wealth
        self.capability_score = 0.0
        self._status = AgentStatus.NOVICE
        # Start with a zero bid so eval runs before any training can still auction.
        self._bid = 0.0

    def get_bid(self) -> Optional[float]:
        """Return the agent's current bid value."""
        return self._bid

    def set_bid(self, value: Optional[float]):
        """Set the agent's bid value.

        Args:
            value: New bid value, or None if not yet assigned.
        """
        self._bid = value

    def get_status(self) -> AgentStatus:
        """Return the agent's current lifecycle status."""
        return self._status

    def set_status(self, value: AgentStatus):
        """Set the agent's lifecycle status.

        Args:
            value: New status (NOVICE, VETERAN, or TYCOON).
        """
        self._status = value

    def check_bankruptcy(self) -> bool:
        """Return whether the agent is bankrupt.

        Returns:
            `True` when the agent's wealth is negative.
        """
        return self.wealth < 0

    def record_death_trace(self, action_history: dict):
        """Record the episode trajectory at bankruptcy time.

        Builds a trace from the full action history with this agent's
        turns highlighted using >>> markers.

        Args:
            action_history: The environment's action_history dict
                mapping step numbers to action data dicts with
                'author' and 'text' keys.
        """
        parts = []
        for step, action_data in sorted(action_history.items()):
            author = action_data.get("author", "")
            text = (action_data.get("text") or "").strip()
            if not text:
                continue
            if author == self.name:
                parts.append(f">>> [Step {step}] [{author}] {text}")
            else:
                parts.append(f"    [Step {step}] [{author}] {text}")
        self._death_trace = "\n\n".join(parts)

    def get_trace_recorded_at_death(self) -> str:
        """Return the trajectory recorded at bankruptcy time.

        Returns:
            The full episode trajectory with this agent's turns
            highlighted with >>> markers. Empty if not yet recorded.
        """
        return self._death_trace

    def record_recent_failure(
        self,
        *,
        trace: str = "",
        task_description: str = "",
        correct_answer: str = "",
    ) -> None:
        """Store the latest failed-task context for future bad-agent births."""
        self.recent_failure_trace = trace
        self.recent_failure_task = task_description
        self.recent_failure_answer = correct_answer

    def has_any_tag(self, tags: set[str]) -> bool:
        """Return whether the agent has any tag in `tags`.

        Args:
            tags: Candidate tags to test.

        Returns:
            `True` when any tag overlaps with the agent's tags.
        """
        return bool(tags.intersection(self.agent_tags))

    def snapshot(self) -> dict:
        """Capture mutable auction state for later restoration.

        Returns:
            A dictionary containing the mutable auction fields for the agent.
        """
        return {
            "wealth": self.wealth,
            "capability_score": self.capability_score,
            "bid": self._bid,
            "status": self._status,
        }

    def restore(self, snapshot: dict):
        """Restore mutable auction state from a saved snapshot.

        Args:
            snapshot: A dictionary previously returned by `snapshot()`.
        """
        self.wealth = snapshot["wealth"]
        self.capability_score = snapshot["capability_score"]
        self._bid = snapshot["bid"]
        self._status = snapshot["status"]

    @abstractmethod
    def match_wakeup_condition(self, env) -> bool:
        """Return whether this agent should wake up for the current environment.

        Args:
            env: The current environment instance.

        Returns:
            `True` when the agent should be eligible to bid this step.
        """
        raise NotImplementedError

    @abstractmethod
    def act(self, env) -> BaseAction:
        """Produce the next action for the current environment.

        Args:
            env: The current environment instance.

        Returns:
            A `BaseAction` subclass representing the agent's next step.
        """
        raise NotImplementedError


def set_agent_id_counter(value: int):
    """Set the global agent ID counter to a specific value."""
    BaseAgent._id_counter = value
