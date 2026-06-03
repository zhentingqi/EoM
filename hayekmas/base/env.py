"""
Base environment contract used by the HayekMAS engine.

Adapters subclass `BaseEnv` and implement `initialize` and `apply`;
the engine depends only on the interface defined here.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from hayekmas.base.agent import BaseAction
from hayekmas.utils.logger import logger


class BaseEnv(ABC):
    """Abstract environment interface used by the Hayek engine.

    Provides shared state (``action_history``, ``terminated``) and common
    query methods so that the engine and adapters can rely on a stable
    contract instead of ad-hoc ``getattr`` access.
    """

    def __init__(self):
        """Initialize the base environment state."""
        self.step_count: int = 0
        self.terminated: bool = False
        self.action_history: Dict[int, Dict[str, Any]] = {}

    @abstractmethod
    def initialize(self):
        """Reset the environment to its initial state."""
        raise NotImplementedError

    def reach_termination(self) -> bool:
        """Return whether the current episode has terminated."""
        return self.terminated

    @abstractmethod
    def apply(self, action: BaseAction) -> float:
        """Apply an action to the environment.

        Args:
            action: The action to execute.

        Returns:
            The immediate reward produced by the environment.
        """
        raise NotImplementedError

    # ─────────────────────────────────────────────────────────────────────────
    # ACTION HISTORY QUERIES
    # ─────────────────────────────────────────────────────────────────────────

    def get_last_author(self) -> Optional[str]:
        """Return the author name from the most recent action, or ``None``."""
        if not self.action_history:
            return None
        last_step = max(self.action_history.keys())
        return self.action_history[last_step].get("author")

    def get_last_role(self) -> Optional[str]:
        """Return the role from the most recent action, or ``None``."""
        if not self.action_history:
            return None
        last_step = max(self.action_history.keys())
        return self.action_history[last_step].get("role")

    def get_last_message_text(self) -> Optional[str]:
        """Return the text from the most recent action, or ``None``."""
        if not self.action_history:
            return None
        last_step = max(self.action_history.keys())
        text = (self.action_history[last_step].get("text") or "").strip()
        return text or None

    def get_agent_trace(self, agent_name: str) -> str:
        """Return the concatenated trace of actions by *agent_name*.

        Matches both exact name and base-name prefix (before the first ``-``),
        so ``"Planner-1234"`` matches actions by any ``"Planner-*"`` author.
        """
        base_name = agent_name.split("-")[0] if "-" in agent_name else agent_name
        trace_parts = []
        for step, action_data in sorted(self.action_history.items()):
            author = action_data.get("author", "")
            author_base = author.split("-")[0] if "-" in author else author
            if author == agent_name or author_base == base_name:
                text = action_data.get("text", "").strip()
                trace_parts.append(f"[Step {step}] {text}")
        return "\n\n".join(trace_parts) if trace_parts else ""

    # ─────────────────────────────────────────────────────────────────────────
    # LOGGING & DESCRIPTIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _log(self, msg: str, indent: int = 0, must_print: bool = False):
        """Write a message through the shared logger.

        Args:
            msg: Message text to log.
            indent: Indentation level for the log output.
            must_print: Whether to force console output even in quiet mode.
        """
        logger.log(msg, indent, must_print)

    def get_state_description(self) -> str:
        """Return a short human-readable environment summary."""
        return f"{self.__class__.__name__}(step={self.step_count})"

    def get_terminal_score(self) -> Optional[float]:
        """Return the terminal score for the completed episode, if any."""
        return None

    def get_path_reward_score(self) -> Optional[float]:
        """Score used to distribute path_reward across the realized agents.

        Defaults to ``get_terminal_score()``.  Adapters whose terminal score
        is *cumulative across episodes* (e.g. iterative-improvement tasks
        where workspace + best-snapshot persist) should override this to
        return the per-episode *delta* — otherwise re-submitting a frozen
        best score keeps paying out path reward every episode for no work.
        For one-shot domains (a different task per episode) the default is
        correct: terminal_score IS the episode's contribution.
        """
        return self.get_terminal_score()

    def get_task_description(self) -> str:
        """Return the task/question text for this episode.

        Used by the bankruptcy analysis prompt so the LLM can compare the
        agent's trace against the actual task.  Adapters should override this.
        """
        return ""

    def get_correct_answer(self) -> str:
        """Return the ground-truth answer for this episode.

        Used by the bankruptcy analysis prompt so the LLM can diagnose what
        went wrong.  Adapters should override this.
        """
        return ""

    def build_episode_metrics(self) -> dict:
        """Return adapter-specific metrics for episode logging."""
        return {}
