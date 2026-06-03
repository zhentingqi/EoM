"""
Researchworld adapter agent layer.

Defines :class:`ResearchAction` and the five specialized agent classes for
long-form scientific reasoning. The agents share a blackboard (the env's
state string) and take one turn at a time. There are no external tools —
every turn is pure LLM reasoning.

Each agent class defines:
- ``FROZEN_SYSTEM_PROMPT`` — identity, hard constraints (never mutated).
- ``TRAINABLE_SYSTEM_PROMPT`` — strategy / heuristics (evolved by Hayek).

The roles are designed to cover the pipeline for a rubric-scored science
problem:

``literature`` → background/definitions/theorems (no new derivation).
``planner``    → outline of sub-parts, ordering, which technique per part.
``deriver``    → the heavy lifter: one concrete derivation or calculation.
``verifier``   → sanity-checks the latest contribution (sign, units, limits).
``answer``     → the only agent allowed to emit ``<final_answer>``. Synthesises
                 the final long-form solution covering every sub-question.
"""

from __future__ import annotations

import random
import re
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from hayekmas.base.agent import AgentStatus, BaseAction, BaseAgent
from hayekmas.base.prompts import (
    clean_trainable_prompt_output,
    format_mutate_agent_prompt,
    format_spawn_from_bankruptcy_prompt,
)
from hayekmas.adapters.researchworld.prompts import (
    ANSWER_PROMPT_POSTFIX,
    format_next_step_prompt,
    format_wakeup_prompt,
)

if TYPE_CHECKING:
    from hayekmas.adapters.researchworld.env import ResearchEnv


# ═══════════════════════════════════════════════════════════════════════════
# ACTION
# ═══════════════════════════════════════════════════════════════════════════

_STEP_RE = re.compile(r"<step>(.*?)</step>", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(
    r"<final_answer>(.*?)</final_answer>", re.DOTALL | re.IGNORECASE
)


class ResearchAction(BaseAction):
    """One free-form reasoning turn in the researchworld domain.

    Expected model output:
    - ``<step>...</step>`` — required body; what the agent contributes this turn.
    - ``<final_answer>...</final_answer>`` — optional; only the Answer agent is
      authorised to emit this, and doing so terminates the episode.

    If neither tag is present, the entire raw text is treated as a step body.
    """

    def __init__(self, text: str, author: str, role: str = ""):
        self.text = text or ""
        self.author = author
        self.role = role

        step_match = _STEP_RE.search(self.text)
        final_match = _FINAL_ANSWER_RE.search(self.text)

        self.step_text: str = step_match.group(1).strip() if step_match else self.text.strip()
        self.final_answer_text: Optional[str] = (
            final_match.group(1).strip() if final_match else None
        )
        self.is_final: bool = self.final_answer_text is not None

    def __repr__(self) -> str:
        tag = "ANSWER" if self.is_final else "STEP"
        preview = self.step_text.strip().replace("\n", " ")[:80]
        return f"ResearchAction({tag}, '{preview}')"


# ═══════════════════════════════════════════════════════════════════════════
# BASE AGENT
# ═══════════════════════════════════════════════════════════════════════════

class ResearchAgent(BaseAgent):
    """Base agent for long-form scientific reasoning."""

    PROMPT_POSTFIX: str = ""

    def __init__(
        self,
        backbone_llm: Callable[[str], str],
        name: str = "",
        initial_bid: Optional[float] = None,
        initial_wealth: float = 0.0,
        logger=None,
    ):
        if not name:
            name = f"{self.__class__.__name__}-{self.TRAINABLE_SYSTEM_PROMPT[:20]}..."
        super().__init__(name=name, initial_bid=initial_bid, initial_wealth=initial_wealth)
        self.backbone_llm = backbone_llm
        self.logger = logger

    def _log(self, msg: str, **kwargs):
        if self.logger:
            self.logger.log(msg, **kwargs)

    # ─────────────────────────────────────────────────────────────────────
    # Wakeup
    # ─────────────────────────────────────────────────────────────────────

    def match_wakeup_condition(self, env: "ResearchEnv") -> bool:
        last_role = env.get_last_role()
        # By default: don't repeat the same role twice in a row.
        if last_role == self.role:
            self._log(
                f"🔔 Wakeup judge [{self.name}]: not accepted (same role: {self.role})",
                indent=2,
            )
            return False

        self._log(f"🔔 Wakeup judge [{self.name}]: calling LLM...", indent=2)
        prompt = format_wakeup_prompt(
            agent_system_prompt=self.get_system_prompt(),
            state=getattr(env, "state", ""),
        )
        response = self.backbone_llm(prompt).strip().lower()
        if "boxed{yes}" in response:
            self._log(f"🔔 Wakeup judge [{self.name}]: YES", indent=2)
            return True
        self._log(f"🔔 Wakeup judge [{self.name}]: NO", indent=2)
        return False

    # ─────────────────────────────────────────────────────────────────────
    # Act
    # ─────────────────────────────────────────────────────────────────────

    def act(self, env: "ResearchEnv") -> ResearchAction:
        use_judge = getattr(env, "use_judge", True)
        terminal_mode = getattr(env, "terminal_mode", False)
        allow_abstain_terminal_mode = getattr(env, "allow_abstain_terminal_mode", True)
        step_count = getattr(env, "step_count", 0)
        max_steps = getattr(env, "max_steps", 10)
        prompt_postfix = getattr(self, "prompt_postfix", "") or self.PROMPT_POSTFIX

        prompt = format_next_step_prompt(
            state=env.state,
            agent_system_prompt=self.get_system_prompt(),
            step_count=step_count,
            max_steps=max_steps,
            use_judge=use_judge,
            terminal_mode=terminal_mode,
            allow_abstain_terminal_mode=allow_abstain_terminal_mode,
            prompt_postfix=prompt_postfix,
        )
        response = self.backbone_llm(prompt)
        action = ResearchAction(text=response, author=self.name, role=self.role)
        if terminal_mode and self.role == "answer" and not action.is_final and action.step_text:
            response = (
                f"{response.rstrip()}\n\n"
                f"<final_answer>\n{action.step_text.strip()}\n</final_answer>"
            )
            action = ResearchAction(text=response, author=self.name, role=self.role)
        return action

    # ─────────────────────────────────────────────────────────────────────
    # Spawning (good / bad births)
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def birth_good_agent(
        cls,
        parent: "ResearchAgent",
        backbone_llm: Callable[[str], str],
        initial_wealth: float = 0.0,
    ) -> "ResearchAgent":
        meta_prompt = format_mutate_agent_prompt(
            parent.frozen_system_prompt,
            parent.trainable_system_prompt,
        )
        trainable_system_prompt = clean_trainable_prompt_output(
            backbone_llm(meta_prompt),
            frozen_system_prompt=parent.frozen_system_prompt,
        )
        if not trainable_system_prompt or len(trainable_system_prompt) < 10:
            trainable_system_prompt = parent.trainable_system_prompt
        agent_id = random.randint(1000, 9999)
        parent_name = parent.name
        base_name = parent_name.split("-")[0] if "-" in parent_name else parent_name
        name = f"{base_name}-{agent_id}"
        parent_class = parent.__class__
        agent = parent_class(
            backbone_llm=backbone_llm,
            name=name,
            initial_wealth=initial_wealth,
            logger=parent.logger,
        )
        agent.frozen_system_prompt = parent.frozen_system_prompt
        agent.trainable_system_prompt = trainable_system_prompt
        return agent

    @classmethod
    def birth_bad_agent(
        cls,
        source_agent: "ResearchAgent",
        backbone_llm: Callable[[str], str],
        initial_wealth: float = 0.0,
        task_description: str = "",
        correct_answer: str = "",
        failure_trace: str = "",
    ) -> "ResearchAgent":
        meta_prompt = format_spawn_from_bankruptcy_prompt(
            source_agent.frozen_system_prompt,
            source_agent.trainable_system_prompt,
            agent_trace=(
                failure_trace
                or source_agent.recent_failure_trace
                or source_agent.get_trace_recorded_at_death()
            ),
            task_description=task_description or source_agent.recent_failure_task,
            correct_answer=correct_answer or source_agent.recent_failure_answer,
        )
        trainable_system_prompt = clean_trainable_prompt_output(
            backbone_llm(meta_prompt),
            frozen_system_prompt=source_agent.frozen_system_prompt,
        )
        if not trainable_system_prompt or len(trainable_system_prompt) < 10:
            trainable_system_prompt = source_agent.trainable_system_prompt
        agent_id = random.randint(1000, 9999)
        base_name = (
            source_agent.name.split("-")[0] if "-" in source_agent.name else source_agent.name
        )
        name = f"{base_name}-{agent_id}"
        source_class = source_agent.__class__
        agent = source_class(
            backbone_llm=backbone_llm,
            name=name,
            initial_wealth=initial_wealth,
            logger=source_agent.logger,
        )
        agent.frozen_system_prompt = source_agent.frozen_system_prompt
        agent.trainable_system_prompt = trainable_system_prompt
        return agent

    # ─────────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────────

    def serialize(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "wealth": self.wealth,
            "capability_score": getattr(self, "capability_score", 0.0),
            "bid": self.get_bid(),
            "status": self.get_status().value,
            "type": self.__class__.__name__,
            "frozen_system_prompt": self.frozen_system_prompt,
            "trainable_system_prompt": self.trainable_system_prompt,
            "root_ancestor_class": getattr(self, "root_ancestor_class", self.__class__.__name__),
            "father_agent_id": getattr(self, "father_agent_id", self.id),
            "father_agent_name": getattr(self, "father_agent_name", self.name),
            "parent_agent_id": getattr(self, "parent_agent_id", None),
            "parent_agent_name": getattr(self, "parent_agent_name", None),
            "spawn_method": getattr(self, "spawn_method", "initial"),
            "tasks_lived": getattr(self, "tasks_lived", 0),
            "bankruptcy_episode": getattr(self, "bankruptcy_episode", None),
            "recent_failure_trace": getattr(self, "recent_failure_trace", ""),
            "recent_failure_task": getattr(self, "recent_failure_task", ""),
            "recent_failure_answer": getattr(self, "recent_failure_answer", ""),
        }

    @staticmethod
    def deserialize(
        data: Dict[str, Any],
        backbone_llm: Optional[Callable[[str], str]] = None,
        logger=None,
    ) -> "ResearchAgent":
        if backbone_llm is None:
            def placeholder_llm(_prompt: str) -> str:
                raise RuntimeError(
                    "ResearchAgent.backbone_llm not set. Inject LLM after loading."
                )
            backbone_llm = placeholder_llm
        agent_type = data.get("type", "ResearchAgent")
        agent_class = _get_agent_class_by_name(agent_type)
        agent = agent_class(
            backbone_llm=backbone_llm,
            name=data.get("name", "ResearchAgent"),
            initial_bid=data.get("bid"),
            initial_wealth=data.get("wealth", 0.0),
            logger=logger,
        )
        if data.get("frozen_system_prompt"):
            agent.frozen_system_prompt = data["frozen_system_prompt"]
        if data.get("trainable_system_prompt"):
            agent.trainable_system_prompt = data["trainable_system_prompt"]
        if "id" in data:
            agent.id = data["id"]
        if "wealth" in data:
            agent.wealth = data["wealth"]
        agent.capability_score = data.get("capability_score", 0.0)
        if "status" in data:
            status_str = data["status"]
            if isinstance(status_str, str):
                agent.set_status(AgentStatus(status_str))
        agent.root_ancestor_class = data.get(
            "root_ancestor_class", agent.__class__.__name__
        )
        agent.father_agent_id = data.get("father_agent_id", agent.id)
        agent.father_agent_name = data.get("father_agent_name", agent.name)
        agent.parent_agent_id = data.get("parent_agent_id")
        agent.parent_agent_name = data.get("parent_agent_name")
        agent.spawn_method = data.get("spawn_method", "initial")
        agent.tasks_lived = data.get("tasks_lived", 0)
        agent.bankruptcy_episode = data.get("bankruptcy_episode")
        agent.recent_failure_trace = data.get("recent_failure_trace", "")
        agent.recent_failure_task = data.get("recent_failure_task", "")
        agent.recent_failure_answer = data.get("recent_failure_answer", "")
        return agent

    def __repr__(self) -> str:
        bid_str = f"${self.get_bid():.2f}" if self.get_bid() is not None else "None"
        return f"{self.__class__.__name__}({self.name}, bid={bid_str}, wealth=${self.wealth:.2f})"


# ═══════════════════════════════════════════════════════════════════════════
# SPECIALIZED AGENTS
# ═══════════════════════════════════════════════════════════════════════════
#
# FROZEN: role identity + hard constraint (never mutated).
# TRAINABLE: strategy/heuristic (evolves via Hayek mutation).
# Every FROZEN prompt forbids emitting <final_answer> except for the Answer
# role, which is the only terminal-tagged class.
# ═══════════════════════════════════════════════════════════════════════════


class LiteratureResearchAgent(ResearchAgent):
    """Domain-knowledge specialist: definitions, theorems, standard formulas."""

    ROLE = "literature"
    AGENT_TAGS = ("research",)

    FROZEN_SYSTEM_PROMPT = (
        "You are the **Literature** agent — a theoretical-background specialist. "
        "Your job is to surface the relevant definitions, standard formulas, "
        "and textbook results that the derivations below will rely on. You must "
        "NOT perform new calculations and you must NOT emit <final_answer>."
    )
    TRAINABLE_SYSTEM_PROMPT = (
        "Read the problem and identify the sub-field (quantum mechanics, "
        "statistical mechanics, condensed matter, etc.). List the key concepts, "
        "definitions, and canonical formulas that will be needed across the "
        "sub-questions, along with the exact expressions (e.g. the quantum "
        "Fisher information formula, the general form of a weak value, etc.).\n"
        "- Keep it compact: name each concept and write its defining equation.\n"
        "- Do NOT solve any sub-question here — just lay the groundwork.\n"
        "- If the state already covers all needed background, add only one "
        "missing piece rather than re-stating what is present."
    )

    # The Literature agent is allowed to self-start an episode.
    SELF_STARTING_ROLES = frozenset({"literature", "planner"})

    def __init__(self, *args, name: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs, name=name if name is not None else "Literature")

    def match_wakeup_condition(self, env: "ResearchEnv") -> bool:
        if not env.get_last_author():
            self._log(f"🔔 Wakeup [{self.name}]: auto-start → YES", indent=2)
            return True
        return super().match_wakeup_condition(env)


class PlannerResearchAgent(ResearchAgent):
    """Outline-builder: enumerates sub-parts and the tactic for each."""

    ROLE = "planner"
    AGENT_TAGS = ("research",)

    FROZEN_SYSTEM_PROMPT = (
        "You are the **Planner** agent. You produce a concise roadmap for "
        "solving the problem — enumerating each sub-question (a), (b), (c), ... "
        "and the tactic to use for each. You must NOT perform derivations or "
        "compute numerical results here, and you must NOT emit <final_answer>."
    )
    TRAINABLE_SYSTEM_PROMPT = (
        "Scan the problem for labelled sub-parts and derivation targets "
        "(e.g. 'compute I(g)', 'find C such that ...'). For each sub-part, name "
        "the technique in one phrase (e.g. 'second-order expansion of the "
        "Kraus operator', 'diagonalise H in the spin-singlet basis').\n"
        "- Output a short numbered outline; no derivations.\n"
        "- Prioritise sub-parts that are prerequisites for later parts.\n"
        "- If a plan already exists in the state, refine the ordering or add "
        "one missing sub-step rather than rewriting the whole plan."
    )

    SELF_STARTING_ROLES = frozenset({"literature", "planner"})

    def __init__(self, *args, name: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs, name=name if name is not None else "Planner")

    def match_wakeup_condition(self, env: "ResearchEnv") -> bool:
        # If Literature has spoken but no plan exists yet, Planner takes over.
        if not env.get_last_author():
            self._log(f"🔔 Wakeup [{self.name}]: auto-start → YES", indent=2)
            return True
        return super().match_wakeup_condition(env)


class DeriverResearchAgent(ResearchAgent):
    """Main workhorse: produces one concrete derivation per turn."""

    ROLE = "deriver"
    AGENT_TAGS = ("research",)

    FROZEN_SYSTEM_PROMPT = (
        "You are the **Deriver** agent — the main workhorse. Each turn you "
        "perform ONE substantive derivation or calculation that advances "
        "exactly one sub-question. You must NOT emit <final_answer>."
    )
    TRAINABLE_SYSTEM_PROMPT = (
        "Identify the earliest sub-question in the plan that is not yet "
        "completed, then carry out its derivation cleanly:\n"
        "- State the sub-part label (e.g. 'Part (b):') at the top of your <step>.\n"
        "- Show the intermediate algebra or physical reasoning; keep equations "
        "compact but include justification for non-trivial substitutions "
        "(limits, approximations, linearisations).\n"
        "- Track signs, unit/dimension consistency, and small-parameter regimes.\n"
        "- If a prior Deriver turn has an error flagged by a Verifier, fix "
        "only that error in this turn instead of starting a new sub-part."
    )


class VerifierResearchAgent(ResearchAgent):
    """Sanity-checker: validates the latest derivation."""

    ROLE = "verifier"
    AGENT_TAGS = ("research",)

    FROZEN_SYSTEM_PROMPT = (
        "You are the **Verifier** agent. You check the most recent "
        "contribution — especially the latest Deriver turn — for correctness. "
        "You must NOT add new derivations from scratch and must NOT emit "
        "<final_answer>."
    )
    TRAINABLE_SYSTEM_PROMPT = (
        "Inspect the latest contribution and report exactly one of:\n"
        "- **Correct.** + one line on the strongest sanity check you applied "
        "  (unit check, limit check, symmetry check, known-case reduction).\n"
        "- **Error:** <what is wrong>. **Fix:** <what the Deriver should do next>.\n"
        "Do not copy or restate the derivation, and do not grade anything "
        "earlier than the most recent substantive step."
    )


class AnswerResearchAgent(ResearchAgent):
    """The ONLY agent authorised to submit a final answer."""

    ROLE = "answer"
    AGENT_TAGS = ("terminal", "final_answer")

    FROZEN_SYSTEM_PROMPT = (
        "You are the **Answer** agent. You are the ONLY agent that may emit "
        "a <final_answer>...</final_answer> block. Synthesise all shared-state "
        "derivations into a complete, sub-question-by-sub-question solution."
    )
    TRAINABLE_SYSTEM_PROMPT = (
        "Produce a single self-contained long-form solution that addresses "
        "EVERY sub-question (a), (b), (c), ... explicitly, in order. For each "
        "part:\n"
        "- Name the sub-part being answered.\n"
        "- Include the key derivation steps and final expression/number.\n"
        "- Reuse results from the shared state; do not re-prove known textbook "
        "facts unnecessarily.\n"
        "- Be explicit about approximations, limit regimes, and assumptions.\n"
        "If the shared state is still too thin to cover some sub-part, note "
        "the gap briefly inside your <final_answer> rather than refusing to "
        "answer the rest."
    )

    PROMPT_POSTFIX = ANSWER_PROMPT_POSTFIX

    # Answer agent waits until at least one Deriver turn has happened.
    _WAKE_AFTER = {"deriver", "verifier"}
    _FALLBACK_AFTER = {"deriver"}

    def __init__(self, *args, name: Optional[str] = None, **kwargs):
        self.prompt_postfix = self.PROMPT_POSTFIX
        super().__init__(*args, **kwargs, name=name if name is not None else "Answer")

    def match_wakeup_condition(self, env: "ResearchEnv") -> bool:
        last_role = env.get_last_role()
        if not last_role:
            self._log(f"🔔 Wakeup [{self.name}]: no prior actions → NO", indent=2)
            return False
        if last_role not in self._WAKE_AFTER:
            self._log(
                f"🔔 Wakeup [{self.name}]: last role was {last_role} → NO "
                "(need deriver/verifier)",
                indent=2,
            )
            return False
        llm_decision = super().match_wakeup_condition(env)
        if llm_decision:
            return True
        if last_role in self._FALLBACK_AFTER:
            self._log(
                f"🔔 Wakeup [{self.name}]: LLM said NO, fallback after {last_role} → YES",
                indent=2,
            )
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

RESEARCH_AGENT_CLASSES = [
    LiteratureResearchAgent,
    PlannerResearchAgent,
    DeriverResearchAgent,
    VerifierResearchAgent,
    AnswerResearchAgent,
]

_AGENT_CLASS_REGISTRY: Dict[str, type] = {
    "ResearchAgent": ResearchAgent,
    "LiteratureResearchAgent": LiteratureResearchAgent,
    "PlannerResearchAgent": PlannerResearchAgent,
    "DeriverResearchAgent": DeriverResearchAgent,
    "VerifierResearchAgent": VerifierResearchAgent,
    "AnswerResearchAgent": AnswerResearchAgent,
}


def _get_agent_class_by_name(class_name: str) -> type:
    return _AGENT_CLASS_REGISTRY.get(class_name, ResearchAgent)
