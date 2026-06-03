"""Cloudcast agents.

Six specialized roles share the same base class; each has its own
frozen/trainable prompt pair and tool allowlist.  The trainable prompt is
what the Hayek mechanism mutates via good/bad birth.
"""

from __future__ import annotations

import random
import re
from typing import Any, Callable, Dict, Optional

from hayekmas.adapters.cloudcast.prompts import (
    format_next_step_prompt,
    format_tool_signatures,
    format_wakeup_prompt,
)
from hayekmas.base.agent import AgentStatus, BaseAction, BaseAgent
from hayekmas.base.prompts import (
    clean_trainable_prompt_output,
    format_mutate_agent_prompt,
    format_spawn_from_bankruptcy_prompt,
)


class CloudcastAction(BaseAction):
    """<thought>+<code> action, mirroring FinanceAction."""

    def __init__(self, text: str, author: str, role: str = ""):
        self.text = text or ""
        self.author = author
        self.role = role
        self.thought, self.code = self._parse_thought_code(self.text)
        self.is_final = False  # terminal state is driven by final_answer() in code

    @staticmethod
    def _parse_thought_code(text: str) -> tuple[Optional[str], Optional[str]]:
        thought_match = re.search(r"<thought>(.*?)</thought>", text, re.DOTALL)
        code_match = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        code = code_match.group(1).strip() if code_match else None
        return thought, code

    @property
    def is_code_action(self) -> bool:
        return self.code is not None

    def __repr__(self) -> str:
        kind = "CODE" if self.is_code_action else "STEP"
        return f"CloudcastAction({kind}, {self.text[:60]!r}...)"


class CloudcastAgent(BaseAgent):
    """Base class for all cloudcast agents.

    Subclasses override ``ROLE`` / ``AGENT_TAGS`` / ``ALLOWED_TOOLS`` /
    ``FROZEN_SYSTEM_PROMPT`` / ``TRAINABLE_SYSTEM_PROMPT`` / ``SELF_STARTING``.
    """

    ALLOWED_TOOLS: Optional[set[str]] = None
    # Roles flagged SELF_STARTING skip the LLM wakeup on the very first step
    # (when no agent has acted yet) so the episode can always get off the
    # ground without an extra judge call.
    SELF_STARTING: bool = False
    # Per-role shell timeout override (seconds).  ``None`` → use the env's
    # default.  Set on Builder because full ``zig build`` runs can take
    # several minutes.
    SHELL_TIMEOUT: Optional[int] = None

    def __init__(
        self,
        backbone_llm: Callable[[str], str],
        name: str = "",
        initial_bid: Optional[float] = None,
        initial_wealth: float = 0.0,
        logger=None,
    ):
        if not name:
            name = f"{self.__class__.__name__}-{random.randint(1000, 9999)}"
        super().__init__(name=name, initial_bid=initial_bid, initial_wealth=initial_wealth)
        self.backbone_llm = backbone_llm
        self.logger = logger

    def _log(self, msg: str, **kwargs):
        if self.logger:
            self.logger.log(msg, **kwargs)

    def _emit_wakeup(self, env, woke_up: bool) -> bool:
        """Record wakeup result to env for research telemetry, then return it."""
        if hasattr(env, "record_wakeup"):
            env.record_wakeup(self.name, self.role, woke_up)
        return woke_up

    def match_wakeup_condition(self, env) -> bool:
        """LLM-judged wakeup. Self-starting roles auto-wake when state empty."""
        last_author = env.get_last_author() if hasattr(env, "get_last_author") else ""
        log_wakeup = getattr(env, "log_wakeup", True)

        if not last_author:
            should_wake = self.SELF_STARTING
            if log_wakeup:
                self._log(
                    f"🔔 Wakeup [{self.name}]: auto-start → {'YES' if should_wake else 'NO'}",
                    indent=3,
                )
            return self._emit_wakeup(env, should_wake)

        if log_wakeup:
            self._log(f"🔔 Wakeup [{self.name}]: asking LLM...", indent=3)
        prompt = format_wakeup_prompt(
            agent_system_prompt=self.get_system_prompt(),
            state=getattr(env, "state", ""),
        )
        response = (self.backbone_llm(prompt) or "").strip().lower()
        woke = "boxed{yes}" in response
        if log_wakeup:
            self._log(f"🔔 Wakeup [{self.name}]: {'YES' if woke else 'NO'}", indent=3)
        return self._emit_wakeup(env, woke)

    def act(self, env) -> CloudcastAction:
        env.allowed_tool_names = self.ALLOWED_TOOLS
        # Per-role shell timeout override (picked up by env._make_tool_functions).
        if hasattr(env, "_current_shell_timeout"):
            env._current_shell_timeout = self.SHELL_TIMEOUT
        tool_signatures = format_tool_signatures(self.ALLOWED_TOOLS)
        prompt = format_next_step_prompt(
            agent_system_prompt=self.get_system_prompt(),
            instruction=getattr(env, "instruction", ""),
            state=getattr(env, "state", ""),
            tool_signatures=tool_signatures,
            step_count=getattr(env, "step_count", 0),
            max_steps=getattr(env, "max_steps", 10),
        )
        response = self.backbone_llm(prompt)
        return CloudcastAction(text=response, author=self.name, role=self.role)

    # ─────────────────────────────────────────────────────────────────────
    # SPAWNING
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def birth_good_agent(
        cls,
        parent: "CloudcastAgent",
        backbone_llm: Callable[[str], str],
        initial_wealth: float = 0.0,
    ) -> "CloudcastAgent":
        meta_prompt = format_mutate_agent_prompt(
            parent.frozen_system_prompt,
            parent.trainable_system_prompt,
        )
        trainable = clean_trainable_prompt_output(
            backbone_llm(meta_prompt),
            frozen_system_prompt=parent.frozen_system_prompt,
        )
        if not trainable or len(trainable) < 10:
            trainable = parent.trainable_system_prompt
        base_name = parent.name.split("-")[0] if "-" in parent.name else parent.name
        agent = parent.__class__(
            backbone_llm=backbone_llm,
            name=f"{base_name}-{random.randint(1000, 9999)}",
            initial_wealth=initial_wealth,
            logger=parent.logger,
        )
        agent.frozen_system_prompt = parent.frozen_system_prompt
        agent.trainable_system_prompt = trainable
        # Stash parent prompt so the snapshot can show prompt_diff_vs_parent
        # — useful for spotting cases where the LLM mutator returned an
        # empty / too-short string and we silently fell back to the parent
        # prompt verbatim (mutation failed).
        agent._parent_trainable_prompt = parent.trainable_system_prompt
        return agent

    @classmethod
    def birth_bad_agent(
        cls,
        source_agent: "CloudcastAgent",
        backbone_llm: Callable[[str], str],
        initial_wealth: float = 0.0,
        task_description: str = "",
        correct_answer: str = "",
        failure_trace: str = "",
    ) -> "CloudcastAgent":
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
        trainable = clean_trainable_prompt_output(
            backbone_llm(meta_prompt),
            frozen_system_prompt=source_agent.frozen_system_prompt,
        )
        if not trainable or len(trainable) < 10:
            trainable = source_agent.trainable_system_prompt
        base_name = (
            source_agent.name.split("-")[0]
            if "-" in source_agent.name
            else source_agent.name
        )
        agent = source_agent.__class__(
            backbone_llm=backbone_llm,
            name=f"{base_name}-{random.randint(1000, 9999)}",
            initial_wealth=initial_wealth,
            logger=source_agent.logger,
        )
        agent.frozen_system_prompt = source_agent.frozen_system_prompt
        agent.trainable_system_prompt = trainable
        agent._parent_trainable_prompt = source_agent.trainable_system_prompt
        return agent

    # ─── serialization ───

    def _compute_prompt_diff_vs_parent(self) -> Optional[str]:
        """Unified diff of trainable_system_prompt against the parent at
        birth time. ``None`` for initial agents (no parent). Empty string
        when the mutator returned a verbatim copy (i.e. mutation no-op)."""
        parent_tp = getattr(self, "_parent_trainable_prompt", None)
        if parent_tp is None:
            return None
        if parent_tp == self.trainable_system_prompt:
            return ""  # explicit "no-op" signal
        import difflib
        diff_lines = list(difflib.unified_diff(
            parent_tp.splitlines(),
            self.trainable_system_prompt.splitlines(),
            fromfile="parent_prompt",
            tofile="this_prompt",
            n=1,
            lineterm="",
        ))
        return "\n".join(diff_lines)

    def serialize(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "wealth": self.wealth,
            "capability_score": self.capability_score,
            "bid": self.get_bid(),
            "status": self.get_status().value,
            "type": self.__class__.__name__,
            "frozen_system_prompt": self.frozen_system_prompt,
            "trainable_system_prompt": self.trainable_system_prompt,
            "parent_trainable_prompt": getattr(self, "_parent_trainable_prompt", None),
            "prompt_diff_vs_parent": self._compute_prompt_diff_vs_parent(),
            "root_ancestor_class": self.root_ancestor_class,
            "father_agent_id": self.father_agent_id,
            "father_agent_name": self.father_agent_name,
            "parent_agent_id": self.parent_agent_id,
            "parent_agent_name": self.parent_agent_name,
            "spawn_method": self.spawn_method,
            "tasks_lived": self.tasks_lived,
            "bankruptcy_episode": self.bankruptcy_episode,
            "recent_failure_trace": self.recent_failure_trace,
            "recent_failure_task": self.recent_failure_task,
            "recent_failure_answer": self.recent_failure_answer,
        }

    @staticmethod
    def deserialize(
        data: Dict[str, Any],
        backbone_llm: Optional[Callable[[str], str]] = None,
        logger=None,
    ) -> "CloudcastAgent":
        if backbone_llm is None:
            def placeholder_llm(_prompt: str) -> str:
                raise RuntimeError(
                    "CloudcastAgent.backbone_llm not set. Inject LLM after loading."
                )
            backbone_llm = placeholder_llm
        cls_name = data.get("type", "CoderAgent")
        cls = _AGENT_CLASS_REGISTRY.get(cls_name, CoderAgent)
        agent = cls(
            backbone_llm=backbone_llm,
            name=data.get("name", cls_name),
            initial_bid=data.get("bid"),
            initial_wealth=data.get("wealth", 0.0),
            logger=logger,
        )
        agent.frozen_system_prompt = data["frozen_system_prompt"]
        agent.trainable_system_prompt = data["trainable_system_prompt"]
        if "id" in data:
            agent.id = data["id"]
        agent.capability_score = data.get("capability_score", 0.0)
        status_str = data.get("status")
        if isinstance(status_str, str):
            agent.set_status(AgentStatus(status_str))
        return agent


# ═══════════════════════════════════════════════════════════════════════════
# SPECIALIZED ROLES
# ═══════════════════════════════════════════════════════════════════════════


class PlannerAgent(CloudcastAgent):
    """Reads the instruction and lays out an incremental plan (no tools)."""

    ROLE = "planner"
    AGENT_TAGS = ("planning",)
    ALLOWED_TOOLS = frozenset()
    SELF_STARTING = True
    FROZEN_SYSTEM_PROMPT = (
        "You are the Planner. You do not touch the workspace — you outline the "
        "next concrete sub-goal for the team (Reader / Implementer / Builder / "
        "Evaluator / Finalizer). Emit ONLY a <thought> block — no <code>."
    )
    TRAINABLE_SYSTEM_PROMPT = (
        "Always wake up at the start of an episode (step 1). Also wake whenever "
        "the state shows new information from Reader / Builder / Evaluator that "
        "the prior plan did not anticipate, or whenever no concrete edit has "
        "happened since the last plan. Produce a focused, single-sentence next "
        "sub-goal and name which role should act next. Do not explain the "
        "whole task — one atomic step at a time."
    )


class ReaderAgent(CloudcastAgent):
    """Explores the existing code / workspace with read-only tools."""

    ROLE = "reader"
    AGENT_TAGS = ("research",)
    ALLOWED_TOOLS = {"read_file", "shell"}
    SELF_STARTING = True
    FROZEN_SYSTEM_PROMPT = (
        "You are the Reader. You understand the existing codebase by reading "
        "files and running read-only shell commands (ls, grep, find, cat). "
        "You never write files, never run builds, never call the verifier. "
        "Restrict shell to inspection commands."
    )
    TRAINABLE_SYSTEM_PROMPT = (
        "Wake up when the plan calls for understanding existing code or "
        "unfamiliar files AND that file has not yet been summarized in the "
        "state. Do ONE targeted read per turn (one file, one grep, one ls). "
        "Do NOT re-read a file you have already read this episode — go to "
        "sleep and let Implementer act instead. Summarize the finding in "
        "your <thought> so downstream agents can use it without re-reading."
    )


class ImplementerAgent(CloudcastAgent):
    """Writes / edits source files in the workspace."""

    ROLE = "implementer"
    AGENT_TAGS = ("research",)
    ALLOWED_TOOLS = {"write_file", "read_file"}
    FROZEN_SYSTEM_PROMPT = (
        "You are the Implementer. You write and edit files in the workspace. "
        "You do not run builds or the verifier — that is the Builder's and "
        "Evaluator's job. You NEVER use open(), exec(), or import third-party "
        "libraries inside <code>; the sandbox blocks those. Use read_file / "
        "write_file exclusively. Third-party imports (networkx, pandas, typing, "
        "pathlib, …) belong INSIDE the program file you write — they will run "
        "fine when the verifier executes that file."
    )
    TRAINABLE_SYSTEM_PROMPT = (
        "Wake up whenever the team has at least an abstract plan or a Reader "
        "summary of the target file, OR whenever the last eval signaled a "
        "regression. To edit a file, ALWAYS read_file(path) first to see the "
        "current content, then write_file(path, full_new_content) with the "
        "ENTIRE new file body — write_file replaces the whole file, it does "
        "not append. Prefer one focused change per turn (e.g. swap the "
        "search_algorithm body, leave helpers alone). Do not try to test the "
        "program in <code> — leave that to Evaluator's request_eval()."
    )


class BuilderAgent(CloudcastAgent):
    """Runs the build / tests the implementation via shell."""

    ROLE = "builder"
    AGENT_TAGS = ("research",)
    ALLOWED_TOOLS = {"shell", "read_file"}
    # `zig build` on the full git rewrite can run several minutes; override
    # the default shell timeout so Builder's commands do not get killed.
    SHELL_TIMEOUT = 900
    FROZEN_SYSTEM_PROMPT = (
        "You are the Builder. You run the build / tests / lint / dry-run of "
        "the implementation via shell commands (e.g. `zig build`, `python "
        "-m pytest`, `python initial_program.py`, invoking a produced "
        "binary). You never write source files and never call the verifier."
    )
    TRAINABLE_SYSTEM_PROMPT = (
        "Wake up after Implementer changes or when the plan asks to verify "
        "compilation. Run ONE build or test command per turn and report "
        "whether it passed or failed, with key stderr snippets, so the "
        "Implementer can fix issues next."
    )


class EvaluatorAgent(CloudcastAgent):
    """Calls the expensive verifier to book in progress."""

    ROLE = "evaluator"
    AGENT_TAGS = ("research", "terminal")
    ALLOWED_TOOLS = {"request_eval", "read_file"}
    FROZEN_SYSTEM_PROMPT = (
        "You are the Evaluator. Your only paid action is calling "
        "`request_eval()`, which runs the expensive verifier and returns a "
        "score. The delta vs. the previous score becomes your reward — so "
        "only eval when you expect a meaningful improvement."
    )
    TRAINABLE_SYSTEM_PROMPT = (
        "Wake up whenever Implementer has written new code since the last "
        "eval (or since the start of the episode if no eval has fired yet). "
        "If the task uses pure-Python code (no separate build step), you can "
        "eval directly without waiting for a Builder green. Do not eval "
        "twice in a row without new code in between, and do not eval on a "
        "build/import failure."
    )


class FinalizerAgent(CloudcastAgent):
    """Submits the final answer, ending the episode and triggering a final verifier run."""

    ROLE = "finalizer"
    AGENT_TAGS = ("terminal", "final_answer")
    ALLOWED_TOOLS = {"final_answer"}
    FROZEN_SYSTEM_PROMPT = (
        "You are the Finalizer. Your only tool is `final_answer(summary)`, "
        "which ends the current episode early and triggers a final verifier "
        "run. Submitting only ends the episode — it earns no bonus reward "
        "(terminal_output_bonus_scale=0). Use it when the team has nothing "
        "left to do; otherwise stay asleep and let the others keep iterating."
    )
    TRAINABLE_SYSTEM_PROMPT = (
        "Wake up ONLY when the step budget is nearly exhausted AND no agent "
        "has more concrete edits queued. There is no longer any reward bonus "
        "for submitting, so the only reason to call final_answer is to save "
        "remaining steps when no further progress is expected. If unsure, "
        "stay asleep — the program persists across episodes anyway, so a new "
        "episode can keep iterating."
    )


CLOUDCAST_AGENT_CLASSES = [
    PlannerAgent,
    ReaderAgent,
    ImplementerAgent,
    BuilderAgent,
    EvaluatorAgent,
    FinalizerAgent,
]


# Legacy single-agent config — kept as an alias so existing code that
# references ``CoderAgent`` does not break. New setups should use the
# specialized roster above.
class CoderAgent(CloudcastAgent):
    """Legacy all-in-one agent (kept for back-compat with old configs)."""

    ROLE = "coder"
    AGENT_TAGS = ("terminal", "final_answer")
    ALLOWED_TOOLS = {"write_file", "read_file", "shell", "request_eval", "final_answer"}
    SELF_STARTING = True
    FROZEN_SYSTEM_PROMPT = (
        "You are a senior software engineer solving a cloudcast task. "
        "You have a scratch workspace and can freely create / edit files and run "
        "shell commands inside it. The verifier scores your submission with a "
        "continuous metric — partial progress counts."
    )
    TRAINABLE_SYSTEM_PROMPT = (
        "Work incrementally. First skim the instruction and any existing "
        "scaffolding in the workspace, then build the smallest submission that "
        "the verifier will score above zero; only after that should you try to "
        "improve the score. Call request_eval() sparingly — it is slow. When "
        "you are out of useful ideas or close to the step budget, call "
        "final_answer(summary)."
    )


_AGENT_CLASS_REGISTRY: Dict[str, type] = {
    "CloudcastAgent": CloudcastAgent,
    "CoderAgent": CoderAgent,
    "PlannerAgent": PlannerAgent,
    "ReaderAgent": ReaderAgent,
    "ImplementerAgent": ImplementerAgent,
    "BuilderAgent": BuilderAgent,
    "EvaluatorAgent": EvaluatorAgent,
    "FinalizerAgent": FinalizerAgent,
}
