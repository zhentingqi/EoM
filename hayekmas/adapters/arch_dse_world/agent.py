"""
arch_dse_world agent layer.

Three roles, strict information chain:

    history.jsonl ←(auto)─ Executor ←(direction)── Planner ←(advice)── Historian

All instances within a single role share the same FROZEN_SYSTEM_PROMPT.
Diversity between instances emerges from wakeup-timing differences:
different agents wake at different episode points, so each reads a
different snapshot of the shared history, and their `experience` field
accumulates independently.

Exactly one advice is on the bus per round: wakeup gating lets a
Historian emit only when the advice/direction buffers are clear, and the
base auction then picks a single winner among the eligible Historians.
The Planner reads that one advice and echoes its author (via a
`<listen_to>` tag) so the credit chain pays the right Historian:
Executor → Planner → Historian-i.

The Executor can submit multiple candidates per turn: in one act() call
it emits 1..K (hardware, mapping) pairs that the env evaluates through
its Timeloop + Accelergy bridge.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from hayekmas.base.agent import BaseAction, BaseAgent, AgentStatus
from hayekmas.adapters.arch_dse_world.env import (
    DSEEnv,
    DSEAction,
    run_timeloop,
    intent_to_mapping,
    validate_mapping_locally,
    _load_workload_prob,
)
from hayekmas.adapters.arch_dse_world.prompts import (
    FROZEN_SYSTEM_PROMPT_HISTORIAN,
    FROZEN_SYSTEM_PROMPT_PLANNER,
    FROZEN_SYSTEM_PROMPT_EXECUTOR,
    HISTORIAN_INSTANCE_TEMPLATE,
    PLANNER_INSTANCE_TEMPLATE,
    EXECUTOR_INSTANCE_TEMPLATE,
    GOOD_BIRTH_PROMPT,
    BAD_BIRTH_PROMPT,
    ACTION_SPACE_BRIEF,
    MODE_HINT_EXPLORER,
    MODE_HINT_EXPLOITER,
    render,
)

if TYPE_CHECKING:
    pass


# ═══════════════════════════════════════════════════════════════════════════
# BASE
# ═══════════════════════════════════════════════════════════════════════════

class DSEAgentBase(BaseAgent):
    """Common machinery for H / P / E roles."""

    AGENT_TAGS: tuple = ("research",)
    ROLE: str = ""
    FROZEN_SYSTEM_PROMPT: str = ""

    def __init__(
        self,
        backbone_llm: Callable[[str], str],
        name: str = "",
        experience: str = "",
        initial_bid: Optional[float] = None,
        initial_wealth: float = 0.0,
        logger=None,
        notebook_dir: Optional[Path] = None,
    ):
        if not name:
            name = f"{self.ROLE.title()}-{BaseAgent._id_counter + 1}"
        super().__init__(name=name, initial_bid=initial_bid, initial_wealth=initial_wealth)
        self.backbone_llm = backbone_llm
        self.logger = logger
        self.experience: str = experience
        self.notebook_dir = Path(notebook_dir) if notebook_dir else None
        if self.notebook_dir:
            self.notebook_dir.mkdir(parents=True, exist_ok=True)

    @property
    def trainable_system_prompt(self) -> str:  # type: ignore[override]
        return getattr(self, "experience", "") or ""

    @trainable_system_prompt.setter
    def trainable_system_prompt(self, value: str) -> None:
        if value:
            self.experience = value

    # ─── private notebook ───────────────────────────────────────────────────

    def _notebook_path(self) -> Optional[Path]:
        if not self.notebook_dir:
            return None
        return self.notebook_dir / f"{self.name}.jsonl"

    def append_notebook(self, entry: Dict[str, Any]) -> None:
        path = self._notebook_path()
        if not path:
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_notebook(self) -> List[Dict[str, Any]]:
        path = self._notebook_path()
        if not path or not path.is_file():
            return []
        out: List[Dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

    # ─── small helpers ─────────────────────────────────────────────────────

    def _log(self, msg: str, **kwargs):
        if self.logger:
            self.logger.log(msg, **kwargs)

    def get_system_prompt(self) -> str:
        if self.frozen_system_prompt:
            return self.frozen_system_prompt + "\n\n" + (self.experience or "")
        return self.experience or ""

    # ─── serialization ─────────────────────────────────────────────────────

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
            "experience": self.experience,
            "trainable_system_prompt": self.experience,
            "root_ancestor_class": getattr(self, "root_ancestor_class", self.__class__.__name__),
            "father_agent_id": getattr(self, "father_agent_id", self.id),
            "father_agent_name": getattr(self, "father_agent_name", self.name),
            "parent_agent_id": getattr(self, "parent_agent_id", None),
            "parent_agent_name": getattr(self, "parent_agent_name", None),
            "spawn_method": getattr(self, "spawn_method", "initial"),
            "tasks_lived": getattr(self, "tasks_lived", 0),
            "bankruptcy_episode": getattr(self, "bankruptcy_episode", None),
        }

    def __repr__(self) -> str:
        bid_str = f"${self.get_bid():.2f}" if self.get_bid() is not None else "None"
        return f"{self.__class__.__name__}({self.name}, bid={bid_str}, wealth=${self.wealth:.2f})"

    # ─── births dispatched per-role ─────────────────────────────────────────

    @classmethod
    def good_birth(
        cls,
        parent: "DSEAgentBase",
        all_role_agents: List["DSEAgentBase"],
        backbone_llm: Callable[[str], str],
        initial_wealth: float = 0.0,
    ) -> "DSEAgentBase":
        blocks: List[str] = []
        for ag in all_role_agents:
            entries = ag.read_notebook()[-20:]
            blocks.append(f"## Notebook of {ag.name} (wealth=${ag.wealth:.2f})\n"
                          + "\n".join(json.dumps(e, ensure_ascii=False) for e in entries))
        all_notebooks_block = "\n\n".join(blocks) if blocks else "(no notebooks recorded yet)"

        meta_prompt = render(
            GOOD_BIRTH_PROMPT,
            role_name=cls.ROLE.title(),
            frozen_system_prompt=parent.frozen_system_prompt,
            all_notebooks_block=all_notebooks_block,
        )
        try:
            new_experience = backbone_llm(meta_prompt).strip()
        except Exception:
            new_experience = parent.experience
        new_experience = _strip_wrap(new_experience)
        if not new_experience or len(new_experience) < 20:
            new_experience = parent.experience

        suffix = random.randint(1000, 9999)
        base = parent.name.split("-")[0] if "-" in parent.name else parent.name
        return cls(
            backbone_llm=backbone_llm,
            name=f"{base}-g{suffix}",
            experience=new_experience,
            initial_wealth=initial_wealth,
            logger=parent.logger,
            notebook_dir=parent.notebook_dir,
        )

    @classmethod
    def bad_birth(
        cls,
        source_agent: "DSEAgentBase",
        backbone_llm: Callable[[str], str],
        initial_wealth: float = 0.0,
    ) -> "DSEAgentBase":
        own_entries = source_agent.read_notebook()[-30:]
        own_block = "\n".join(json.dumps(e, ensure_ascii=False) for e in own_entries) or "(empty notebook)"

        meta_prompt = render(
            BAD_BIRTH_PROMPT,
            role_name=cls.ROLE.title(),
            frozen_system_prompt=source_agent.frozen_system_prompt,
            old_experience=source_agent.experience,
            own_notebook_block=own_block,
        )
        try:
            new_experience = backbone_llm(meta_prompt).strip()
        except Exception:
            new_experience = source_agent.experience
        new_experience = _strip_wrap(new_experience)
        if not new_experience or len(new_experience) < 20:
            new_experience = source_agent.experience

        suffix = random.randint(1000, 9999)
        base = source_agent.name.split("-")[0] if "-" in source_agent.name else source_agent.name
        return cls(
            backbone_llm=backbone_llm,
            name=f"{base}-r{suffix}",
            experience=new_experience,
            initial_wealth=initial_wealth,
            logger=source_agent.logger,
            notebook_dir=source_agent.notebook_dir,
        )


# ═══════════════════════════════════════════════════════════════════════════
# HISTORIAN
# ═══════════════════════════════════════════════════════════════════════════

class HistorianDSEAgent(DSEAgentBase):
    ROLE = "historian"
    AGENT_TAGS = ("historian", "research")
    FROZEN_SYSTEM_PROMPT = FROZEN_SYSTEM_PROMPT_HISTORIAN

    HISTORY_WAKE_THRESHOLD = 0   # bootstrap immediately (chain self-sustains)
    HISTORY_REWAKE_DELTA = 1     # wake on any new history (was 5 — chain died
                                 # because E's 1-3 submits/turn never reached 5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history_size_at_last_wake = -1

    def match_wakeup_condition(self, env: DSEEnv) -> bool:
        if env.terminated or env.calls_used >= env.max_calls:
            return False
        # During preheat sessions the Executor runs solo, so the
        # Historian sits out.
        if getattr(env, "in_preheat", False):
            return False
        # Strict chain: H only wakes when both buffers are clear (i.e. the
        # previous E has fully submitted and the chain has reset). Without
        # this, H could win the auction after P emits direction but before
        # E executes — E would then pay its bid to H instead of P, breaking
        # the credit chain that pays the producer of the direction E used.
        if env.advice_buffer is not None:
            return False
        if env.direction_buffer is not None:
            return False
        n = len(env.history)
        last = self._history_size_at_last_wake
        # Bootstrap: H always speaks first if it has never woken on this env
        # (covers fresh-task case too — env reset shrinks history).
        if last < 0 or n < last:
            return True
        return n - last >= self.HISTORY_REWAKE_DELTA

    def act(self, env: DSEEnv) -> DSEAction:
        prompt = render(
            HISTORIAN_INSTANCE_TEMPLATE,
            frozen_system_prompt=self.frozen_system_prompt,
            experience=self.experience or "(no prior experience)",
            task_block=env.render_task_block(),
            # Pass the FULL submit history (every submit + sim result) to
            # all roles. Cap=200 sits well above the realistic per-task max
            # (budget × sessions ≤ 50). Inter-agent messages stay immediate
            # (no advice/direction history).
            history_block=env.render_history_for_agent(max_entries=200),
            best_edp=f"{env.best_edp:.3e}" if env.best_edp is not None else "n/a",
            submits_since_last_break=env.submits_since_last_break(),
            calls_used=env.calls_used,
            max_calls=env.max_calls,
        )
        raw = self.backbone_llm(prompt).strip()
        self._history_size_at_last_wake = len(env.history)

        # In the three-role design, the Historian emits advice text and
        # the Planner gates on whether advice is fresh — no separate
        # raise-hand role.
        action = DSEAction(text=raw, author=self.name, role=self.role,
                           kind=DSEAction.KIND_ADVICE)
        action.author_id = self.id  # type: ignore[attr-defined]
        self.append_notebook({
            "kind": "advice",
            "step": env.step_count + 1,
            "task_id": env.task.id,
            "history_size_at_emit": len(env.history),
            "best_edp_at_emit": env.best_edp,
            "advice_text": raw,
        })
        return action


# ═══════════════════════════════════════════════════════════════════════════
# PLANNER
# ═══════════════════════════════════════════════════════════════════════════

class PlannerDSEAgent(DSEAgentBase):
    ROLE = "planner"
    AGENT_TAGS = ("planner", "research")
    FROZEN_SYSTEM_PROMPT = FROZEN_SYSTEM_PROMPT_PLANNER

    def match_wakeup_condition(self, env: DSEEnv) -> bool:
        if env.terminated or env.calls_used >= env.max_calls:
            return False
        # During preheat sessions the Executor runs solo, so the Planner
        # sits out.
        if getattr(env, "in_preheat", False):
            return False
        # Planner wakes immediately when fresh advice is on the bus.
        if env.advice_buffer is None:
            return False
        if env.direction_buffer is not None:
            return False
        return True

    def act(self, env: DSEEnv) -> DSEAction:
        # Planner reads ONLY the current advice on the bus (the most
        # recent Historian's emit). Inter-agent communication is
        # immediate; past advice from other Historians is intentionally
        # NOT shown.
        prompt = render(
            PLANNER_INSTANCE_TEMPLATE,
            frozen_system_prompt=self.frozen_system_prompt,
            experience=self.experience or "(no prior experience)",
            task_block=env.render_task_block(),
            advice_buffer=env.render_advice_buffer(),
            history_block=env.render_history_for_agent(max_entries=200),
            best_edp=f"{env.best_edp:.3e}" if env.best_edp is not None else "n/a",
            submits_since_last_break=env.submits_since_last_break(),
            calls_used=env.calls_used,
            max_calls=env.max_calls,
        )
        raw = self.backbone_llm(prompt).strip()

        # The Planner only emits a direction (plus the <listen_to> author);
        # it never submits to the env.
        listened, direction_text = _parse_planner_output(raw)
        action = DSEAction(
            text=direction_text,
            author=self.name,
            role=self.role,
            kind=DSEAction.KIND_DIRECTION,
            chose_historian=listened,
        )
        action.author_id = self.id  # type: ignore[attr-defined]
        self.append_notebook({
            "kind": "direction",
            "step": env.step_count + 1,
            "task_id": env.task.id,
            "listened_to": listened,
            "direction_text": direction_text,
            "best_edp_at_emit": env.best_edp,
        })
        return action


def _parse_planner_output(raw: str) -> tuple[str, str]:
    listened = ""
    listen_match = re.search(r"<listen_to>(.*?)</listen_to>", raw, re.DOTALL | re.IGNORECASE)
    if listen_match:
        listened = listen_match.group(1).strip()
    direction = ""
    dir_match = re.search(r"<direction>(.*?)</direction>", raw, re.DOTALL | re.IGNORECASE)
    if dir_match:
        direction = dir_match.group(1).strip()
    if not direction:
        direction = raw
    return listened, direction


# ═══════════════════════════════════════════════════════════════════════════
# EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════

class ExecutorDSEAgent(DSEAgentBase):
    ROLE = "executor"
    AGENT_TAGS = ("executor", "research", "terminal")
    FROZEN_SYSTEM_PROMPT = FROZEN_SYSTEM_PROMPT_EXECUTOR
    MAX_SUBMITS_PER_TURN = 5
    # Local pre-check retry budget. When the Executor emits candidates
    # that fail the cheap, in-process pre-check (e.g. malformed mapping
    # tokens), we re-prompt the LLM up to this many times with the
    # rejection reasons attached, BEFORE running any of them through the
    # expensive Timeloop simulator. This keeps the per-task budget for
    # *evaluated* submits while still giving the LLM feedback to repair
    # obviously broken candidates.
    MAX_LOCAL_RETRIES_PER_SLOT = 3

    def match_wakeup_condition(self, env: DSEEnv) -> bool:
        if env.terminated or env.calls_used >= env.max_calls:
            return False
        # During preheat sessions, the Executor wakes any time budget
        # remains — there is no Planner direction to wait for.
        if getattr(env, "in_preheat", False):
            return env.budget_remaining() > 0
        return env.direction_buffer is not None

    def _build_prompt(
        self,
        env: DSEEnv,
        *,
        previously_rejected: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Render the Executor instance template, optionally appending a
        rejected-candidates block. The rejection block is built from
        the previous attempt's local pre-check failures so the LLM can
        see *why* its candidates were rejected before re-emitting."""
        rejected_block = ""
        if previously_rejected:
            lines = [
                "# Previously rejected this turn (local pre-check; ZERO budget consumed)",
                (
                    "These candidates failed local validation BEFORE Timeloop ran. "
                    "Read each error and avoid the same mistake; budget is unchanged."
                ),
                "",
            ]
            for i, rej in enumerate(previously_rejected, 1):
                lines.append(
                    f"[{i}] {rej.get('summary', '(candidate)')}\n"
                    f"    error: {rej.get('error', '(unknown)')}"
                )
            rejected_block = "\n".join(lines) + "\n\n"

        base = render(
            EXECUTOR_INSTANCE_TEMPLATE,
            frozen_system_prompt=self.frozen_system_prompt.replace(
                "{{action_space_brief}}", ACTION_SPACE_BRIEF
            ),
            experience=self.experience or "(no prior experience)",
            task_block=env.render_task_block(),
            direction_block=env.render_direction_buffer(),
            # Full submit history with simulator results, shown to all roles.
            history_block=env.render_history_for_agent(max_entries=200),
            budget_remaining=env.budget_remaining(),
            best_edp=f"{env.best_edp:.3e}" if env.best_edp is not None else "n/a",
            submits_since_last_break=env.submits_since_last_break(),
        )
        if rejected_block:
            base = base + "\n\n" + rejected_block
        return base

    def act(self, env: DSEEnv) -> DSEAction:
        # Load workload prob once for intent expansion / pre-check.
        prob = _load_workload_prob(env.workspace)

        # Outer LLM-retry loop. The first generation is "attempt 0"; if
        # the *entire* batch of candidates fails the local pre-check, we
        # re-prompt the LLM up to MAX_LOCAL_RETRIES_PER_SLOT more times
        # with the rejection reasons attached. As soon as at least one
        # candidate passes pre-check, we proceed to Timeloop with the
        # valid subset.
        rationale = ""
        candidates: List[Dict[str, Any]] = []
        previously_rejected: List[Dict[str, str]] = []

        # The first prompt has no rejected block. We'll generate, run candidates
        # through pre-check, and re-prompt if the *entire* batch was rejected.
        for attempt in range(self.MAX_LOCAL_RETRIES_PER_SLOT + 1):
            prompt = self._build_prompt(env, previously_rejected=previously_rejected)
            raw = self.backbone_llm(prompt).strip()

            # Parse the Executor's text into a rationale + a list of
            # (hardware, mapping) candidates to evaluate.
            rationale, candidates = _parse_executor_output(raw, prob=prob)

            # Pre-check ALL candidates locally; surface failed ones for next-attempt feedback.
            this_attempt_rejected: List[Dict[str, str]] = []
            valid_candidates: List[Dict[str, Any]] = []
            for cand in candidates:
                hw = cand.get("hw") or {}
                mapping = cand.get("mapping") or ""
                intent_error = cand.get("intent_error")
                # Reject early if intent expansion failed
                if intent_error and not mapping:
                    this_attempt_rejected.append({
                        "summary": f"hw={hw} intent={cand.get('intent')!r}",
                        "error": f"intent->mapping: {intent_error}",
                    })
                    continue
                if not mapping:
                    this_attempt_rejected.append({
                        "summary": f"hw={hw} (no mapping field)",
                        "error": "no mapping (and no intent) provided",
                    })
                    continue
                if prob:
                    ok_pre, err_pre = validate_mapping_locally(prob, hw, mapping)
                    if not ok_pre:
                        this_attempt_rejected.append({
                            "summary": f"hw={hw} mapping={mapping[:80]!r}",
                            "error": err_pre,
                        })
                        continue
                valid_candidates.append(cand)

            if valid_candidates:
                # We have at least one good candidate — proceed to submit.
                # Carry forward the rejection list as a notebook entry but DO NOT
                # block; mixing valid + invalid in the same batch is fine.
                if this_attempt_rejected:
                    self.append_notebook({
                        "kind": "local_precheck_partial_reject",
                        "step": env.step_count + 1,
                        "task_id": env.task.id,
                        "attempt": attempt,
                        "n_rejected": len(this_attempt_rejected),
                        "n_kept": len(valid_candidates),
                        "rejected": this_attempt_rejected,
                    })
                candidates = valid_candidates
                break

            # ALL candidates failed local pre-check → retry the LLM.
            previously_rejected = (previously_rejected + this_attempt_rejected)[-8:]
            self.append_notebook({
                "kind": "local_precheck_full_reject",
                "step": env.step_count + 1,
                "task_id": env.task.id,
                "attempt": attempt,
                "n_rejected": len(this_attempt_rejected),
                "rejected": this_attempt_rejected,
            })

            if attempt >= self.MAX_LOCAL_RETRIES_PER_SLOT:
                # Exhausted retries; emit empty submit (no eval calls consumed).
                self.append_notebook({
                    "kind": "slot_exhausted",
                    "step": env.step_count + 1,
                    "task_id": env.task.id,
                    "n_attempts": attempt + 1,
                    "final_rejected": this_attempt_rejected,
                })
                action = DSEAction(
                    text=(rationale or "(slot exhausted: all candidates failed local pre-check)"),
                    author=self.name,
                    role=self.role,
                    kind=DSEAction.KIND_SUBMIT,
                    submissions=[],
                )
                action.author_id = self.id  # type: ignore[attr-defined]
                return action

        cap = min(self.MAX_SUBMITS_PER_TURN, env.budget_remaining())
        candidates = candidates[:cap]

        submissions: List[Dict[str, Any]] = []
        for cand in candidates:
            hw = cand.get("hw") or {}
            mapping = cand.get("mapping") or ""
            r = run_timeloop(env.workspace, hw=hw, mapping=mapping, rationale=rationale)
            # Defense-in-depth: if the pre-check fires again here (rare,
            # since we already filtered above), do not charge the budget
            # for the failed candidate.
            local_rejected = bool(r.raw.get("local_pre_check_failed"))
            calls_used_after = int(
                r.raw.get("calls_used", env.calls_used if local_rejected else env.calls_used + 1)
            )
            submissions.append({
                "hw": hw,
                "mapping": mapping,
                "intent": cand.get("intent"),
                "edp": r.edp,
                "cycles": r.cycles,
                "energy_uJ": r.energy_uJ,
                "valid": r.valid,
                "raw": r.raw,
                "calls_used_after": calls_used_after,
            })
            self.append_notebook({
                "kind": "submit",
                "step": env.step_count + 1,
                "task_id": env.task.id,
                "hw": hw,
                "mapping": mapping,
                "intent": cand.get("intent"),
                "edp": r.edp,
                "valid": r.valid,
                "local_pre_check_failed": local_rejected,
                "rationale": rationale,
            })
            if r.raw.get("remaining", 1) == 0 or r.raw.get("calls_used", 0) >= env.max_calls:
                break

        action = DSEAction(
            text=rationale,
            author=self.name,
            role=self.role,
            kind=DSEAction.KIND_SUBMIT,
            submissions=submissions,
        )
        action.author_id = self.id  # type: ignore[attr-defined]
        return action


_EXEC_JSON_RE = re.compile(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", re.DOTALL)


def _parse_executor_output(
    raw: str,
    *,
    prob: Optional[Dict[str, int]] = None,
) -> tuple[str, List[Dict[str, Any]]]:
    """Pull rationale + candidates from a fenced JSON block. Supports two
    candidate schemas:

      Old (raw mapping fallback):
          {"hw": {...}, "mapping": "L3[WIO] ..."}

      New (structured intent — preferred):
          {"hw": {...}, "intent": {"K_spatial_L2": 16, "C_spatial_L1": 16, ...}}

    If `intent` is present and `prob` is provided, we deterministically
    expand it via intent_to_mapping(); a successful expansion sets the
    candidate's `mapping` AND records the intent. If intent expansion fails,
    we still record the failure so the retry loop can show the error to the
    LLM next time. Each candidate dict in the returned list has keys:
        hw           : dict
        mapping      : str  (final symbolic string; "" if intent failed)
        intent       : dict | None  (the intent the LLM gave, if any)
        intent_error : str | None   (error from intent_to_mapping if it failed)
    """
    m = _EXEC_JSON_RE.search(raw)
    blob = m.group(1) if m else None
    if blob is None:
        s = raw.find("{")
        e = raw.rfind("}")
        if 0 <= s < e:
            blob = raw[s : e + 1]
    if not blob:
        return ("(could not parse Executor output as JSON)", [])
    try:
        data = json.loads(blob)
    except Exception:
        return ("(JSON parse failure)", [])
    rationale = str(data.get("rationale", "")).strip()
    raw_cands = data.get("candidates", []) or []
    out: List[Dict[str, Any]] = []
    for c in raw_cands:
        if not isinstance(c, dict):
            continue
        hw = c.get("hw") or {}
        if not isinstance(hw, dict):
            continue
        intent = c.get("intent")
        mapping = c.get("mapping") or ""
        intent_error: Optional[str] = None

        # Prefer structured intent if provided
        if isinstance(intent, dict):
            if prob:
                ok, mapping_from_intent, err = intent_to_mapping(hw, intent, prob)
                if ok and mapping_from_intent:
                    mapping = mapping_from_intent
                else:
                    intent_error = err
                    # If LLM only gave intent (no fallback mapping), keep
                    # mapping="" so retry loop sees the failure.
            else:
                intent_error = "intent given but workload prob not available"

        # Backward compatibility: raw mapping path
        if not mapping or not isinstance(mapping, str):
            if intent_error is None:
                intent_error = "missing both mapping and intent fields"
            out.append({
                "hw": hw,
                "mapping": "",
                "intent": intent if isinstance(intent, dict) else None,
                "intent_error": intent_error,
            })
            continue
        out.append({
            "hw": hw,
            "mapping": mapping.strip(),
            "intent": intent if isinstance(intent, dict) else None,
            "intent_error": intent_error,
        })
    return rationale, out


def _strip_wrap(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    for pre in ("Experience:", "experience:", "## ", "# "):
        if s.startswith(pre):
            s = s[len(pre):].strip()
    return s


# ═══════════════════════════════════════════════════════════════════════════
# REGISTRY + STARTER POPULATION
# ═══════════════════════════════════════════════════════════════════════════

DSE_AGENT_CLASSES = [HistorianDSEAgent, PlannerDSEAgent, ExecutorDSEAgent]

_AGENT_CLASS_REGISTRY: Dict[str, type] = {
    "HistorianDSEAgent": HistorianDSEAgent,
    "PlannerDSEAgent": PlannerDSEAgent,
    "ExecutorDSEAgent": ExecutorDSEAgent,
}


def deserialize_dse_agent(
    data: Dict[str, Any],
    backbone_llm: Optional[Callable[[str], str]] = None,
    logger=None,
    notebook_dir: Optional[Path] = None,
) -> DSEAgentBase:
    cls_name = data.get("type", "HistorianDSEAgent")
    cls = _AGENT_CLASS_REGISTRY.get(cls_name, HistorianDSEAgent)
    if backbone_llm is None:
        def placeholder(_p): raise RuntimeError("backbone_llm not bound")
        backbone_llm = placeholder
    agent = cls(
        backbone_llm=backbone_llm,
        name=data.get("name", cls.__name__),
        experience=data.get("experience", data.get("trainable_system_prompt", "")),
        initial_bid=data.get("bid"),
        initial_wealth=data.get("wealth", 0.0),
        logger=logger,
        notebook_dir=notebook_dir,
    )
    if "id" in data:
        agent.id = data["id"]
    if "wealth" in data:
        agent.wealth = data["wealth"]
    agent.capability_score = data.get("capability_score", 0.0)
    if isinstance(data.get("status"), str):
        agent.set_status(AgentStatus(data["status"]))
    agent.root_ancestor_class = data.get("root_ancestor_class", agent.__class__.__name__)
    agent.father_agent_id = data.get("father_agent_id", agent.id)
    agent.father_agent_name = data.get("father_agent_name", agent.name)
    agent.parent_agent_id = data.get("parent_agent_id")
    agent.parent_agent_name = data.get("parent_agent_name")
    agent.spawn_method = data.get("spawn_method", "initial")
    agent.tasks_lived = data.get("tasks_lived", 0)
    return agent


def build_starter_population(
    backbone_llm: Callable[[str], str],
    *,
    initial_wealth: float,
    n_historian: int = 3,
    n_planner: int = 3,
    n_executor: int = 3,
    notebook_dir: Optional[Path] = None,
    logger=None,
    experience_priors: Optional[Dict[str, List[str]]] = None,
    role_mode_split: bool = False,
) -> List[DSEAgentBase]:
    """3-agent topology starter pop: H/P/E only (no Raiser).

    If `experience_priors` is provided as e.g.
        {"historian": ["bias toward K-spatial...", "bias toward PQ-at-L1..."],
         "planner":   ["refine around best...", "explore new axis..."],
         "executor":  ["pe_dim ∈ {16,32}...", "pe_dim ∈ {64,128}..."]}
    each starter agent in role R is seeded with one prior, rotating through
    the list. The auction mechanism then naturally amplifies the prior whose
    bias matches the layer's optimum (high wealth, more births) and starves
    the wrong-prior agents (low wealth, eventual bankruptcy → bad-birth
    repair). This is the seeded-priors mechanism showcase: instead of all
    starter agents being identical empty-experience copies, they start with
    diverse competing strategies that the market filters.

    With ``role_mode_split=True``, within each role the FIRST starter
    agent gets an [EXPLORER] mode-hint prefix and the rest get [EXPLOITER].
    Both hints are short (~50 words) and sit ABOVE the per-layer prior so
    the prior still drives concrete strategy — the hint only nudges the
    agent's stance (breadth vs. depth).
    """
    priors = experience_priors or {}
    def _prior(role: str, idx: int) -> str:
        lst = priors.get(role) or []
        return lst[idx % len(lst)] if lst else ""

    def _experience(role: str, idx: int) -> str:
        prior = _prior(role, idx)
        if not role_mode_split:
            return prior
        # Slot 0 = explorer; others = exploiters. Hint sits above prior.
        hint = MODE_HINT_EXPLORER if idx == 0 else MODE_HINT_EXPLOITER
        return f"{hint}\n{prior}".strip()

    pop: List[DSEAgentBase] = []
    for i in range(n_historian):
        pop.append(HistorianDSEAgent(
            backbone_llm=backbone_llm,
            name=f"Historian-{i + 1}",
            experience=_experience("historian", i),
            initial_wealth=initial_wealth,
            logger=logger,
            notebook_dir=notebook_dir,
        ))
    for i in range(n_planner):
        pop.append(PlannerDSEAgent(
            backbone_llm=backbone_llm,
            name=f"Planner-{i + 1}",
            experience=_experience("planner", i),
            initial_wealth=initial_wealth,
            logger=logger,
            notebook_dir=notebook_dir,
        ))
    for i in range(n_executor):
        pop.append(ExecutorDSEAgent(
            backbone_llm=backbone_llm,
            name=f"Executor-{i + 1}",
            experience=_experience("executor", i),
            initial_wealth=initial_wealth,
            logger=logger,
            notebook_dir=notebook_dir,
        ))
    return pop
