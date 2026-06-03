"""
Researchworld adapter environment.

Tasks are long-form scientific-reasoning questions. A rubric (list of
``Points: X, Item: ...`` entries) is the ground-truth signal; we score the
final answer with an LLM judge that reads the rubric and the candidate
solution.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from hayekmas.base.agent import BaseAction
from hayekmas.base.env import BaseEnv
from hayekmas.base.config import DEFAULT_HAYEK_CONFIG, RewardConfig
from hayekmas.adapters.researchworld.agent import ResearchAction


# ═══════════════════════════════════════════════════════════════════════════
# TASK
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ResearchTask:
    """One long-form scientific-reasoning task.

    Attributes:
        id: Stable task id (falls back to ``task_group_id`` or row index).
        problem: Full problem text (context + question).
        rubric: Raw rubric string as provided in the dataset.
        subject: Optional subject tag (``physics``/``chemistry``/...).
    """

    id: str
    problem: str
    rubric: str
    subject: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════

def get_default_research_data_dir() -> Path:
    """Return the adapter-local directory for researchworld task splits."""
    return Path(__file__).resolve().parent / "configs"


def get_default_research_test_path() -> Path:
    """Default eval/test JSONL when none is configured."""
    return get_default_research_data_dir() / "research_test.jsonl"


def get_default_research_train_path() -> Path:
    """Default train JSONL; falls back to eval/test if train is absent."""
    candidate = get_default_research_data_dir() / "research_train.jsonl"
    return candidate if candidate.exists() else get_default_research_test_path()


def _task_id_from_row(row: Dict[str, Any], index: int) -> str:
    for key in ("id", "task_group_id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"research_{index:04d}"


def load_tasks_from_jsonl(
    filepath: Path,
    max_tasks: Optional[int] = None,
) -> List[ResearchTask]:
    """Load research tasks from one JSONL file (one task per line)."""
    tasks: List[ResearchTask] = []
    with open(filepath, encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            problem = (row.get("problem") or "").strip()
            rubric = (row.get("answer") or row.get("rubric") or "").strip()
            if not problem or not rubric:
                continue
            tasks.append(
                ResearchTask(
                    id=_task_id_from_row(row, idx),
                    problem=problem,
                    rubric=rubric,
                    subject=(row.get("subject") or "").strip(),
                )
            )
            if max_tasks is not None and len(tasks) >= max_tasks:
                break
    return tasks


def load_research_tasks(
    paths: List[Path],
    limit: Optional[int] = None,
) -> List[ResearchTask]:
    """Load researchworld tasks from one or more JSONL files."""
    all_tasks: List[ResearchTask] = []
    for path in paths:
        remaining = None if limit is None else limit - len(all_tasks)
        if remaining is not None and remaining <= 0:
            break
        all_tasks.extend(load_tasks_from_jsonl(path, max_tasks=remaining))
    return all_tasks


# ═══════════════════════════════════════════════════════════════════════════
# RUBRIC-BASED LLM JUDGE
# ═══════════════════════════════════════════════════════════════════════════

def _parse_judge_response(response: str) -> Tuple[float, str]:
    """Extract (score, reason) from the judge's text reply."""
    score = 0.0
    reason = ""
    for line in response.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper.startswith("SCORE:"):
            rest = stripped[6:].strip().lstrip(":").strip()
            try:
                token = rest.split()[0].rstrip(",")
                score = float(token)
            except (ValueError, IndexError):
                pass
            score = max(0.0, min(1.0, score))
        elif upper.startswith("REASON:"):
            reason = stripped[7:].strip().lstrip(":").strip()
    if not reason:
        reason = f"Score {score:.3f} (no reason parsed)."
    return score, reason


def score_answer_against_rubric(
    llm_fn: Callable[[str], str],
    problem: str,
    rubric: str,
    actual_output: str,
    threshold: float = 0.5,
) -> Tuple[bool, float, str]:
    """Score a candidate solution against a rubric using an LLM judge.

    Returns ``(passed, score, reason)``. ``passed`` is ``score >= threshold``.
    Any exception falls back to ``(False, 0.0, "<error>")`` so periodic tests
    continue even when a single judge call fails.
    """
    # Import inside the function to avoid a hard circular dependency at module
    # load time (prompts imports nothing from env, but keep this defensive).
    from hayekmas.adapters.researchworld.prompts import RUBRIC_JUDGE_PROMPT

    prompt = (
        RUBRIC_JUDGE_PROMPT
        .replace("<<<problem>>>", problem)
        .replace("<<<rubric>>>", rubric)
        .replace("<<<actual_output>>>", actual_output or "")
    )
    try:
        response = llm_fn(prompt).strip()
        score, reason = _parse_judge_response(response)
        return score >= threshold, score, reason
    except Exception as exc:  # noqa: BLE001
        return False, 0.0, f"Rubric judge error: {exc!r}"


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════

_FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL | re.IGNORECASE)


class ResearchEnv(BaseEnv):
    """Environment for one long-form scientific-reasoning task.

    A turn is a free-form ``<step>`` block authored by one agent. The Answer
    agent (the only terminal-tagged class) may additionally emit
    ``<final_answer>...</final_answer>``; when that happens the env runs the
    rubric-based LLM judge and terminates.
    """

    def __init__(
        self,
        task: str,
        rubric: str,
        *,
        reward_config: Optional[RewardConfig] = None,
        use_judge: bool = True,
        judge_threshold: float = 0.5,
        judge_verbose: bool = False,
        max_steps: int = 10,
        subject: str = "",
    ):
        super().__init__()

        self.task = task
        self.expected_output = rubric
        self.subject = subject
        self.name = f"research_{hash(task) % 100000:05d}"
        self.reward_config = (
            deepcopy(reward_config)
            if reward_config is not None
            else deepcopy(DEFAULT_HAYEK_CONFIG.reward)
        )
        self.use_judge = use_judge
        self.judge_threshold = judge_threshold
        self.judge_verbose = judge_verbose
        self.max_steps = max_steps

        # Filled in externally by the runtime (so the env can call the same
        # LLM as the agents to grade its own answer).
        self.llm_fn: Optional[Callable[[str], str]] = None

        # Populated by ``_handle_final_answer``.
        self.final_answer: Optional[str] = None
        self._last_terminal_score: Optional[float] = None
        self._last_judge_reason: Optional[str] = None
        self._last_judge_passed: Optional[bool] = None
        self._final_answer_author: Optional[str] = None
        self._last_final_reward: float = 0.0

        self.init_state: str = ""
        self.state: str = ""

        self.initialize()

    # ─────────────────────────────────────────────────────────────────────────
    # Env lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def initialize(self):
        self.step_count = 0
        self.terminated = False
        self.action_history = {}
        self.final_answer = None
        self._last_terminal_score = None
        self._last_judge_reason = None
        self._last_judge_passed = None
        self._final_answer_author = None
        self._last_final_reward = 0.0

        self.init_state = (
            "## Scientific problem\n\n"
            f"{self.task}\n\n"
            "## Shared solution notes (contributions from each agent will appear below)\n"
        )
        self.state = self.init_state

    def get_state_description(self) -> str:
        status = "terminated" if self.terminated else "in_progress"
        return f"ResearchEnv({self.name}, step={self.step_count}, {status})"

    def get_terminal_score(self) -> Optional[float]:
        return self._last_terminal_score

    def get_task_description(self) -> str:
        return self.task

    def get_correct_answer(self) -> str:
        return self.expected_output

    def get_final_answer_author(self) -> Optional[str]:
        return self._final_answer_author

    def get_last_final_reward(self) -> float:
        return self._last_final_reward

    def is_successful(self) -> Optional[bool]:
        return self._last_judge_passed

    # ─────────────────────────────────────────────────────────────────────────
    # Action application
    # ─────────────────────────────────────────────────────────────────────────

    def apply(self, action: BaseAction) -> float:
        if self.terminated:
            return 0.0
        if not isinstance(action, ResearchAction):
            raise TypeError("ResearchEnv expects actions of type ResearchAction.")

        self.step_count += 1
        step_text = (action.text or "").strip()

        self.action_history[self.step_count] = {
            "author": action.author or "",
            "role": action.role or "",
            "text": step_text,
            "is_final": action.is_final,
        }

        author_prefix = f"[Message from agent: {action.author}] " if action.author else ""
        if step_text:
            self.state += f"\n\n{author_prefix}{step_text}"

        if action.is_final and action.role == "answer":
            final_answer = action.final_answer_text or step_text
            return self._handle_final_answer(final_answer, action.author)
        return 0.0

    def _handle_final_answer(self, answer: str, author: Optional[str]) -> float:
        self.final_answer = answer
        self.terminated = True
        self._final_answer_author = author or None
        passed, score, reason = self.check_answer_correct(answer, self.expected_output)
        score_for_reward = (
            score if score is not None else self.reward_config.missing_terminal_score
        )
        reward = self.reward_config.terminal_output_bonus(score_for_reward)
        self._last_final_reward = reward
        self._last_terminal_score = score
        self._last_judge_reason = reason or ""
        self._last_judge_passed = passed
        self._log("\n🏁 Episode TERMINATED (rubric judge)", indent=2)
        score_str = f"{score:.3f}" if score is not None else "N/A"
        preview = answer.strip().replace("\n", " ")[:160]
        self._log(
            f"   📝 Answer: '{preview}' | {passed} | {score_str} | {reason[:120]} | Reward: {reward:.4f}",
            indent=2,
        )
        return reward

    # ─────────────────────────────────────────────────────────────────────────
    # Grading
    # ─────────────────────────────────────────────────────────────────────────

    def check_answer_correct(
        self,
        model_answer: str,
        rubric: str,
    ) -> Tuple[bool, float, str]:
        """Score ``model_answer`` with the rubric-based LLM judge."""
        if not self.use_judge:
            # Without a judge we have no way to score long-form answers.
            return False, 0.0, "Judge disabled — cannot score long-form answers."
        if self.llm_fn is None:
            return False, 0.0, "No llm_fn attached to env; cannot grade."
        return score_answer_against_rubric(
            llm_fn=self.llm_fn,
            problem=self.task,
            rubric=rubric,
            actual_output=model_answer,
            threshold=self.judge_threshold,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Re-export helper used by other modules
# ═══════════════════════════════════════════════════════════════════════════

def extract_final_answer_block(text: str) -> Optional[str]:
    """Return the first ``<final_answer>...</final_answer>`` content or None."""
    match = _FINAL_ANSWER_RE.search(text or "")
    return match.group(1).strip() if match else None
