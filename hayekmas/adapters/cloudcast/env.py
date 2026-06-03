"""Cloudcast environment.

Each episode:
1. Copies the task's ``environment/`` template into a per-episode tmp
   workspace.
2. Hands a whitelisted tool kit to the smolagents Python executor so the
   agent can write files, run shell commands, and request an evaluation.
3. Parses continuous scores out of the task's ``tests/compute_reward.py``
   and feeds the final score through ``RewardConfig.terminal_output_bonus``.

The adapter deliberately does **not** launch the task's Docker image — host
execution is expected to have the required toolchain. When the toolchain is
missing, the verifier fails gracefully and terminal_score is 0.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from hayekmas.adapters.cloudcast.agent import CloudcastAction
from hayekmas.adapters.cloudcast.task import CloudcastTask
from hayekmas.adapters.cloudcast.tools import (
    make_read_file,
    make_request_eval,
    make_shell,
    make_write_file,
)
from hayekmas.base.agent import BaseAction
from hayekmas.base.config import DEFAULT_HAYEK_CONFIG, RewardConfig
from hayekmas.base.env import BaseEnv
from hayekmas.utils.logger import logger


_SENTINEL = object()
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDORED_SMOLAGENTS_SRC = _REPO_ROOT / "third_party" / "smolagents" / "src"


def _ensure_smolagents_importable() -> None:
    if _VENDORED_SMOLAGENTS_SRC.is_dir():
        src = str(_VENDORED_SMOLAGENTS_SRC)
        if src not in sys.path:
            sys.path.insert(0, src)


@dataclass
class VerifierResult:
    """Outcome of one verifier invocation."""

    raw_score: Optional[float]
    normalized_score: Optional[float]
    reason: str
    subscores: list
    raw_reward_json: Optional[Dict[str, Any]] = None

    @property
    def available(self) -> bool:
        return self.raw_score is not None


class CloudcastEnv(BaseEnv):
    """Environment for one cloudcast task.

    Args:
        task: Parsed :class:`CloudcastTask`.
        reward_config: Optional :class:`RewardConfig`.  Terminal score is
            passed through ``reward_config.terminal_output_bonus``.
        max_steps: Episode step budget enforced by the engine loop.
        code_execution_timeout: Per-``<code>``-block wall-clock limit.
        shell_timeout: Default timeout for ``shell()`` calls (s).
        verifier_timeout: Wall-clock limit for ``compute_reward.py`` (s).
        workspace_parent: Directory under which the per-episode workspace is
            created.  Defaults to ``tempfile.gettempdir()``.
        keep_workspace: If ``True``, the workspace is not cleaned up on
            termination — useful when debugging.
    """

    def __init__(
        self,
        task: CloudcastTask,
        *,
        reward_config: Optional[RewardConfig] = None,
        max_steps: int = 10,
        code_execution_timeout: int = 600,
        shell_timeout: int = 300,
        verifier_timeout: int = 1800,
        workspace_parent: Optional[Path] = None,
        keep_workspace: bool = False,
        preserve_workspace_across_episodes: bool = False,
    ):
        super().__init__()
        self.task = task
        self.instruction = task.instruction
        self.expected_output = ""
        self.name = f"cloudcast_{task.id}"
        self.reward_config = (
            deepcopy(reward_config)
            if reward_config is not None
            else deepcopy(DEFAULT_HAYEK_CONFIG.reward)
        )
        self.max_steps = max_steps
        self.code_execution_timeout = code_execution_timeout
        self.shell_timeout = shell_timeout
        self.verifier_timeout = verifier_timeout
        self.workspace_parent = Path(workspace_parent or tempfile.gettempdir())
        self.keep_workspace = keep_workspace
        self.preserve_workspace_across_episodes = preserve_workspace_across_episodes

        self.workspace_dir: Optional[Path] = None
        self.state: str = ""
        self.final_answer: Optional[str] = None
        self.allowed_tool_names: Optional[set[str]] = None

        self._code_executor_state: Dict[str, Any] = {}
        self._last_terminal_score: Optional[float] = None
        self._last_verifier_result: Optional[VerifierResult] = None
        self._last_final_reward: float = 0.0
        self._final_answer_author: Optional[str] = None

        # Checkpoint-reward state: flipped by `request_eval()` mid-episode.
        self._last_checkpoint_score: float = 0.0
        self._checkpoint_fired: bool = False
        self._last_checkpoint_delta: float = 0.0

        # Best-snapshot rollback state. When a verifier run produces a new
        # high score, the workspace's `initial_program.py` (if present) is
        # snapshotted to `.best_program_snapshot.py`; if a later episode
        # starts with a syntactically-broken or runtime-broken program, the
        # snapshot is restored so a single bad write cannot poison every
        # subsequent episode.
        self._best_score_ever: float = float("-inf")
        self._last_verifier_had_error: bool = False
        self._rollbacks_done: int = 0

        # Research telemetry: wakeup judge results per step, plus current-agent
        # overrides for per-role shell timeouts.
        self._wakeup_log: list = []
        self._current_shell_timeout: Optional[int] = None

        self.initialize()

    # ─────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────

    def initialize(self):
        """Per-episode reset.

        Always resets step_count, action_history, terminated flag, and
        per-episode final-answer state.  The workspace directory and the
        running checkpoint score are preserved across episodes when
        ``preserve_workspace_across_episodes=True``, so that a long task
        can be worked on incrementally over many Hayek episodes.
        """
        self.step_count = 0
        self.terminated = False
        self.action_history = {}
        self.final_answer = None
        self.state = ""
        self._last_terminal_score = None
        self._last_verifier_result = None
        self._last_final_reward = 0.0
        self._final_answer_author = None
        self._checkpoint_fired = False
        self._last_checkpoint_delta = 0.0
        self._wakeup_log = []
        self._current_shell_timeout = None

        if self.preserve_workspace_across_episodes and self.workspace_dir is not None:
            # Keep workspace + code executor state + checkpoint score so
            # each episode is a fresh auction over a persistent codebase.
            # Before handing the workspace to the next episode, roll back
            # `initial_program.py` to the last known-good snapshot if the
            # current copy is broken OR has regressed below the best score
            # ever achieved. Without the regression check, a sequence of
            # episodes that each made the program 1% slower would compound
            # into a much worse program than the seed by ep10.
            self._restore_best_if_regressed()
            return

        # Fresh workspace path: reset executor state and checkpoint score.
        self._code_executor_state = {}
        self._last_checkpoint_score = 0.0

        self.workspace_parent.mkdir(parents=True, exist_ok=True)
        self.workspace_dir = Path(
            tempfile.mkdtemp(prefix=f"cloudcast_{self.task.id}_", dir=self.workspace_parent)
        )
        if self.task.workspace_template and self.task.workspace_template.is_dir():
            # Copy initial state into the workspace root; ignore huge build
            # trees (cloudcast ships `git-src` etc. — we still want those
            # so the agent can read them, so no filter for now).
            for entry in self.task.workspace_template.iterdir():
                dst = self.workspace_dir / entry.name
                if entry.is_dir():
                    shutil.copytree(entry, dst, symlinks=True)
                else:
                    shutil.copy2(entry, dst)

        # Seed-as-baseline snapshots: copy `initial_program.py` to BOTH
        # snapshot paths so we always have a known-good fallback even
        # before the first verifier call. Without this, an agent whose
        # very first write breaks the program leaves us with no rollback
        # target. The seed is guaranteed runnable (it's what the task
        # ships with), so it's safe to use as the initial last-runnable.
        program = self.workspace_dir / self._ROLLBACK_PROGRAM_NAME
        runnable_snapshot = self.workspace_dir / self._ROLLBACK_SNAPSHOT_NAME
        best_snapshot = self.workspace_dir / self._BEST_SNAPSHOT_NAME
        if program.is_file():
            try:
                if not runnable_snapshot.is_file():
                    shutil.copy2(program, runnable_snapshot)
                if not best_snapshot.is_file():
                    shutil.copy2(program, best_snapshot)
                # Bootstrap best=0.0 so the first new high gets snapshot.
                self._best_score_ever = 0.0
                self._log(
                    f"💾 Seed program saved as baseline (last_runnable + best @ score=0.0000)",
                    indent=2,
                )
            except OSError as exc:
                self._log(f"⚠️  Seed snapshot failed: {exc}", indent=2)

    def cleanup(self) -> None:
        if self.workspace_dir and self.workspace_dir.is_dir() and not self.keep_workspace:
            shutil.rmtree(self.workspace_dir, ignore_errors=True)
            self.workspace_dir = None

    # ─────────────────────────────────────────────────────────────────────
    # ENGINE CONTRACT
    # ─────────────────────────────────────────────────────────────────────

    def apply(self, action: BaseAction) -> float:
        if self.terminated:
            return 0.0
        if not isinstance(action, CloudcastAction):
            raise TypeError("CloudcastEnv expects CloudcastAction")

        self.step_count += 1
        step_text = (action.text or "").strip()
        self.action_history[self.step_count] = {
            "author": action.author or "",
            "role": action.role or "",
            "text": step_text,
            "is_final": action.is_final,
            "is_code_action": action.is_code_action,
        }

        display_text = action.thought or step_text
        author_prefix = f"[{action.author}] " if action.author else ""
        if display_text:
            self.state += f"\n\n{author_prefix}{display_text}"

        reward = 0.0
        if not action.is_code_action:
            return reward

        self._log(f"🔧 Executing <code> in {self.workspace_dir}", indent=2)
        code_output, final_value = self._execute_code(action.code or "")
        log_output = code_output if len(code_output) <= 1024 else code_output[:1024] + "\n... [truncated]"
        self._log(f"   Code output:\n{log_output}", indent=2)
        self.state += f"\n\n[Code Output] {code_output}"

        # Checkpoint delta reward: fires when the code block called
        # `request_eval()` without triggering `final_answer()`.  The delta is
        # computed against the last checkpoint score, scaled by
        # `env_reward_scale`, and returned to the step winner via stepwise
        # reward (see HayekMAS._run_auction_action_loop when
        # reward_scheme=path_reward_and_stepwise_reward).
        if self._checkpoint_fired and final_value is _SENTINEL:
            self._checkpoint_fired = False
            delta = self._last_checkpoint_delta
            scaled = self.reward_config.env_reward_scale * delta
            self._log(
                f"📈 Checkpoint: score={self._last_checkpoint_score:.4f} "
                f"delta={delta:+.4f} → reward={scaled:+.4f}",
                indent=2,
            )
            reward += scaled

        if final_value is not _SENTINEL:
            self.final_answer = str(final_value) if final_value is not None else ""
            self.terminated = True
            self._final_answer_author = action.author or None

            result = self._run_verifier()
            self._last_verifier_result = result
            terminal_score = result.normalized_score if result.available else 0.0
            self._last_terminal_score = terminal_score
            # Clear any pending checkpoint fire — the terminal run supersedes it.
            self._checkpoint_fired = False
            self._last_checkpoint_score = terminal_score
            reward += self.reward_config.terminal_output_bonus(terminal_score)
            self._last_final_reward = reward
            self._log(
                f"🏁 Terminated. raw={result.raw_score} norm={terminal_score} reward={reward} "
                f"reason={result.reason[:200]}",
                indent=2,
            )

        return reward

    def _record_checkpoint_score(self, new_score: float) -> None:
        """Callback from the `request_eval` tool: stash the delta for apply().

        Applies asymmetric shaping (RewardConfig.shaped_checkpoint_delta):
        a broken-program eval (configs=0/N) takes a flat penalty, and
        regressions get amplified by ``regression_multiplier``. Positive
        deltas pass through unchanged."""
        raw_delta = new_score - self._last_checkpoint_score
        shaped = self.reward_config.shaped_checkpoint_delta(
            raw_delta,
            was_broken=self._last_verifier_had_error,
        )
        self._last_checkpoint_delta = shaped
        self._last_checkpoint_score = new_score
        self._checkpoint_fired = True

    def record_wakeup(self, agent_name: str, role: str, woke_up: bool) -> None:
        """Log a wakeup judge result for later per-role analysis."""
        self._wakeup_log.append({
            "before_step": self.step_count + 1,
            "agent_name": agent_name,
            "role": role or "",
            "woke_up": bool(woke_up),
        })

    # ─────────────────────────────────────────────────────────────────────
    # CODE EXECUTION
    # ─────────────────────────────────────────────────────────────────────

    def _start_exec_watchdog(self, hard_timeout: float) -> threading.Event:
        """Hard-kill the entire process if a `<code>` execution outruns
        ``hard_timeout`` seconds.

        Background: smolagents' AST interpreter occasionally enters states
        where the worker thread is stuck in native code (networkx C / pandas)
        and `Future.result(timeout=...)` does not raise (or raises but the
        thread leak accumulates). We've observed multiple training runs
        die this way without producing any further log line.

        This watchdog runs in a daemon thread, polls every second, and
        calls ``os._exit(1)`` when the hard timeout is exceeded. The
        cloudcast_parallel.py parent then sees the child exit nonzero and
        marks the run as failed — better than a permanently-hung process.
        Returns a stop ``Event`` the caller sets to cancel the watchdog.
        """
        stop_event = threading.Event()
        started = time.monotonic()

        def _watch():
            while not stop_event.is_set():
                elapsed = time.monotonic() - started
                if elapsed > hard_timeout:
                    msg = (
                        f"\n[code-watchdog] code execution stuck for "
                        f"{elapsed:.0f}s (limit {hard_timeout:.0f}s); "
                        f"hard-exiting process to recover.\n"
                    )
                    try:
                        sys.stderr.write(msg)
                        sys.stderr.flush()
                    except Exception:
                        pass
                    os._exit(1)
                stop_event.wait(timeout=1.0)

        threading.Thread(target=_watch, daemon=True).start()
        return stop_event

    def _execute_code(self, code: str) -> Tuple[str, Any]:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

        _ensure_smolagents_importable()
        from smolagents.local_python_executor import (
            BASE_PYTHON_TOOLS,
            InterpreterError,
            evaluate_python_code,
        )
        from smolagents.utils import BASE_BUILTIN_MODULES

        static_tools = BASE_PYTHON_TOOLS.copy()
        static_tools.update(self._make_tool_functions())
        state = dict(self._code_executor_state)
        state["json"] = json

        def _run():
            return evaluate_python_code(
                code=code,
                static_tools=static_tools,
                custom_tools={},
                state=state,
                authorized_imports=list(BASE_BUILTIN_MODULES),
            )

        timeout = self.code_execution_timeout
        # Hard watchdog: 5× the soft timeout. Set HAYEKMAS_DISABLE_CODE_WATCHDOG=1
        # to skip it (e.g., when you want a long resume to ride out occasional
        # native hangs without auto-kill). Default-on catches GIL-deadlock + the
        # silent native-crash case.
        if os.environ.get("HAYEKMAS_DISABLE_CODE_WATCHDOG", "").strip() in ("1", "true", "yes"):
            hard_watchdog_stop = threading.Event()  # no-op stop event
        else:
            hard_watchdog_stop = self._start_exec_watchdog(hard_timeout=timeout * 5)
        # NOTE: do NOT use `with ThreadPoolExecutor(...) as pool:` here.
        # The context-manager exit calls shutdown(wait=True), which blocks
        # forever if the worker thread is hung in native code (networkx C,
        # pandas IO) — Python cannot cancel a running thread. We saw 3/4
        # parallel runs go zombie this way. Manual lifecycle + wait=False
        # on timeout leaks a thread (OS reclaims at process exit) but
        # keeps the parent process responsive.
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(_run)
            try:
                result, is_final_answer = future.result(timeout=timeout)
            except FuturesTimeoutError:
                pool.shutdown(wait=False)  # don't block on the stuck thread
                self._log(
                    f"⚠️  code execution timed out after {timeout}s "
                    f"(thread abandoned; OS will reclaim on exit)",
                    indent=2,
                )
                return f"Error: code execution timed out after {timeout}s", _SENTINEL
            else:
                pool.shutdown(wait=True)
        except InterpreterError as e:
            msg = str(e)
            if ("Forbidden" in msg or "not among" in msg) and self.allowed_tool_names:
                allowed = ", ".join(sorted(self.allowed_tool_names))
                msg += f"\nYour allowed tools: {allowed}"
            return f"Error: {msg}", _SENTINEL
        except Exception as e:  # noqa: BLE001
            return f"Error: {type(e).__name__}: {e}", _SENTINEL
        finally:
            # Cancel the hard-kill watchdog: code finished one way or another.
            hard_watchdog_stop.set()

        for k, v in state.items():
            if not k.startswith("_"):
                self._code_executor_state[k] = v

        logs = str(state.get("_print_outputs", "")).strip()
        if is_final_answer:
            return (logs + f"\nfinal_answer submitted: {str(result)[:200]}").strip(), result
        if not logs:
            logs = "" if result is None else str(result).strip() or "No Outputs"
        return logs, _SENTINEL

    def _make_tool_functions(self) -> Dict[str, Callable]:
        assert self.workspace_dir is not None, "initialize() must be called first"
        effective_shell_timeout = self._current_shell_timeout or self.shell_timeout
        all_tools: Dict[str, Callable] = {
            "write_file": make_write_file(self.workspace_dir),
            "read_file": make_read_file(self.workspace_dir),
            "shell": make_shell(self.workspace_dir, default_timeout=effective_shell_timeout),
            "request_eval": make_request_eval(
                self._verifier_tool_callback,
                on_checkpoint=self._record_checkpoint_score,
            ),
        }
        # `final_answer` is injected by smolagents itself — we only need to
        # include a sentinel so it is allow-listed.  Providing a Python
        # callable named `final_answer` makes smolagents treat it as the
        # terminal tool.
        all_tools["final_answer"] = _final_answer_sentinel
        if self.allowed_tool_names is None:
            return all_tools
        return {name: fn for name, fn in all_tools.items() if name in self.allowed_tool_names}

    # ─────────────────────────────────────────────────────────────────────
    # VERIFIER
    # ─────────────────────────────────────────────────────────────────────

    def _verifier_tool_callback(self) -> Dict[str, Any]:
        """Shape the VerifierResult for the `request_eval` tool."""
        result = self._run_verifier()
        return {
            "score": result.normalized_score,
            "raw_score": result.raw_score,
            "reason": result.reason,
        }

    def _run_verifier(self) -> VerifierResult:
        """Invoke the task's ``compute_reward.py`` and parse its output."""
        if self.task.verifier_script is None:
            return VerifierResult(None, None, "no verifier script", [])
        assert self.workspace_dir is not None

        out_dir = self.workspace_dir / ".verifier_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use the same interpreter as the hayekmas process so the verifier
        # sees the same site-packages (networkx, pandas, …).  Hardcoding
        # "python3" silently picks up whichever system python is on $PATH,
        # which on macOS conda setups is /usr/bin/python3 with no deps.
        import sys as _sys
        cmd = [
            _sys.executable,
            str(self.task.verifier_script),
            "--app-dir",
            str(self.workspace_dir),
            "--output-dir",
            str(out_dir),
        ]
        # Heartbeat log BEFORE the subprocess so external watchdogs (e.g.
        # cloudcast_parallel.py's stall detector) see fresh log activity
        # even if the verifier itself runs silent for several minutes.
        # Without this, a 4-min verifier silence + a 4-min stall-timeout
        # gives a coin-flip false-positive kill.
        self._log(
            f"🔬 verifier started (timeout {self.verifier_timeout}s) "
            f"in {self.workspace_dir}",
            indent=2,
        )
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.verifier_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            self._log(
                f"⚠️  verifier subprocess timed out after {self.verifier_timeout}s",
                indent=2,
            )
            return VerifierResult(None, None, "verifier timed out", [])
        except FileNotFoundError as exc:
            return VerifierResult(None, None, f"verifier launch failed: {exc}", [])

        reward_json = out_dir / "reward.json"
        raw_data: Optional[Dict[str, Any]] = None
        if reward_json.is_file():
            try:
                raw_data = json.loads(reward_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raw_data = None

        if raw_data is None:
            stderr = (completed.stderr or "")[-400:]
            reason = f"verifier exited {completed.returncode}; no reward.json. stderr: {stderr}"
            return VerifierResult(None, None, reason, [])

        raw_score = _extract_score(raw_data)
        normalized = _normalize_score(raw_score, raw_data)
        subscores = raw_data.get("subscores") or []
        reason = raw_data.get("reason") or f"score={raw_score}"

        # Track whether the verifier hit a runtime / breakage state. Used by
        # _restore_best_if_regressed() to decide whether to roll back at the
        # next initialize().
        self._last_verifier_had_error = (
            raw_data.get("total_cost") is None
            or int(raw_data.get("failed_configs", 0) or 0) > 0
        )
        # Two snapshot-update policies:
        #   1. Update best-snapshot ONLY on a strictly new high score
        #      (kept for reporting / metrics).
        #   2. Update last-runnable snapshot on EVERY clean verifier run
        #      regardless of score — this is the rollback target.
        # Mid-episode rollback fires only when the program is broken
        # (configs=0 / parse error), NOT on score regression. That gives
        # the team room to explore worse-scoring detours that may unlock
        # better basins later.
        if normalized is not None and not self._last_verifier_had_error:
            self._snapshot_runnable_program(normalized)
            if normalized > self._best_score_ever:
                self._snapshot_best_program(normalized)
        elif self._last_verifier_had_error:
            # Program is broken (configs=0/N or parse error). Roll back
            # in-place to the last runnable copy so the rest of the
            # episode iterates from a working baseline.
            self._restore_runnable_if_broken()

        return VerifierResult(
            raw_score=raw_score,
            normalized_score=normalized,
            reason=str(reason),
            subscores=subscores,
            raw_reward_json=raw_data,
        )

    # ─────────────────────────────────────────────────────────────────────
    # ROLLBACK SAFETY NET
    # ─────────────────────────────────────────────────────────────────────

    _ROLLBACK_PROGRAM_NAME = "initial_program.py"
    _ROLLBACK_SNAPSHOT_NAME = ".last_runnable_program.py"   # rollback target
    _BEST_SNAPSHOT_NAME = ".best_program_snapshot.py"      # informational

    def _snapshot_best_program(self, score: float) -> None:
        """Save a copy of the current program when it's a NEW best score.
        Kept for reporting only — the rollback target is the
        last-runnable snapshot, NOT this one. Letting the agent always
        roll back to best locks the team out of exploring detours that
        regress short-term but unlock a deeper basin."""
        if self.workspace_dir is None:
            return
        program = self.workspace_dir / self._ROLLBACK_PROGRAM_NAME
        if not program.is_file():
            return
        snapshot = self.workspace_dir / self._BEST_SNAPSHOT_NAME
        try:
            shutil.copy2(program, snapshot)
            self._best_score_ever = score
            self._log(
                f"💾 Snapshot new best at score={score:.4f} → {snapshot.name}",
                indent=2,
            )
        except OSError as exc:
            self._log(f"⚠️  Best snapshot failed: {exc}", indent=2)

    def _snapshot_runnable_program(self, score: float) -> None:
        """Save a copy of the current program after EVERY clean verifier run
        (configs=N/N), regardless of score. This is the rollback target —
        we restore from it when the program later goes broken, but we
        don't restore on score regression. That way the team can take
        exploratory detours that score worse short-term, as long as the
        program still compiles and runs."""
        if self.workspace_dir is None:
            return
        program = self.workspace_dir / self._ROLLBACK_PROGRAM_NAME
        if not program.is_file():
            return
        snapshot = self.workspace_dir / self._ROLLBACK_SNAPSHOT_NAME
        try:
            shutil.copy2(program, snapshot)
            self._log(
                f"💾 Saved last runnable at score={score:.4f} → {snapshot.name}",
                indent=2,
            )
        except OSError as exc:
            self._log(f"⚠️  Runnable snapshot failed: {exc}", indent=2)

    def _restore_runnable_if_broken(self) -> None:
        """At episode start (or mid-episode): if the program is currently
        broken (configs=0 from last verifier OR fails parse), restore the
        last-runnable snapshot. Does NOT roll back on score regression
        — the new policy is "continue from last compilable program",
        not "continue from best".
        """
        if self.workspace_dir is None:
            return
        program = self.workspace_dir / self._ROLLBACK_PROGRAM_NAME
        snapshot = self.workspace_dir / self._ROLLBACK_SNAPSHOT_NAME
        if not (program.is_file() and snapshot.is_file()):
            return

        # Cheap no-op when we're already at the snapshot.
        try:
            if program.read_bytes() == snapshot.read_bytes():
                return
        except OSError:
            pass

        broken = self._last_verifier_had_error
        if not broken:
            try:
                compile(program.read_text(encoding="utf-8"), str(program), "exec")
            except (SyntaxError, ValueError):
                broken = True

        if not broken:
            return

        try:
            shutil.copy2(snapshot, program)
            self._last_verifier_had_error = False
            self._rollbacks_done += 1
            self._log(
                f"⏪ Restored last-runnable program "
                f"(rollbacks={self._rollbacks_done})",
                indent=2,
            )
        except OSError as exc:
            self._log(f"⚠️  Runnable restore failed: {exc}", indent=2)

    # Backwards-compat shims — both old methods now route to the new policy.
    def _restore_best_if_regressed(self) -> None:
        self._restore_runnable_if_broken()

    def _restore_best_in_episode(self) -> None:
        self._restore_runnable_if_broken()

    def _rollback_if_broken(self) -> None:
        self._restore_runnable_if_broken()

    # ─────────────────────────────────────────────────────────────────────
    # METRIC HOOKS
    # ─────────────────────────────────────────────────────────────────────

    def is_successful(self) -> Optional[bool]:
        if self._last_terminal_score is None:
            return None
        return self._last_terminal_score > 0.0

    def get_terminal_score(self) -> Optional[float]:
        return self._last_terminal_score

    def get_last_final_reward(self) -> float:
        return self._last_final_reward

    def get_final_answer_author(self) -> Optional[str]:
        return self._final_answer_author

    def _read_program_snapshot(self, name: str) -> Optional[str]:
        """Best-effort read of a workspace file. Returns None if missing."""
        if self.workspace_dir is None:
            return None
        target = self.workspace_dir / name
        if not target.is_file():
            return None
        try:
            return target.read_text(encoding="utf-8")
        except OSError:
            return None

    def build_episode_metrics(self) -> Dict[str, Any]:
        raw = self._last_verifier_result
        # Aggregate wakeup events by role: how often each role was judged vs
        # how often it actually woke up.  Ratio tells us whether a role is
        # being invoked too liberally or is atrophying.
        wakeup_by_role: Dict[str, Dict[str, int]] = {}
        for ev in self._wakeup_log:
            slot = wakeup_by_role.setdefault(ev["role"] or "unknown", {"judged": 0, "woke": 0})
            slot["judged"] += 1
            if ev["woke_up"]:
                slot["woke"] += 1

        # Per-episode action log: full <thought>/<code> text from each acting
        # agent, in step order, so the snapshot is reproducible offline.
        action_history_dump = [
            {
                "step": step,
                "author": data.get("author", ""),
                "role": data.get("role", ""),
                "is_code_action": bool(data.get("is_code_action")),
                "is_final": bool(data.get("is_final")),
                "text": data.get("text", ""),
            }
            for step, data in sorted(self.action_history.items())
        ]

        # Program snapshots at episode end. ``initial_program.py`` is the
        # currently-deployed program (may have been rolled back to best by
        # `_restore_best_in_episode` mid-episode, so it equals the snapshot).
        # ``.best_program_snapshot.py`` is the cumulative best.
        program_files = {
            self._ROLLBACK_PROGRAM_NAME: self._read_program_snapshot(
                self._ROLLBACK_PROGRAM_NAME
            ),
            self._ROLLBACK_SNAPSHOT_NAME: self._read_program_snapshot(
                self._ROLLBACK_SNAPSHOT_NAME
            ),
            self._BEST_SNAPSHOT_NAME: self._read_program_snapshot(
                self._BEST_SNAPSHOT_NAME
            ),
        }

        return {
            "terminal_score": self._last_terminal_score,
            "task_completed": self.is_successful(),
            "final_output": self.final_answer,
            "final_output_author": self._final_answer_author,
            "final_output_reward": self._last_final_reward,
            "verifier_raw_score": raw.raw_score if raw else None,
            "verifier_reason": raw.reason if raw else None,
            "verifier_subscores": raw.subscores if raw else [],
            "last_checkpoint_score": self._last_checkpoint_score,
            "best_score_ever": (
                None if self._best_score_ever == float("-inf") else self._best_score_ever
            ),
            "rollbacks_done": self._rollbacks_done,
            "wakeup_by_role": wakeup_by_role,
            "wakeup_events_total": len(self._wakeup_log),
            "wakeup_events": list(self._wakeup_log),
            "action_history": action_history_dump,
            "program_files": program_files,
        }

    def get_state_description(self) -> str:
        if self.terminated:
            status = "success" if self.is_successful() else "terminated"
            return f"CloudcastEnv({self.name}, score={self._last_terminal_score}, {status})"
        return f"CloudcastEnv({self.name}, step={self.step_count}, in_progress)"

    def get_task_description(self) -> str:
        return self.instruction[:2048]

    def get_correct_answer(self) -> str:
        return ""


def _final_answer_sentinel(*_args, **_kwargs):
    """Placeholder — smolagents detects the `final_answer` key and wraps it
    to raise ``FinalAnswerException`` at call-time; this body is never run."""
    return None


def _extract_score(reward_data: Dict[str, Any]) -> Optional[float]:
    """Pull a numeric score out of a compute_reward.py payload."""
    for key in ("score", "reward", "raw_score"):
        val = reward_data.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _normalize_score(raw: Optional[float], reward_data: Dict[str, Any]) -> Optional[float]:
    """Cap arbitrary verifier scores at 1.0; allow negatives through.

    The lower bound used to be 0, which absorbed every regression-from-baseline
    into a flat 0 and killed the checkpoint-delta signal for tasks whose seed
    program is non-trivial (e.g. cloudcast: seed cost 1035 → first attempt
    1100 and 1044 both flatten to 0, delta = 0, no learning signal). Letting
    negative scores through keeps the delta in env._record_checkpoint_score
    informative across the entire trajectory. Tasks whose raw score is
    naturally in [0, 1] (git-to-zig's pass-rate) are unaffected.
    """
    if raw is None:
        return None
    direction = reward_data.get("metric_direction") or reward_data.get("metric_family")
    if direction == "lower_is_better":
        return min(1.0, 1.0 / (1.0 + max(raw, 0.0)))
    return min(1.0, raw)
