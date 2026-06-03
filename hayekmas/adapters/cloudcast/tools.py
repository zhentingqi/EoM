"""Workspace-manipulation tools exposed to cloudcast agents.

These tools operate on a local workspace directory (``env.workspace_dir``)
and are executed inside the smolagents AST interpreter. Each tool is a
thin wrapper that closes over env state; ``CloudcastEnv._make_tool_functions``
materializes the per-step tool dict.

Security note: this module intentionally shells out to user-chosen
commands via ``shell`` — the agent runs untrusted code.  Isolation is
delegated to the host (run under Docker / an ephemeral VM in practice).
The workspace path is scoped to a per-episode temp dir; all relative
paths are resolved against it and absolute paths are rejected.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional


MAX_READ_BYTES = 256 * 1024
MAX_SHELL_OUTPUT_CHARS = 8 * 1024


# Tokens in a shell command that persist changes to the workspace.  The
# shell tool is meant for inspection (ls, grep, find, cat) and for running
# the build/test toolchain (`zig build`, invoking the produced binary).
# Content writes must go through `write_file` so role separation between
# Builder/Reader (shell-only) and Implementer (write_file-only) is actually
# enforceable, not just prompt-level.
_SHELL_WRITE_TOKENS = (
    " > ",
    " >> ",
    "| tee ",
    " tee ",
    "dd of=",
    "sed -i",
    " cp ",
    " mv ",
    " rm -rf ",
    " rm -fr ",
)


def _contains_write_op(cmd: str) -> bool:
    """Return True when *cmd* looks like it writes files via shell."""
    padded = f" {cmd} "
    return any(tok in padded for tok in _SHELL_WRITE_TOKENS)


def _resolve_in_workspace(workspace_dir: Path, path: str) -> Path:
    """Resolve *path* against workspace_dir, refusing escapes."""
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    target = (workspace_dir / path).resolve()
    try:
        target.relative_to(workspace_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Path {path!r} escapes the workspace at {workspace_dir}"
        ) from exc
    return target


def make_write_file(workspace_dir: Path) -> Callable[..., str]:
    def write_file(path: str, content: str) -> str:
        target = _resolve_in_workspace(workspace_dir, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path}"

    return write_file


def make_read_file(workspace_dir: Path) -> Callable[..., str]:
    def read_file(path: str, max_bytes: int = MAX_READ_BYTES) -> str:
        target = _resolve_in_workspace(workspace_dir, path)
        if not target.is_file():
            raise FileNotFoundError(f"{path} not found in workspace")
        data = target.read_bytes()[:max_bytes]
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")

    return read_file


def make_shell(
    workspace_dir: Path,
    default_timeout: int,
) -> Callable[..., str]:
    def shell(cmd: str, timeout: Optional[int] = None) -> str:
        if _contains_write_op(cmd):
            return (
                "[blocked] shell cannot create or modify files in this role. "
                "Use write_file(path, content) to persist changes. "
                "Inspection commands (ls, grep, find, cat) and build/test "
                "commands (zig build, invoking binaries) are allowed."
            )
        use_timeout = timeout if timeout is not None else default_timeout
        completed = subprocess.run(
            cmd,
            shell=True,
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            timeout=use_timeout,
        )
        stdout = (completed.stdout or "")[-MAX_SHELL_OUTPUT_CHARS:]
        stderr = (completed.stderr or "")[-MAX_SHELL_OUTPUT_CHARS:]
        return (
            f"[exit={completed.returncode}]\n"
            f"--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{stderr}"
        )

    return shell


def make_request_eval(
    eval_fn: Callable[[], Dict[str, Any]],
    on_checkpoint: Optional[Callable[[float], None]] = None,
) -> Callable[..., str]:
    """Wrap the verifier invocation so the agent can trigger a scoring run.

    When ``on_checkpoint`` is supplied, it is invoked with the normalized
    score on every successful verifier call.  The env uses this to convert
    the score into a stepwise delta reward.
    """

    def request_eval() -> str:
        result = eval_fn()
        score = result.get("score")
        reason = result.get("reason") or ""
        if score is None:
            return f"verifier unavailable: {reason}"
        if on_checkpoint is not None:
            on_checkpoint(float(score))
        return f"score={score:.4f}  reason={reason}"

    return request_eval


def make_restore_best_snapshot(
    workspace_dir: Path,
    program_name: str,
    snapshot_name: str,
    get_best_score: Callable[[], float],
) -> Callable[..., str]:
    """One-shot revert tool: copy the best snapshot over the program file.

    Cheaper than asking the agent to read 50KB and write it back. Returns
    the best score for confirmation. No-op (with explanatory error) when no
    snapshot has been saved yet.
    """
    import shutil as _shutil

    def restore_best_snapshot() -> str:
        snapshot = workspace_dir / snapshot_name
        program = workspace_dir / program_name
        if not snapshot.is_file():
            return (
                f"no snapshot yet — request_eval() must succeed at least once "
                f"with a clean program before {snapshot_name} exists"
            )
        try:
            _shutil.copy2(snapshot, program)
        except OSError as exc:
            return f"restore failed: {exc}"
        best = get_best_score()
        return (
            f"restored {program_name} from best snapshot "
            f"(best_score_ever={best:+.4f}). Call request_eval() to confirm "
            f"or final_answer() to submit at this score."
        )

    return restore_best_snapshot
