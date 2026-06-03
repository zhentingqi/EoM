"""Cloudcast task schema and loader.

Loads a single cloudcast task from its on-disk directory layout:

    <task_dir>/
        task.toml         # metadata + resource limits
        instruction.md    # problem statement
        environment/      # workspace template (copied to workspace per episode)
        tests/            # verifier harness (compute_reward.py + test.sh)

Only data needed by the HayekMAS engine is captured. Docker / Harbor-specific
keys in task.toml are preserved in ``raw_toml`` but not interpreted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class CloudcastTask:
    """A single cloudcast task resolved from disk."""

    id: str
    task_dir: Path
    instruction: str
    workspace_template: Optional[Path]
    verifier_script: Optional[Path]
    verifier_test_sh: Optional[Path]
    raw_toml: Dict[str, Any] = field(default_factory=dict)

    @property
    def question(self) -> str:
        """Back-compat alias used by generic pipeline helpers."""
        return self.instruction

    @property
    def expected_answer(self) -> str:
        return ""

    def __repr__(self) -> str:
        return f"CloudcastTask(id={self.id!r}, dir={self.task_dir.name!r})"


def load_cloudcast_task(task_dir: str | Path) -> CloudcastTask:
    """Load a cloudcast task from *task_dir*.

    The directory must contain ``task.toml`` and ``instruction.md``.  The
    ``environment/`` subdirectory is used as the workspace template if
    present.  The verifier entry point is ``tests/compute_reward.py``.
    """
    path = Path(task_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Task directory not found: {path}")

    toml_path = path / "task.toml"
    instruction_path = path / "instruction.md"
    if not toml_path.is_file():
        raise FileNotFoundError(f"task.toml missing in {path}")
    if not instruction_path.is_file():
        raise FileNotFoundError(f"instruction.md missing in {path}")

    with open(toml_path, "rb") as f:
        raw_toml = tomllib.load(f)

    instruction = instruction_path.read_text(encoding="utf-8")

    workspace_template = path / "environment"
    if not workspace_template.is_dir():
        workspace_template = None

    verifier_script = path / "tests" / "compute_reward.py"
    if not verifier_script.is_file():
        verifier_script = None

    verifier_test_sh = path / "tests" / "test.sh"
    if not verifier_test_sh.is_file():
        verifier_test_sh = None

    return CloudcastTask(
        id=path.name,
        task_dir=path,
        instruction=instruction,
        workspace_template=workspace_template,
        verifier_script=verifier_script,
        verifier_test_sh=verifier_test_sh,
        raw_toml=raw_toml,
    )


def load_cloudcast_tasks(task_dirs: List[str | Path]) -> List[CloudcastTask]:
    """Load multiple tasks from a list of directories."""
    return [load_cloudcast_task(d) for d in task_dirs]
