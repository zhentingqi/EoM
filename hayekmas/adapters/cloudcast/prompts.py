"""Prompt templates for the cloudcast adapter."""

from __future__ import annotations

from typing import Optional

from hayekmas.base.prompts import WAKEUP_PROMPT


NEXT_STEP_PROMPT_TEMPLATE = """\
{agent_system_prompt}

## Task instruction
{instruction}

## Workspace
You edit files inside a scratch workspace seeded from the task's initial state.
**Use relative paths only** — the workspace root is the current working directory (`.`).
Absolute paths are refused.  If the task instruction mentions absolute paths
(e.g. `/app/<some-dir>/`), translate them to the matching relative path inside
the workspace.  To see what is at the workspace root, call `shell("ls -la")`.

## Progress so far
{state}

## Tools available to you this turn
{tool_signatures}

## Output format
Emit one <thought>...</thought> block describing what you will do next, then a
single <code>...</code> block that executes **Python** code calling the tools
listed above as regular Python functions.  If your role has no tools this
turn (e.g. pure planning), emit only the <thought> block and omit <code>.

## Sandbox restrictions on the <code> block
The <code> block runs inside a tiny Python sandbox.  **Almost no imports are
allowed** — only stdlib basics like `math`, `re`, `random`, `collections`,
`itertools`, `statistics`.  In particular these all FAIL:

    import networkx, import pandas, import numpy           # ❌ third-party
    from typing import Dict, List                          # ❌ typing
    from __future__ import annotations                     # ❌ future
    from pathlib import Path                               # ❌ pathlib
    import sys, import os, import importlib, import json   # ❌ blocked

`open()`, `exec()`, `eval()`, `__import__()` are likewise blocked.

These restrictions apply ONLY to your <code> block.  The actual program
file (`initial_program.py`) is run by the verifier with full system Python
where `networkx`, `pandas`, `typing`, etc. are all available.  So:

- to **read** a file, call `read_file("initial_program.py")` — never `open(...)`.
- to **write** a file, call `write_file("initial_program.py", new_content)`
  with the FULL new content — never `open(..., "w")`.
- to **test** the program, call `request_eval()` — never `import networkx;
  exec(...)`.  The sandbox cannot run the program itself.

### Correct <code> examples
    <code>
    shell("ls -la")
    </code>

    <code>
    src = read_file("initial_program.py")
    print(src[:800])
    </code>

    <code>
    write_file("initial_program.py", "import networkx as nx\\n\\ndef search_algorithm(...): ...\\n")
    </code>

    <code>
    request_eval()
    </code>

    <code>
    final_answer("multi-path broadcast; cost dropped from 1035 to 720")
    </code>

### Common mistakes — DO NOT do these
    <code>
    ls -la                                  # ❌ bare shell syntax; this is Python, not bash
    </code>

    <code>
    {{"cmd": "ls -la", "timeout": 120}}     # ❌ dict literal; tools are functions, call them
    </code>

    <code>
    import networkx as nx                   # ❌ blocked; only the program file may import it
    G = nx.DiGraph()
    </code>

    <code>
    open("initial_program.py", "w").write(content)   # ❌ open() is blocked; use write_file()
    </code>

Keep each step focused; the verifier is expensive, so `request_eval()` should
only be called when the workspace is expected to score meaningfully higher
than before.  When the submission is ready, call `final_answer("short summary")`;
this finishes the episode and triggers a final verifier run.  Do NOT call
`final_answer()` if the latest verifier score has not improved over the seed —
just stop emitting code so the next role can act, and let a future episode
make progress.

Current step: {step_count}/{max_steps}
"""


ALL_TOOL_SIGNATURES = {
    "write_file": "- write_file(path: str, content: str) -> str  # write file in workspace",
    "read_file": "- read_file(path: str, max_bytes: int = 262144) -> str  # read file in workspace",
    "shell": "- shell(cmd: str, timeout: int | None = None) -> str  # run shell command in workspace",
    "request_eval": "- request_eval() -> str  # trigger the verifier; delta score is paid to you",
    "final_answer": "- final_answer(summary: str) -> None  # finish the episode; triggers final verifier run",
}


def format_tool_signatures(allowed: Optional[set[str]]) -> str:
    """Render tool signatures filtered by this agent's allowlist."""
    if allowed is None:
        selected = list(ALL_TOOL_SIGNATURES.values())
    else:
        selected = [
            sig for name, sig in ALL_TOOL_SIGNATURES.items() if name in allowed
        ]
    if not selected:
        return "(this agent has no tools; emit only a <thought> block)"
    return "\n".join(selected)


def format_next_step_prompt(
    *,
    agent_system_prompt: str,
    instruction: str,
    state: str,
    tool_signatures: str,
    step_count: int,
    max_steps: int,
) -> str:
    return NEXT_STEP_PROMPT_TEMPLATE.format(
        agent_system_prompt=agent_system_prompt,
        instruction=instruction,
        state=state if state.strip() else "(no actions yet)",
        tool_signatures=tool_signatures,
        step_count=step_count,
        max_steps=max_steps,
    )


def format_wakeup_prompt(agent_system_prompt: str, state: str) -> str:
    """Wakeup judge prompt reusing the base template."""
    return (
        WAKEUP_PROMPT
        .replace("<<<agent_system_prompt>>>", agent_system_prompt)
        .replace("<<<state>>>", state)
    )


# Legacy alias kept for backwards compatibility.
TOOL_SIGNATURES = format_tool_signatures(None)
