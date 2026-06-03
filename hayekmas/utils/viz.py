"""Post-hoc visualization of a HayekMAS training run.

Reads the JSONL metrics files written by ``Trainer._write_task_outputs`` under
``outputs/<run_dir>/`` and renders a self-contained interactive HTML dashboard
with a task slider. No engine changes — pure reader.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover - gated at import time
    raise ImportError(
        "plotly is required for visualization. Install with: pip install -e '.[viz]'"
    ) from exc


_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
    "#bcbd22", "#7f7f7f",
]

_SYMBOL_BY_SPAWN = {
    "initial": "circle",
    "good_birth": "diamond",
    "bankruptcy_birth": "triangle-up",
}


def _role_color(role: str) -> str:
    h = int(hashlib.md5(role.encode()).hexdigest(), 16)
    return _PALETTE[h % len(_PALETTE)]


def _role_from_class(cls_name: str) -> str:
    s = cls_name or ""
    for suffix in ("FinanceAgent", "MathAgent", "Agent"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return (s or cls_name).lower()


@dataclass
class AgentTrajectory:
    id: int
    name: str
    cls: str
    role: str
    spawn_method: str
    father_agent_id: int | None
    parent_agent_id: int | None
    parent_agent_name: str | None
    root_ancestor_class: str
    birth_task_idx: int
    death_task_idx: int | None = None
    wealth: dict[int, float] = field(default_factory=dict)
    capability_score: dict[int, float] = field(default_factory=dict)
    status: dict[int, str] = field(default_factory=dict)
    tasks_lived: dict[int, int] = field(default_factory=dict)
    trainable_prompt: str = ""


@dataclass
class TaskOutcome:
    task_idx: int
    task_id: str
    success: bool
    terminal_score: float
    bankruptcies: int
    wealth_highest: float
    wealth_average: float
    wealth_lowest: float
    final_output: str
    judge_reason: str
    path_agent_names: list[str]
    population_size: int


@dataclass
class RunData:
    run_dir: Path
    agents: dict[int, AgentTrajectory]
    tasks: list[TaskOutcome]
    roles: list[str]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_run(run_dir: Path | str) -> RunData:
    run_dir = Path(run_dir)
    pop_path = run_dir / "population_metrics.jsonl"
    task_path = run_dir / "training_metrics.jsonl"
    if not pop_path.exists() or not task_path.exists():
        raise FileNotFoundError(
            f"Run directory {run_dir} is missing population_metrics.jsonl or training_metrics.jsonl"
        )
    pop_rows = _read_jsonl(pop_path)
    task_rows = _read_jsonl(task_path)
    n = min(len(pop_rows), len(task_rows))
    if n == 0:
        raise ValueError(f"Run {run_dir} has no tasks")

    def _f(row: dict, key: str, default: float = 0.0) -> float:
        v = row.get(key)
        return float(v) if v is not None else default

    def _i(row: dict, key: str, default: int = 0) -> int:
        v = row.get(key)
        return int(v) if v is not None else default

    def _s(row: dict, key: str, default: str = "") -> str:
        v = row.get(key)
        return str(v) if v is not None else default

    tasks: list[TaskOutcome] = []
    for i, row in enumerate(task_rows[:n]):
        tasks.append(
            TaskOutcome(
                task_idx=i,
                task_id=_s(row, "task_id", f"task-{i}"),
                success=bool(row.get("success", False)),
                terminal_score=_f(row, "terminal_score"),
                bankruptcies=_i(row, "bankruptcies"),
                wealth_highest=_f(row, "wealth_highest"),
                wealth_average=_f(row, "wealth_average"),
                wealth_lowest=_f(row, "wealth_lowest"),
                final_output=_s(row, "final_output"),
                judge_reason=_s(row, "judge_reason"),
                path_agent_names=list(row.get("path_agent_names") or []),
                population_size=_i(row, "population_size"),
            )
        )

    agents: dict[int, AgentTrajectory] = {}
    last_seen: dict[int, int] = {}
    roles: list[str] = []
    for t, pop_row in enumerate(pop_rows[:n]):
        for a in pop_row.get("agents", []) or []:
            aid = int(a["id"])
            role = _role_from_class(a.get("class", ""))
            if role not in roles:
                roles.append(role)
            if aid not in agents:
                agents[aid] = AgentTrajectory(
                    id=aid,
                    name=str(a.get("name", f"agent_{aid}")),
                    cls=str(a.get("class", "")),
                    role=role,
                    spawn_method=str(a.get("spawn_method", "initial")),
                    father_agent_id=a.get("father_agent_id"),
                    parent_agent_id=a.get("parent_agent_id"),
                    parent_agent_name=a.get("parent_agent_name"),
                    root_ancestor_class=str(a.get("root_ancestor_class", "")),
                    birth_task_idx=t,
                )
            ag = agents[aid]
            ag.wealth[t] = float(a.get("wealth", 0.0))
            ag.capability_score[t] = float(a.get("capability_score", 0.0))
            ag.status[t] = str(a.get("status", "novice"))
            ag.tasks_lived[t] = int(a.get("tasks_lived", 0))
            tp = a.get("trainable_system_prompt")
            if tp:
                ag.trainable_prompt = str(tp)
            last_seen[aid] = t

    last_idx = n - 1
    for aid, ag in agents.items():
        if last_seen[aid] < last_idx:
            ag.death_task_idx = last_seen[aid] + 1

    return RunData(run_dir=run_dir, agents=agents, tasks=tasks, roles=roles)


def _agent_alive(ag: AgentTrajectory, t: int) -> bool:
    return t >= ag.birth_task_idx and (ag.death_task_idx is None or t < ag.death_task_idx)


def _jitter(aid: int, amplitude: float = 0.3) -> float:
    return ((aid * 2654435761) % 1000 / 1000.0 - 0.5) * 2.0 * amplitude


def _node_positions(run: RunData) -> dict[int, tuple[float, float]]:
    role_lane = {r: i for i, r in enumerate(run.roles)}
    return {
        aid: (float(ag.birth_task_idx) + _jitter(aid, 0.15), role_lane[ag.role] + _jitter(aid, 0.28))
        for aid, ag in run.agents.items()
    }


def _frame_tree_nodes(
    run: RunData, t: int, node_ids: list[int], max_wealth: float
) -> dict[str, Any]:
    sizes, colors, opacities, line_widths, hovers, texts = [], [], [], [], [], []
    for i in node_ids:
        ag = run.agents[i]
        if t < ag.birth_task_idx:
            sizes.append(6); colors.append("#eeeeee"); opacities.append(0.0)
            line_widths.append(0); hovers.append(""); texts.append("")
        elif _agent_alive(ag, t):
            w = ag.wealth.get(t, 0.0)
            base = 10.0 + 40.0 * (w / max_wealth if max_wealth > 0 else 0.0)
            sizes.append(base)
            colors.append(_role_color(ag.role))
            opacities.append(0.95)
            matured = ag.status.get(t, "novice") in ("veteran", "tycoon")
            line_widths.append(2 if matured else 1)
            hovers.append(
                f"<b>{ag.name}</b> (id {ag.id})<br>"
                f"role: {ag.role}<br>"
                f"wealth: {w:.3f}<br>"
                f"capability: {ag.capability_score.get(t, 0.0):.3f}<br>"
                f"status: {ag.status.get(t, 'novice')}<br>"
                f"spawn: {ag.spawn_method}<br>"
                f"parent: {ag.parent_agent_name or '—'}<br>"
                f"tasks lived: {ag.tasks_lived.get(t, 0)}"
            )
            texts.append(ag.name)
        else:  # dead
            sizes.append(10); colors.append("#cccccc"); opacities.append(0.35)
            line_widths.append(1)
            death = ag.death_task_idx or 0
            last_w = ag.wealth.get(max(death - 1, ag.birth_task_idx), 0.0)
            hovers.append(
                f"<b>{ag.name}</b> (id {ag.id}) — DIED at task {death}<br>"
                f"final wealth: {last_w:.3f}"
            )
            texts.append("")
    return dict(
        marker=dict(
            size=sizes, color=colors, opacity=opacities,
            line=dict(color="black", width=line_widths),
        ),
        text=texts,
        hovertext=hovers,
    )


def _frame_bubbles(run: RunData, t: int, max_cap: float) -> dict[str, Any]:
    role_idx = {r: i for i, r in enumerate(run.roles)}
    xs, ys, sizes, colors, syms, hovers = [], [], [], [], [], []
    for ag in run.agents.values():
        if not _agent_alive(ag, t):
            continue
        w = ag.wealth.get(t, 0.0)
        cap = ag.capability_score.get(t, 0.0)
        xs.append(role_idx[ag.role] + _jitter(ag.id, 0.32))
        ys.append(w)
        sizes.append(14.0 + 30.0 * (cap / max_cap if max_cap > 0 else 0.0))
        colors.append(_role_color(ag.role))
        syms.append(_SYMBOL_BY_SPAWN.get(ag.spawn_method, "circle"))
        hovers.append(
            f"<b>{ag.name}</b> (id {ag.id})<br>"
            f"wealth: {w:.3f}<br>"
            f"capability: {cap:.3f}<br>"
            f"status: {ag.status.get(t, 'novice')}<br>"
            f"spawn: {ag.spawn_method}"
        )
    return dict(
        x=xs, y=ys,
        marker=dict(
            size=sizes, color=colors, symbol=syms,
            line=dict(color="black", width=1), opacity=0.85,
        ),
        hovertext=hovers,
    )


def build_figure(run: RunData) -> go.Figure:
    n = len(run.tasks)
    if n == 0:
        raise ValueError(f"No tasks in run {run.run_dir}")

    pos = _node_positions(run)
    node_ids = sorted(run.agents.keys())
    max_wealth = max(
        (w for ag in run.agents.values() for w in ag.wealth.values()), default=1.0
    ) or 1.0
    max_cap = max(
        (c for ag in run.agents.values() for c in ag.capability_score.values()), default=1.0
    ) or 1.0

    pop_counts: dict[str, list[int]] = {r: [0] * n for r in run.roles}
    for t in range(n):
        for ag in run.agents.values():
            if _agent_alive(ag, t):
                pop_counts[ag.role][t] += 1
    pop_max = max((sum(pop_counts[r][t] for r in run.roles) for t in range(n)), default=1) + 1

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Lineage tree (x=birth task · y=role lane · size=wealth · shape=spawn)",
            "Wealth by role at current task",
            "Population stacked by role + avg wealth (red = current task)",
            "Task outcomes (green=success, red=fail)",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
        column_widths=[0.58, 0.42],
        row_heights=[0.58, 0.42],
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
    )

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for ag in run.agents.values():
        pid = ag.parent_agent_id
        if pid is None or pid not in run.agents:
            continue
        px, py = pos[pid]
        cx, cy = pos[ag.id]
        edge_x += [px, cx, None]
        edge_y += [py, cy, None]
    fig.add_trace(
        go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(color="#bbbbbb", width=1),
            hoverinfo="skip", showlegend=False, name="lineage",
        ),
        row=1, col=1,
    )  # idx 0

    init_tree = _frame_tree_nodes(run, 0, node_ids, max_wealth)
    fig.add_trace(
        go.Scatter(
            x=[pos[i][0] for i in node_ids],
            y=[pos[i][1] for i in node_ids],
            mode="markers+text",
            textposition="top center",
            textfont=dict(size=9),
            hoverinfo="text",
            showlegend=False,
            name="agents",
            **init_tree,
        ),
        row=1, col=1,
    )  # idx 1
    tree_idx = 1

    init_bubbles = _frame_bubbles(run, 0, max_cap)
    fig.add_trace(
        go.Scatter(
            mode="markers", hoverinfo="text",
            showlegend=False, name="bubbles",
            **init_bubbles,
        ),
        row=1, col=2,
    )  # idx 2
    bubbles_idx = 2

    for r in run.roles:
        fig.add_trace(
            go.Scatter(
                x=list(range(n)), y=pop_counts[r],
                mode="lines", stackgroup="pop",
                name=r,
                line=dict(width=0.5, color=_role_color(r)),
                fillcolor=_role_color(r), opacity=0.55,
                hovertemplate=f"{r}: %{{y}} agents at task %{{x}}<extra></extra>",
            ),
            row=2, col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=list(range(n)),
            y=[task.wealth_average for task in run.tasks],
            mode="lines+markers",
            name="avg wealth",
            line=dict(color="black", width=2, dash="dot"),
            yaxis="y",
            hovertemplate="avg wealth %{y:.3f} @ task %{x}<extra></extra>",
        ),
        row=2, col=1,
    )

    timeline_cursor_idx = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[0, 0], y=[0, pop_max],
            mode="lines",
            line=dict(color="red", width=2),
            hoverinfo="skip", showlegend=False, name="cursor",
        ),
        row=2, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=list(range(n)), y=[1] * n,
            marker=dict(
                color=["#2ca02c" if task.success else "#d62728" for task in run.tasks],
                line=dict(color="white", width=1),
            ),
            hovertext=[
                f"<b>{task.task_id}</b><br>"
                f"success: {task.success}<br>"
                f"score: {task.terminal_score:.3f}<br>"
                f"bankruptcies: {task.bankruptcies}<br>"
                f"path: {', '.join(task.path_agent_names)}<br>"
                f"output: {task.final_output[:120]}<br>"
                f"judge: {task.judge_reason[:200]}"
                for task in run.tasks
            ],
            hoverinfo="text",
            showlegend=False, name="outcomes",
        ),
        row=2, col=2,
    )

    outcome_cursor_idx = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[0], y=[1.18],
            mode="markers",
            marker=dict(symbol="triangle-down", size=18, color="black"),
            hoverinfo="skip", showlegend=False, name="current task",
        ),
        row=2, col=2,
    )

    frames: list[go.Frame] = []
    for t in range(n):
        tree_upd = _frame_tree_nodes(run, t, node_ids, max_wealth)
        bubble_upd = _frame_bubbles(run, t, max_cap)
        frames.append(
            go.Frame(
                name=str(t),
                data=[
                    go.Scatter(**tree_upd),
                    go.Scatter(**bubble_upd),
                    go.Scatter(x=[t, t], y=[0, pop_max]),
                    go.Scatter(x=[t], y=[1.18]),
                ],
                traces=[tree_idx, bubbles_idx, timeline_cursor_idx, outcome_cursor_idx],
            )
        )
    fig.frames = frames

    fig.update_xaxes(title_text="birth task →", row=1, col=1, showgrid=True)
    fig.update_yaxes(
        title_text="role lane",
        tickvals=list(range(len(run.roles))), ticktext=run.roles,
        row=1, col=1, showgrid=True, gridcolor="#f0f0f0",
    )
    fig.update_xaxes(
        title_text="role",
        tickvals=list(range(len(run.roles))), ticktext=run.roles,
        row=1, col=2,
    )
    fig.update_yaxes(title_text="wealth", row=1, col=2, rangemode="tozero")
    fig.update_xaxes(title_text="task idx", row=2, col=1)
    fig.update_yaxes(title_text="count / avg wealth", row=2, col=1)
    fig.update_xaxes(title_text="task idx", row=2, col=2)
    fig.update_yaxes(showticklabels=False, range=[0, 1.45], row=2, col=2)

    fig.update_layout(
        title=dict(
            text=(
                f"HayekMAS · <b>{run.run_dir.name}</b> · "
                f"{n} tasks · {len(run.agents)} agents ever · "
                f"roles: {', '.join(run.roles)}"
            ),
            x=0.02, xanchor="left",
        ),
        height=920,
        hovermode="closest",
        margin=dict(l=60, r=20, t=80, b=120),
        legend=dict(orientation="h", y=-0.28, x=0.02),
        updatemenus=[{
            "type": "buttons", "showactive": False,
            "x": 0.02, "y": -0.15, "xanchor": "left",
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": 600, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 150},
                    }],
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                },
            ],
        }],
        sliders=[{
            "active": 0, "y": -0.18, "x": 0.13, "len": 0.82,
            "pad": {"t": 10, "b": 10},
            "currentvalue": {"prefix": "task ", "font": {"size": 13}},
            "steps": [
                {
                    "label": f"{t}",
                    "method": "animate",
                    "args": [[str(t)], {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                }
                for t in range(n)
            ],
        }],
    )
    return fig


def write_html(fig: go.Figure, out_path: Path | str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        out_path,
        include_plotlyjs="inline",
        full_html=True,
        auto_play=False,
    )
    return out_path
