"""Cloudcast adapter runtime.

Multi-agent v2: a specialized roster (Planner / Reader / Implementer /
Builder / Evaluator / Finalizer) shares one workspace per task. Each
episode is a Hayek auction window; ``request_eval()`` calls inside an
episode produce stepwise delta rewards so training has dense signal even
on 20-hour tasks.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hayekmas.adapters.cloudcast.agent import (
    CLOUDCAST_AGENT_CLASSES,
    CloudcastAgent,
)
from hayekmas.adapters.cloudcast.env import CloudcastEnv
from hayekmas.adapters.cloudcast.task import CloudcastTask, load_cloudcast_task
from hayekmas.base.agent import BaseAgent
from hayekmas.base.config import DEFAULT_HAYEK_CONFIG, HayekConfig
from hayekmas.base.mas import HayekMAS
from hayekmas.utils.llm import LLMClient, LLMConfig, get_llm_client
from hayekmas.utils.logger import logger


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CURVE_SCRIPT = _REPO_ROOT / "scripts" / "cloudcast_curve.py"


def _maybe_render_curve_html(out_path: Path, task_id: str) -> None:
    """Re-render the cloudcast trajectory HTML next to the run JSON.

    No-op for non-cloudcast tasks (the curve script is cloudcast-specific:
    it parses cost from `verifier_reason` text and overlays a target line
    at cost=650). Failures are swallowed — broken plotly imports must not
    take down a long-running training job.
    """
    if "cloudcast" not in (task_id or "").lower():
        return
    if not _CURVE_SCRIPT.is_file():
        return
    try:
        import importlib.util as _ilu

        spec = _ilu.spec_from_file_location("_cloudcast_curve", _CURVE_SCRIPT)
        if spec is None or spec.loader is None:
            return
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)

        episodes, _meta = mod.load_episodes(out_path)
        if not episodes:
            return
        label = out_path.stem
        fig, _summary = mod.build_figure(
            [(label, out_path, episodes)], mod.DEFAULT_TARGET_COST
        )
        html_path = out_path.with_suffix(".html")
        mod.write_html(fig, html_path)
        logger.log(f"   📈 Re-rendered {html_path.name}", indent=1)
    except Exception as exc:  # noqa: BLE001
        # Don't kill training over a plotting failure — log and continue.
        logger.log(f"   ⚠️  HTML re-render skipped: {type(exc).__name__}: {exc}", indent=1)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CloudcastRunConfig:
    task_dir: str
    max_steps: int = 20
    code_execution_timeout: int = 600
    shell_timeout: int = 300
    verifier_timeout: int = 1800
    workspace_parent: Optional[str] = None
    keep_workspace: bool = False
    preserve_workspace_across_episodes: bool = True
    num_episodes: int = 1
    output: str = "outputs/cloudcast.json"


@dataclass
class CloudcastRuntimeConfig:
    llm: LLMConfig
    domain: str = "cloudcast"
    mode: str = "eval"
    profile: str = ""
    run: CloudcastRunConfig = field(default_factory=lambda: CloudcastRunConfig(task_dir=""))
    mas: HayekConfig = field(default_factory=lambda: deepcopy(DEFAULT_HAYEK_CONFIG))


_META_KEYS = frozenset({"adapter_config", "profile", "domain"})


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_runtime_config(raw: Dict[str, Any]) -> CloudcastRuntimeConfig:
    if raw.get("domain") != "cloudcast":
        raise ValueError(f"Unsupported domain for cloudcast runtime: {raw.get('domain')}")
    profile = (raw.get("profile") or "").strip()
    if not profile:
        raise ValueError("Top-level config must include a non-empty `profile`.")

    adapter_path = raw.get("adapter_config")
    if not adapter_path:
        raise ValueError("Top-level config must include `adapter_config`.")
    with open(adapter_path, "r", encoding="utf-8") as f:
        section = json.load(f)
    if not isinstance(section, dict):
        raise ValueError(f"Adapter config must be a JSON object: {adapter_path}")

    overlay = {k: v for k, v in raw.items() if k not in _META_KEYS}
    merged = _deep_merge(section, overlay)

    model_raw = merged.get("model") or {}
    llm_api = model_raw.get("api")
    llm_name = model_raw.get("name")
    if not llm_api or not llm_name:
        raise ValueError("`model.api` and `model.name` are required.")
    extras = {
        k: v
        for k, v in model_raw.items()
        if k not in {"api", "name", "api_key", "api_base"}
    }

    run_raw = merged.get("run") or {}
    task_dir = run_raw.get("task_dir")
    if not task_dir:
        raise ValueError("`run.task_dir` must point at a cloudcast task directory.")

    return CloudcastRuntimeConfig(
        llm=LLMConfig(
            api=llm_api,
            name=llm_name,
            api_key=model_raw.get("api_key"),
            api_base=model_raw.get("api_base"),
            extra_kwargs=extras,
        ),
        mode=merged.get("mode", "eval"),
        profile=profile,
        run=CloudcastRunConfig(
            task_dir=task_dir,
            max_steps=run_raw.get("max_steps", 20),
            code_execution_timeout=run_raw.get("code_execution_timeout", 600),
            shell_timeout=run_raw.get("shell_timeout", 300),
            verifier_timeout=run_raw.get("verifier_timeout", 1800),
            workspace_parent=run_raw.get("workspace_parent"),
            keep_workspace=run_raw.get("keep_workspace", False),
            preserve_workspace_across_episodes=run_raw.get(
                "preserve_workspace_across_episodes", True
            ),
            num_episodes=run_raw.get("num_episodes", 1),
            output=run_raw.get("output", "outputs/cloudcast.json"),
        ),
        mas=HayekConfig.from_dict(merged.get("mas") or {}),
    )


# ═══════════════════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def _build_llm_client(cfg: CloudcastRuntimeConfig) -> LLMClient:
    logger.log("=" * 70)
    logger.log("🤖 SETTING UP LLM BACKEND")
    logger.log("=" * 70)
    logger.log(f"   API: {cfg.llm.api}")
    logger.log(f"   Model: {cfg.llm.name}")
    kwargs = dict(cfg.llm.extra_kwargs)
    if cfg.llm.api_base:
        kwargs["api_base"] = cfg.llm.api_base
    kwargs["api_key"] = cfg.llm.api_key
    return get_llm_client(cfg.llm.api, model=cfg.llm.name, **kwargs)


def _create_env(task: CloudcastTask, cfg: CloudcastRuntimeConfig) -> CloudcastEnv:
    return CloudcastEnv(
        task=task,
        reward_config=cfg.mas.reward,
        max_steps=cfg.mas.engine.max_steps_per_episode or cfg.run.max_steps,
        code_execution_timeout=cfg.run.code_execution_timeout,
        shell_timeout=cfg.run.shell_timeout,
        verifier_timeout=cfg.run.verifier_timeout,
        workspace_parent=Path(cfg.run.workspace_parent) if cfg.run.workspace_parent else None,
        keep_workspace=cfg.run.keep_workspace,
        preserve_workspace_across_episodes=cfg.run.preserve_workspace_across_episodes,
    )


def _create_agents(llm_client: LLMClient) -> List[CloudcastAgent]:
    backbone = llm_client.as_callable()
    return [cls(backbone_llm=backbone, logger=logger) for cls in CLOUDCAST_AGENT_CLASSES]


def _make_good_birth_factory(
    llm_client: LLMClient, initial_wealth: float
) -> Callable[[BaseAgent], BaseAgent]:
    backbone = llm_client.as_callable()

    def factory(parent: BaseAgent) -> BaseAgent:
        return parent.__class__.birth_good_agent(
            parent=parent,
            backbone_llm=backbone,
            initial_wealth=initial_wealth,
        )

    return factory


def _make_bad_birth_factory(
    llm_client: LLMClient, initial_wealth: float
) -> Callable[..., BaseAgent]:
    backbone = llm_client.as_callable()

    def factory(source_agent: BaseAgent, **kwargs) -> BaseAgent:
        return source_agent.__class__.birth_bad_agent(
            source_agent=source_agent,
            backbone_llm=backbone,
            initial_wealth=initial_wealth,
            task_description=kwargs.get("task_description", ""),
            correct_answer=kwargs.get("correct_answer", ""),
            failure_trace=kwargs.get("failure_trace", ""),
        )

    return factory


def run(cfg: CloudcastRuntimeConfig) -> None:
    logger.configure(verbose=True, log_dir="logs", profile=cfg.profile)
    logger.print_mode_banner(cfg.mode.upper(), f"CLOUDCAST {cfg.mode.upper()}")

    llm_client = _build_llm_client(cfg)
    task = load_cloudcast_task(cfg.run.task_dir)
    logger.log(f"\n📂 Loaded task: {task.id}  from {task.task_dir}")

    mas_config = deepcopy(cfg.mas)
    # Terminal-step candidates: Evaluator (still useful for a last measurement)
    # and Finalizer (submits). Restricting the last step to these prevents an
    # Implementer from wasting the final step on a half-applied write.
    mas_config.terminal.candidate_agent_tags = ("terminal",)
    mas = HayekMAS(config=mas_config)

    initial_wealth = mas_config.engine.initial_wealth
    for agent in _create_agents(llm_client):
        agent.initialize(initial_wealth=initial_wealth)
        mas.population.add_agent(agent)
        logger.log(f"   ✓ agent {agent.name}  (role={agent.role})")
    mas._initial_agents = list(mas.population.get_all())

    mas.set_agent_factory(
        birth_good_agent=_make_good_birth_factory(llm_client, initial_wealth),
        birth_bad_agent=_make_bad_birth_factory(llm_client, initial_wealth),
    )

    if cfg.mode == "train":
        mas.train()
    else:
        mas.eval()

    # One CloudcastEnv per task. When `preserve_workspace_across_episodes` is
    # True, the workspace directory persists across episodes so the agent
    # team can build on prior work; only the per-episode step counter /
    # action history / checkpoint-fire flag get reset by env.initialize().
    env = _create_env(task, cfg)
    episode_metrics: List[Dict[str, Any]] = []
    out_path = Path(cfg.run.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_progress() -> None:
        # Incremental write so external watchers (e.g. cloudcast_curve.py
        # --watch) can re-render a live trajectory while training is still
        # running. Final write at the end re-emits population_final.
        out_path.write_text(
            json.dumps(
                {"task": task.id, "episodes": episode_metrics, "in_progress": True},
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        _maybe_render_curve_html(out_path, task.id)

    def _snapshot_population() -> List[Dict[str, Any]]:
        """Serialize every alive agent at episode end.
        Captures wealth, status, frozen + trainable prompts, lineage so a
        future reader can reconstruct the full population state per episode.
        """
        out: List[Dict[str, Any]] = []
        for agent in mas.population.get_all():
            try:
                out.append(agent.serialize())
            except Exception as exc:  # noqa: BLE001
                out.append({
                    "name": getattr(agent, "name", "?"),
                    "type": agent.__class__.__name__,
                    "serialize_error": f"{type(exc).__name__}: {exc}",
                })
        return out

    try:
        for ep in range(cfg.run.num_episodes):
            logger.log(f"\n{'█' * 70}\n📋 Episode {ep + 1}/{cfg.run.num_episodes}\n{'─' * 70}")
            mas.run_one_episode(env)
            metrics = env.build_episode_metrics()
            metrics["population_snapshot"] = _snapshot_population()
            episode_metrics.append({"episode": ep, **metrics})
            score = metrics.get("terminal_score")
            cp_score = metrics.get("last_checkpoint_score")
            logger.log(
                f"   → terminal_score={score}  checkpoint_score={cp_score}  "
                f"success={metrics.get('task_completed')}  "
                f"pop={len(metrics['population_snapshot'])}"
            )
            _write_progress()
            # NOTE: we used to break out of the episode loop whenever
            # env.terminated became True (final_answer was called).  That made
            # sense for a one-shot submission task like git-to-zig, but for
            # evolving tasks where the workspace persists across episodes
            # (preserve_workspace_across_episodes=True), each new episode is
            # another chance for the population to mutate and improve the
            # checked-in code.  env.initialize() at the top of the next
            # episode resets `terminated`, so we just keep going.
    finally:
        env.cleanup()

    out_path.write_text(
        json.dumps(
            {
                "task": task.id,
                "episodes": episode_metrics,
                "population_final": [
                    {
                        "name": a.name,
                        "role": a.role,
                        "wealth": a.wealth,
                        "type": a.__class__.__name__,
                    }
                    for a in mas.population.get_all()
                ],
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    logger.log(f"\n💾 Wrote {out_path}")
    _maybe_render_curve_html(out_path, task.id)
    logger.close()


def main(raw_config: Dict[str, Any]) -> None:
    run(load_runtime_config(raw_config))
