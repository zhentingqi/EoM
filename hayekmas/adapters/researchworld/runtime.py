"""
Researchworld adapter runtime.

Orchestration point for the researchworld adapter: config parsing, object
construction, and the ``Trainer`` / ``Evaluator`` entrypoints. The engine
itself is domain-agnostic; this file plugs ``ResearchAgent`` + ``ResearchEnv``
into it.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from hayekmas.adapters.researchworld.agent import (
    RESEARCH_AGENT_CLASSES,
    ResearchAgent,
)
from hayekmas.adapters.researchworld.env import (
    ResearchEnv,
    ResearchTask,
    get_default_research_test_path,
    get_default_research_train_path,
    load_research_tasks,
    score_answer_against_rubric,
)
from hayekmas.base.agent import BaseAgent
from hayekmas.base.config import DEFAULT_HAYEK_CONFIG, HayekConfig, RewardConfig
from hayekmas.base.mas import HayekMAS
from hayekmas.base.pipeline import Evaluator, Trainer
from hayekmas.utils.llm import LLMClient, LLMConfig, get_llm_client
from hayekmas.utils.logger import logger


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ResearchSplitConfig:
    files: Optional[List[str]] = None
    max_tasks: Optional[int] = None


@dataclass
class ResearchDataConfig:
    train: ResearchSplitConfig = field(default_factory=ResearchSplitConfig)
    test: ResearchSplitConfig = field(default_factory=ResearchSplitConfig)


@dataclass
class ResearchJudgeConfig:
    enabled: bool = True
    threshold: float = 0.5
    verbose: bool = False


@dataclass
class ResearchRunConfig:
    num_epochs: int = 2
    verbose: bool = True
    output: str = "checkpoints/hayek_research.json"
    checkpoint: Optional[str] = None
    checkpoint_steps: Optional[str] = None
    resume_run_dir: Optional[str] = None
    resume_from_task: Optional[int] = None
    max_eval_workers: int = 10


@dataclass
class ResearchRuntimeConfig:
    llm: LLMConfig
    domain: str = "researchworld"
    mode: str = "train"
    profile: str = ""
    data: ResearchDataConfig = field(default_factory=ResearchDataConfig)
    judge: ResearchJudgeConfig = field(default_factory=ResearchJudgeConfig)
    run: ResearchRunConfig = field(default_factory=ResearchRunConfig)
    mas: HayekConfig = field(default_factory=lambda: deepcopy(DEFAULT_HAYEK_CONFIG))


_CONFIG_META_KEYS = frozenset({"adapter_config", "profile", "domain"})


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def configure_research_hayek_config(config: HayekConfig) -> HayekConfig:
    """Apply researchworld-specific engine defaults."""
    config.terminal.candidate_agent_tags = ("terminal",)
    return config


def load_research_runtime_config(raw: Dict[str, Any]) -> ResearchRuntimeConfig:
    """Parse a researchworld runtime config from merged global + adapter JSON."""
    if raw.get("domain") != "researchworld":
        raise ValueError(
            f"Unsupported domain for research runtime: {raw.get('domain')}"
        )

    profile = raw.get("profile")
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("Top-level config must include a non-empty `profile`.")
    profile = profile.strip()

    adapter_config_path = raw.get("adapter_config")
    if adapter_config_path:
        with open(Path(adapter_config_path), "r", encoding="utf-8") as fh:
            section = json.load(fh)
        if not isinstance(section, dict):
            raise ValueError(
                f"Adapter config must be a JSON object: {adapter_config_path}"
            )
    else:
        section = {}

    overlay = {k: v for k, v in raw.items() if k not in _CONFIG_META_KEYS}
    merged = _deep_merge(section, overlay)

    data_raw = merged.get("data", {})
    train_raw = data_raw.get("train", {})
    test_raw = data_raw.get("test", {})
    train_split = ResearchSplitConfig(
        files=train_raw.get("files"),
        max_tasks=train_raw.get("max_tasks"),
    )
    test_split = ResearchSplitConfig(
        files=test_raw.get("files"),
        max_tasks=test_raw.get("max_tasks"),
    )

    model_raw = merged.get("model", {})
    llm_api = model_raw.get("api")
    if llm_api is None:
        raise ValueError("`model.api` must be set in the config.")
    llm_name = model_raw.get("name", "")
    if llm_name is None:
        llm_name = ""
    extra_model_kwargs = {
        k: v
        for k, v in model_raw.items()
        if k not in {"api", "name", "api_key", "api_base"}
    }

    judge_raw = merged.get("judge", {})
    run_merged = merged.get("run", {})
    mas_raw = merged.get("mas", {})

    return ResearchRuntimeConfig(
        domain="researchworld",
        mode=merged.get("mode", "train"),
        profile=profile,
        llm=LLMConfig(
            api=llm_api,
            name=llm_name,
            api_key=model_raw.get("api_key"),
            api_base=model_raw.get("api_base"),
            extra_kwargs=extra_model_kwargs,
        ),
        data=ResearchDataConfig(train=train_split, test=test_split),
        judge=ResearchJudgeConfig(
            enabled=judge_raw.get("enabled", True),
            threshold=judge_raw.get("threshold", 0.5),
            verbose=judge_raw.get("verbose", False),
        ),
        run=ResearchRunConfig(
            num_epochs=run_merged.get("num_epochs", 2),
            verbose=run_merged.get("verbose", True),
            output=run_merged.get("output", "checkpoints/hayek_research.json"),
            checkpoint=run_merged.get("checkpoint"),
            checkpoint_steps=run_merged.get("checkpoint_steps"),
            resume_run_dir=run_merged.get("resume_run_dir"),
            resume_from_task=run_merged.get("resume_from_task"),
            max_eval_workers=run_merged.get("max_eval_workers", 10),
        ),
        mas=configure_research_hayek_config(HayekConfig.from_dict(mas_raw)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# FACTORIES
# ═══════════════════════════════════════════════════════════════════════════

def create_research_agents(llm_client: LLMClient) -> List[BaseAgent]:
    """Instantiate the initial five-agent roster."""
    return [
        cls(backbone_llm=llm_client.as_callable(), logger=logger)
        for cls in RESEARCH_AGENT_CLASSES
    ]


def make_research_agent_deserializer(
    llm_client: LLMClient,
) -> Callable[[Dict], ResearchAgent]:
    """Return a checkpoint deserializer for research agents."""

    def deserialize(data: Dict) -> ResearchAgent:
        return ResearchAgent.deserialize(
            data,
            backbone_llm=llm_client.as_callable(),
            logger=logger,
        )

    return deserialize


def create_research_env(
    task: ResearchTask,
    *,
    llm_client: LLMClient,
    reward_config: Optional[RewardConfig] = None,
    use_judge: bool = True,
    judge_threshold: float = 0.5,
    judge_verbose: bool = False,
    max_steps: int = 10,
) -> ResearchEnv:
    """Build a :class:`ResearchEnv` and attach the judge LLM callable."""
    env = ResearchEnv(
        task=task.problem,
        rubric=task.rubric,
        subject=task.subject,
        reward_config=reward_config,
        use_judge=use_judge,
        judge_threshold=judge_threshold,
        judge_verbose=judge_verbose,
        max_steps=max_steps,
    )
    env.llm_fn = llm_client.as_callable()
    return env


def _build_test_prediction_record(env: Any, task: ResearchTask) -> Dict[str, Any]:
    """Serialize one periodic-test episode into a re-scorable prediction row."""
    final_answer = getattr(env, "final_answer", None)
    terminal_score = getattr(env, "_last_terminal_score", None)
    if terminal_score is None and hasattr(env, "get_terminal_score"):
        terminal_score = env.get_terminal_score()
    return {
        "id": task.id,
        "subject": task.subject,
        "problem": (task.problem or "")[:400],
        "rubric": (task.rubric or "")[:1000],
        "answer": (final_answer or "")[:2000],
        "score": float(terminal_score) if terminal_score is not None else None,
        "passed": (
            bool(getattr(env, "_last_judge_passed", None))
            if getattr(env, "_last_judge_passed", None) is not None
            else None
        ),
        "judge_reason": (getattr(env, "_last_judge_reason", None) or "")[:600],
        "steps": getattr(env, "step_count", None),
        "terminated": bool(getattr(env, "terminated", False)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════════════════════

class ResearchTrainer(Trainer):
    """Trainer for the researchworld workflow."""

    def __init__(self, llm_client: LLMClient, config: ResearchRuntimeConfig):
        super().__init__(
            num_epochs=config.run.num_epochs,
            ckpt_save_path=config.run.output,
            step_ckpt_save_path=config.run.checkpoint_steps,
            verbose=config.run.verbose,
            profile=config.profile,
        )
        self.runtime_config = config
        self.llm_client = llm_client
        self.hayek_config = deepcopy(config.mas)
        self.test_tasks: Optional[List[ResearchTask]] = None

    @property
    def domain_name(self) -> str:
        return "Research"

    # ── task loading ─────────────────────────────────────────────────────

    def load_tasks(
        self,
        data_files: Optional[List[str]],
        limit: Optional[int],
    ) -> List[ResearchTask]:
        paths = (
            [Path(p) for p in data_files]
            if data_files
            else [get_default_research_train_path()]
        )
        return load_research_tasks(paths, limit=limit)

    # ── MAS factories ────────────────────────────────────────────────────

    def create_agents(self) -> List[BaseAgent]:
        return create_research_agents(self.llm_client)

    def load_mas_from_checkpoint(self, checkpoint_path: str) -> HayekMAS:
        return HayekMAS.load(
            checkpoint_path,
            agent_deserializer=make_research_agent_deserializer(self.llm_client),
        )

    def create_agent_factory_good_birth(self):
        backbone_llm = self.llm_client.as_callable()

        def birth_good_agent(parent: ResearchAgent) -> ResearchAgent:
            return ResearchAgent.birth_good_agent(
                parent=parent,
                backbone_llm=backbone_llm,
                initial_wealth=self.hayek_config.engine.initial_wealth,
            )

        return birth_good_agent

    def create_agent_factory_bad_birth(self):
        backbone_llm = self.llm_client.as_callable()

        def birth_bad_agent(
            source_agent: ResearchAgent,
            task_description: str = "",
            correct_answer: str = "",
            failure_trace: str = "",
        ) -> ResearchAgent:
            return ResearchAgent.birth_bad_agent(
                source_agent=source_agent,
                backbone_llm=backbone_llm,
                initial_wealth=self.hayek_config.engine.initial_wealth,
                task_description=task_description,
                correct_answer=correct_answer,
                failure_trace=failure_trace,
            )

        return birth_bad_agent

    # ── env + success check ──────────────────────────────────────────────

    def create_env(self, task: ResearchTask) -> ResearchEnv:
        return create_research_env(
            task,
            llm_client=self.llm_client,
            reward_config=self.hayek_config.reward,
            use_judge=self.runtime_config.judge.enabled,
            judge_threshold=self.runtime_config.judge.threshold,
            judge_verbose=self.runtime_config.judge.verbose,
            max_steps=self.hayek_config.engine.max_steps_per_episode,
        )

    def check_success(self, env: ResearchEnv, task: ResearchTask) -> bool:
        _ = task
        if not env.final_answer:
            return False
        cached = env.is_successful()
        if cached is not None:
            return cached
        passed, _, _ = env.check_answer_correct(env.final_answer, env.expected_output)
        return passed

    # ── logging helpers ──────────────────────────────────────────────────

    def get_task_id(self, task: Any) -> str:
        return task.id

    def get_task_description(self, task: Any) -> str:
        text = task.problem
        return f"{text[:100]}..." if len(text) > 100 else text

    def print_task_details(self, task: ResearchTask):
        logger.log(f"   Problem: {task.problem[:120]}")
        logger.log(f"   Subject: {task.subject}")

    def print_success(self, env: ResearchEnv, task: ResearchTask):
        _ = task
        preview = (env.final_answer or "").strip().replace("\n", " ")[:160]
        logger.log(f"\n   ✅ PASSED (score={env.get_terminal_score()}) | {preview}")

    def print_failure(self, env: ResearchEnv, task: ResearchTask):
        _ = task
        logger.log(
            f"\n   ❌ FAILED (score={env.get_terminal_score()}) | "
            f"{(env.final_answer or '')[:120]}"
        )

    # ── periodic test ────────────────────────────────────────────────────

    def _load_test_tasks(self) -> List[ResearchTask]:
        if self.test_tasks is not None:
            return self.test_tasks
        test_cfg = self.runtime_config.data.test
        if test_cfg.files:
            test_path = Path(test_cfg.files[0])
        else:
            test_path = get_default_research_test_path()
        self.test_tasks = load_research_tasks([test_path], limit=test_cfg.max_tasks)
        return self.test_tasks

    def _serialize_mas_snapshot(self) -> Dict[str, Any]:
        if self.mas is None:
            raise RuntimeError(
                "MAS must exist before serializing a periodic test snapshot."
            )
        snapshot = self.mas.serialize_settings()
        snapshot["population"] = [
            agent.serialize()
            if hasattr(agent, "serialize")
            else {
                "id": agent.id,
                "name": agent.name,
                "wealth": agent.wealth,
                "capability_score": getattr(agent, "capability_score", 0.0),
                "bid": agent.get_bid(),
                "status": agent.get_status().value,
                "type": agent.__class__.__name__,
            }
            for agent in self.mas.population.get_all()
        ]
        return snapshot

    def _run_periodic_test_task(
        self,
        task: ResearchTask,
        mas_snapshot: Dict[str, Any],
    ) -> Tuple[bool, Optional[float], Dict[str, Any]]:
        with logger.scoped_silence():
            mas = HayekMAS.load_from_data(
                mas_snapshot,
                agent_deserializer=make_research_agent_deserializer(self.llm_client),
            )
            mas.eval()
            env = create_research_env(
                task,
                llm_client=self.llm_client,
                reward_config=mas.config.reward,
                use_judge=self.runtime_config.judge.enabled,
                judge_threshold=self.runtime_config.judge.threshold,
                judge_verbose=self.runtime_config.judge.verbose,
                max_steps=mas.max_steps_per_episode,
            )
            mas.run_one_episode(env)
        prediction = _build_test_prediction_record(env, task)
        return self.check_success(env, task), env.get_terminal_score(), prediction

    def run_periodic_test(
        self,
        *,
        epoch: int,
        tasks_completed: int,
    ) -> Optional[dict]:
        if self.mas is None:
            return None

        test_tasks = self._load_test_tasks()
        if not test_tasks:
            return None

        logger.log("\n" + "=" * 70)
        logger.log(
            f"🧪 PERIODIC TEST EVAL after {tasks_completed} training tasks "
            f"(epoch {epoch}, {len(test_tasks)} test tasks)"
        )
        logger.log("=" * 70)

        was_training = self.mas.training
        saved_episode_count = self.mas.episode_count
        saved_last_termination_reason = self.mas.last_termination_reason
        saved_last_episode_metrics = dict(self.mas.last_episode_metrics)

        success_count = 0
        terminal_scores: List[float] = []
        failed_task_count = 0
        completed_task_count = 0
        prediction_records: List[Dict[str, Any]] = []
        max_workers = max(1, self.hayek_config.evaluation.periodic_test_max_workers)
        use_parallel = (
            self.hayek_config.evaluation.periodic_test_parallel_enabled
            and max_workers > 1
            and len(test_tasks) > 1
        )

        self.mas.eval()
        try:
            if use_parallel:
                logger.log(
                    f"   Periodic test progress: 0/{len(test_tasks)} completed "
                    f"(parallel, workers={max_workers})",
                    indent=1,
                )
                mas_snapshot = self._serialize_mas_snapshot()
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_task = {
                        executor.submit(
                            self._run_periodic_test_task, task, mas_snapshot
                        ): task
                        for task in test_tasks
                    }
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            success, score, prediction = future.result()
                        except Exception as exc:  # noqa: BLE001
                            failed_task_count += 1
                            completed_task_count += 1
                            logger.log(
                                f"   [{completed_task_count}/{len(test_tasks)}] "
                                f"⚠️  Periodic test task {task.id} failed: {exc}",
                                indent=1,
                            )
                            prediction_records.append({
                                "id": task.id,
                                "problem": (task.problem or "")[:400],
                                "rubric": (task.rubric or "")[:1000],
                                "answer": "",
                                "score": None,
                                "passed": None,
                                "judge_reason": f"exception: {exc!r}"[:600],
                                "steps": None,
                                "terminated": False,
                                "error": str(exc),
                            })
                            continue
                        prediction_records.append(prediction)
                        if success:
                            success_count += 1
                        if score is not None:
                            terminal_scores.append(score)
                        completed_task_count += 1
                        logger.log(
                            f"   [{completed_task_count}/{len(test_tasks)}] "
                            f"Periodic test task {task.id} done "
                            f"({'success' if success else 'fail'})",
                            indent=1,
                        )
            else:
                for task in test_tasks:
                    try:
                        env = self.create_env(task)
                        self.mas.run_one_episode(env)
                        success = self.check_success(env, task)
                        if success:
                            success_count += 1
                        score = env.get_terminal_score()
                        if score is not None:
                            terminal_scores.append(score)
                        prediction_records.append(_build_test_prediction_record(env, task))
                    except Exception as exc:  # noqa: BLE001
                        failed_task_count += 1
                        prediction_records.append({
                            "id": task.id,
                            "problem": (task.problem or "")[:400],
                            "rubric": (task.rubric or "")[:1000],
                            "answer": "",
                            "score": None,
                            "passed": None,
                            "judge_reason": f"exception: {exc!r}"[:600],
                            "steps": None,
                            "terminated": False,
                            "error": str(exc),
                        })
                        logger.log(
                            f"   ⚠️  Periodic test task {task.id} failed: {exc}",
                            indent=1,
                        )
                    finally:
                        completed_task_count += 1
        finally:
            self.mas.episode_count = saved_episode_count
            self.mas.last_termination_reason = saved_last_termination_reason
            self.mas.last_episode_metrics = saved_last_episode_metrics
            if was_training:
                self.mas.train()
            else:
                self.mas.eval()

        total = len(test_tasks)
        accuracy = success_count / total if total else 0.0
        avg_terminal_score = (
            sum(terminal_scores) / len(terminal_scores) if terminal_scores else None
        )
        if avg_terminal_score is not None:
            logger.log(
                f"   Periodic test score={accuracy:.4f} | success={success_count}/{total} "
                f"| avg_terminal_score={avg_terminal_score:.4f}"
            )
        else:
            logger.log(
                f"   Periodic test score={accuracy:.4f} | success={success_count}/{total}"
            )
        if failed_task_count:
            logger.log(f"   Periodic test failures: {failed_task_count}", indent=1)

        # Dump per-task predictions for future re-scoring.
        predictions_path: Optional[Path] = None
        if prediction_records and getattr(self, "outputs_dir", None):
            outputs_dir = Path(self.outputs_dir)
            outputs_dir.mkdir(parents=True, exist_ok=True)
            predictions_path = (
                outputs_dir / f"test_predictions_step{tasks_completed:03d}.jsonl"
            )
            with open(predictions_path, "w", encoding="utf-8") as fh:
                for rec in prediction_records:
                    rec_with_ctx = dict(rec)
                    rec_with_ctx.setdefault("epoch", epoch)
                    rec_with_ctx.setdefault("after_train_tasks", tasks_completed)
                    rec_with_ctx.setdefault(
                        "judge_threshold", self.runtime_config.judge.threshold
                    )
                    fh.write(json.dumps(rec_with_ctx, ensure_ascii=True, default=str) + "\n")
            logger.log(f"   Wrote {predictions_path.name}", indent=1)

        return {
            "epoch": epoch,
            "after_train_tasks": tasks_completed,
            "test_stage": "pretrain" if tasks_completed == 0 else "periodic",
            "test_score": accuracy,
            "test_accuracy": accuracy,
            "test_success": success_count,
            "test_total": total,
            "test_avg_terminal_score": avg_terminal_score,
            "test_failed_task_count": failed_task_count,
            "test_parallel_enabled": use_parallel,
            "test_max_workers": max_workers,
            "test_data_path": str(
                self.runtime_config.data.test.files[0]
                if self.runtime_config.data.test.files
                else get_default_research_test_path()
            ),
            "test_predictions_path": (
                str(predictions_path.name) if predictions_path is not None else None
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════

class ResearchEvaluator(Evaluator):

    def __init__(self, llm_client: LLMClient, config: ResearchRuntimeConfig):
        super().__init__(
            checkpoint_path=config.run.checkpoint or config.run.output,
            verbose=config.run.verbose,
            hayek_config_overrides={
                "max_steps_per_episode": config.mas.engine.max_steps_per_episode
            },
            profile=config.profile,
        )
        self.runtime_config = config
        self.llm_client = llm_client

    @property
    def domain_name(self) -> str:
        return "Research"

    def create_deserializer(self):
        return make_research_agent_deserializer(self.llm_client)

    def load_tasks(self, data_files, limit):
        paths = (
            [Path(p) for p in data_files]
            if data_files
            else [get_default_research_test_path()]
        )
        return load_research_tasks(paths, limit=limit)

    def create_env(self, task: ResearchTask) -> ResearchEnv:
        max_steps = self.mas.max_steps_per_episode if self.mas is not None else 10
        reward_config = self.mas.config.reward if self.mas is not None else None
        return create_research_env(
            task,
            llm_client=self.llm_client,
            reward_config=reward_config,
            use_judge=self.runtime_config.judge.enabled,
            judge_threshold=self.runtime_config.judge.threshold,
            judge_verbose=self.runtime_config.judge.verbose,
            max_steps=max_steps or 10,
        )

    def check_success(self, env: ResearchEnv, task: ResearchTask) -> bool:
        _ = task
        if not env.final_answer:
            return False
        cached = env.is_successful()
        if cached is not None:
            return cached
        passed, _, _ = env.check_answer_correct(env.final_answer, env.expected_output)
        return passed


# ═══════════════════════════════════════════════════════════════════════════
# LLM CLIENT
# ═══════════════════════════════════════════════════════════════════════════

def _build_llm_client(config: ResearchRuntimeConfig) -> LLMClient:
    logger.log("\n" + "=" * 70)
    logger.log("🤖 SETTING UP LLM BACKEND (researchworld)")
    logger.log("=" * 70)
    logger.log(f"   API: {config.llm.api}")
    if config.llm.name:
        logger.log(f"   Model: {config.llm.name}")

    client_kwargs = dict(config.llm.extra_kwargs)
    if config.llm.api_base:
        client_kwargs["api_base"] = config.llm.api_base
    client_kwargs["api_key"] = config.llm.api_key

    client = get_llm_client(config.llm.api, model=config.llm.name, **client_kwargs)
    logger.log(f"   Client: {client}")
    return client


# ═══════════════════════════════════════════════════════════════════════════
# ENTRYPOINTS
# ═══════════════════════════════════════════════════════════════════════════

def run(config: ResearchRuntimeConfig) -> None:
    if config.mode not in {"train", "eval"}:
        raise ValueError(f"Unsupported research runtime mode: {config.mode}")

    llm_client = _build_llm_client(config)
    if config.judge.enabled:
        logger.log("   Judge: ENABLED (rubric-based LLM judge)")

    if config.mode == "train":
        trainer = ResearchTrainer(llm_client=llm_client, config=config)
        trainer.train(
            data_files=config.data.train.files,
            max_tasks=config.data.train.max_tasks,
            resume_run_dir=config.run.resume_run_dir,
            resume_from_task=config.run.resume_from_task,
        )
        trainer.save()
        logger.log("\n✨ Training complete!")
        logger.close()
        return

    evaluator = ResearchEvaluator(llm_client=llm_client, config=config)
    eval_cfg = config.data.test if config.data.test.files else config.data.train
    try:
        evaluator.evaluate(
            data_files=eval_cfg.files,
            limit=eval_cfg.max_tasks,
            max_workers=config.run.max_eval_workers,
        )
    except FileNotFoundError as exc:
        logger.log(f"\n❌ {exc}")
        return
    logger.close()


def main(raw_config: Dict[str, Any]) -> None:
    run(load_research_runtime_config(raw_config))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m hayekmas.adapters.researchworld.runtime <config.json>"
        )
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    main(raw)
