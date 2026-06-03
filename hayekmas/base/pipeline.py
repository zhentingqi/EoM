"""
Core runtime scaffolding for adapter train and evaluation workflows.

This file defines the generic training and evaluation pipeline base classes
used by adapter runtimes. Adapters provide domain-specific task loading,
environment construction, success checks, and checkpoint deserialization.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import glob
import json
import os
import threading

from hayekmas.base.agent import BaseAgent
from hayekmas.base.env import BaseEnv
from hayekmas.base.config import DEFAULT_HAYEK_CONFIG, HayekConfig
from hayekmas.base.mas import HayekMAS
from hayekmas.utils.logger import logger


class Trainer(ABC):
    """Generic training scaffold shared by adapter runtimes.

    Args:
        num_epochs: Number of epochs to train for.
        ckpt_save_path: Output path for the final checkpoint.
        step_ckpt_save_path: Optional directory for per-step checkpoints.
        verbose: Whether to emit verbose logs.
    """
    def __init__(
        self,
        num_epochs: int,    # Number of epochs to train (passes through all tasks)
        ckpt_save_path: str, # Path to save the trained system
        step_ckpt_save_path: Optional[str] = None, # If set, write step checkpoints (step_N.json) per episode here.
        verbose: bool = True, # Whether to print detailed logs
        profile: str = "",  # Experiment profile name, reflected in output dir
    ):
        self.num_epochs = num_epochs
        self.ckpt_save_path = ckpt_save_path
        self.step_ckpt_save_path = step_ckpt_save_path
        self.verbose = verbose
        self.profile = profile

        # Initialize MAS attributes
        self.mas = None
        self.tasks: List[Any] = []
        self.outputs_dir: Optional[Path] = None
        self.training_metrics_path: Optional[Path] = None
        self.agent_task_metrics_path: Optional[Path] = None
        self.population_metrics_path: Optional[Path] = None
        self.checkpoints_dir: Optional[Path] = None
        self.per_task_log_dir: Optional[Path] = None
        self.run_state_path: Optional[Path] = None
        self.progress_history_path: Optional[Path] = None
        self.hayek_config: HayekConfig = deepcopy(DEFAULT_HAYEK_CONFIG)
        self.experiment_settings_path: Optional[Path] = None

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Return the domain name."""
        pass

    @abstractmethod
    def create_agents(self) -> List[BaseAgent]:
        """Create and return domain-specific agents."""
        pass

    @abstractmethod
    def load_tasks(self, data_files: Optional[List[str]], limit: Optional[int]) -> List[Any]:
        """Load training tasks."""
        pass

    @abstractmethod
    def create_env(self, task: Any) -> BaseEnv:
        """Create an environment from a task."""
        pass

    @abstractmethod
    def check_success(self, env: BaseEnv, task: Any) -> bool:
        """Check if a task was successfully solved."""
        pass

    def get_task_id(self, task: Any) -> str:
        """Get a string identifier for a task."""
        if hasattr(task, "id"):
            return task.id
        return str(task)

    def get_task_description(self, task: Any) -> str:
        """Get a short description of a task."""
        max_length = 1024
        if hasattr(task, "question"):
            text = task.question
            return f"{text[:max_length]} ... [truncated]" if len(text) > max_length else text
        s = str(task)
        return f"{s[:max_length]} ... [truncated]" if len(s) > max_length else s

    def print_task_details(self, task: Any):
        """Print task-specific details. Override for custom formatting."""
        desc = self.get_task_description(task)
        logger.log(f"   {desc}")

    def print_success(self, env: BaseEnv, task: Any):
        """Print success message. Override for custom formatting."""
        logger.log(f"\n   ✅ SOLVED in {env.step_count} steps!")

    def print_failure(self, env: BaseEnv, task: Any):
        """Print failure message. Override for custom formatting."""
        logger.log(f"\n   ❌ NOT SOLVED")

    @abstractmethod
    def create_agent_factory_good_birth(self) -> Callable[[BaseAgent], BaseAgent]:
        """
        Create and return the good-birth factory.

        The factory takes a strong agent and returns a more exploratory child.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def create_agent_factory_bad_birth(self) -> Callable:
        """
        Create and return the bad-birth factory.

        This factory takes a source agent plus failure context and creates an
        improved replacement by analyzing why the agent failed.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def load_mas_from_checkpoint(self, checkpoint_path: str):
        """
        Load a domain-specific MAS checkpoint with the right agent deserializer.

        Must return a HayekMAS instance.
        """
        pass

    def setup(self, data_files: Optional[List[str]] = None, max_tasks: Optional[int] = None):
        """Set up the trainer state before entering the train loop.

        Args:
            data_files: Optional training data file paths.
            max_tasks: Optional cap on the number of loaded tasks.
        """
        from hayekmas.base.mas import HayekMAS

        # Configure logging
        logger.configure(verbose=self.verbose, log_dir="logs", profile=self.profile)

        self._print_header()

        # Load tasks first
        logger.log("=" * 70)
        logger.log("📂 LOADING TRAINING DATA")
        logger.log("=" * 70)
        self.tasks = self.load_tasks(data_files, max_tasks)
        logger.log(f"\n✅ Loaded {len(self.tasks)} tasks")

        # Create MAS
        logger.log("\n" + "=" * 70)
        logger.log("👥 CREATING AGENTS")
        logger.log("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_{os.getpid()}"
        self.outputs_dir = Path("outputs") / f"{self.profile}_{run_id}"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.training_metrics_path = self.outputs_dir / "training_metrics.jsonl"
        self.agent_task_metrics_path = self.outputs_dir / "agent_task_metrics.jsonl"
        self.population_metrics_path = self.outputs_dir / "population_metrics.jsonl"
        self.checkpoints_dir = self.outputs_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.per_task_log_dir = self.outputs_dir / "per_task_log"
        self.per_task_log_dir.mkdir(parents=True, exist_ok=True)
        self.run_state_path = self.outputs_dir / "run_state.json"
        self.progress_history_path = self.outputs_dir / "progress_history.jsonl"
        self.experiment_settings_path = self.outputs_dir / "experiment_settings.json"
        logger.log(f"   Metrics output: {self.outputs_dir}")

        self.mas = HayekMAS(config=self.hayek_config)

        # Create and add agents
        agents = self.create_agents()
        logger.log(f"\n   Creating {len(agents)} agents:\n")
        for agent in agents:
            agent.initialize(initial_wealth=self.hayek_config.engine.initial_wealth)
            self.mas.population.add_agent(agent)
            logger.log(f"   ✓ {agent.name}")
        self.mas._initial_agents = list(agents)

        # Set up evolution factories.
        good_birth_factory = self.create_agent_factory_good_birth()
        bad_birth_factory = self.create_agent_factory_bad_birth()
        self.mas.set_agent_factory(good_birth_factory, bad_birth_factory)
        self._write_experiment_settings(
            event="fresh_start",
            data_files=data_files,
            max_tasks=max_tasks,
        )

    def _restore_from_run_state(
        self,
        *,
        resume_run_dir: str,
        data_files: Optional[List[str]],
        max_tasks: Optional[int],
        resume_from_task: Optional[int],
    ) -> Dict[str, int]:
        """
        Restore trainer + MAS state from a previous run directory.

        Returns resume counters for train-loop bootstrapping.
        """
        run_dir = Path(resume_run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run dir not found: {run_dir}")

        self.outputs_dir = run_dir
        self.training_metrics_path = run_dir / "training_metrics.jsonl"
        self.agent_task_metrics_path = run_dir / "agent_task_metrics.jsonl"
        self.population_metrics_path = run_dir / "population_metrics.jsonl"
        self.checkpoints_dir = run_dir / "checkpoints"
        self.per_task_log_dir = run_dir / "per_task_log"
        self.per_task_log_dir.mkdir(parents=True, exist_ok=True)
        self.run_state_path = run_dir / "run_state.json"
        self.progress_history_path = run_dir / "progress_history.jsonl"
        self.experiment_settings_path = run_dir / "experiment_settings.json"

        if not self.run_state_path.exists():
            raise FileNotFoundError(
                f"run_state.json not found in {run_dir}. Cannot resume this run."
            )
        with open(self.run_state_path, "r", encoding="utf-8") as f:
            run_state = json.load(f)

        resume_record = None
        if resume_from_task is not None:
            if not self.progress_history_path.exists():
                raise FileNotFoundError(
                    f"progress_history.jsonl not found in {run_dir}; cannot resume from task {resume_from_task}."
                )
            with open(self.progress_history_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if int(item.get("tasks_completed", -1)) == int(resume_from_task):
                        resume_record = item
            if resume_record is None:
                raise ValueError(
                    f"Task {resume_from_task} not found in progress history for run {run_dir}."
                )

        state_for_resume = resume_record if resume_record is not None else run_state
        checkpoint_path = state_for_resume.get("mas_checkpoint_path")

        # Fallback to latest checkpoint if state path is unavailable.
        if not checkpoint_path:
            pattern = str((self.checkpoints_dir or run_dir).joinpath("task_*.json"))
            candidates = sorted(glob.glob(pattern))
            if not candidates:
                raise FileNotFoundError(
                    f"No MAS checkpoints found in {(self.checkpoints_dir or run_dir)}."
                )
            checkpoint_path = candidates[-1]

        self.mas = self.load_mas_from_checkpoint(checkpoint_path)

        # Re-attach factories for continued evolution after loading.
        good_birth_factory = self.create_agent_factory_good_birth()
        bad_birth_factory = self.create_agent_factory_bad_birth()
        self.mas.set_agent_factory(good_birth_factory, bad_birth_factory)
        self._write_experiment_settings(
            event="resume",
            data_files=data_files,
            max_tasks=max_tasks,
            extra={
                "resume_run_dir": str(run_dir),
                "resume_from_task": resume_from_task,
                "resumed_checkpoint_path": checkpoint_path,
                "resumed_tasks_completed": int(state_for_resume.get("tasks_completed", 0)),
            },
        )

        # Respect explicit args first, then persisted values.
        restored_data_files = data_files
        if restored_data_files is None:
            restored_data_files = state_for_resume.get("data_files")

        restored_max_tasks = max_tasks
        if restored_max_tasks is None:
            restored_max_tasks = state_for_resume.get("max_tasks")

        self.tasks = self.load_tasks(restored_data_files, restored_max_tasks)
        logger.log(f"   ↪ Resumed outputs: {run_dir}")
        logger.log(f"   ↪ Loaded checkpoint: {checkpoint_path}")
        logger.log(
            f"   ↪ Resume point: epoch={state_for_resume.get('next_epoch', 0) + 1}, "
            f"task_index={state_for_resume.get('next_task_index', 0) + 1}, "
            f"tasks_completed={state_for_resume.get('tasks_completed', 0)}"
        )

        return {
            "start_epoch": int(state_for_resume.get("next_epoch", 0)),
            "start_task_index": int(state_for_resume.get("next_task_index", 0)),
            "tasks_completed": int(state_for_resume.get("tasks_completed", 0)),
            "total_processed": int(state_for_resume.get("total_tasks_processed", 0)),
            "total_success": int(state_for_resume.get("total_success_count", 0)),
        }

    def _save_progress(
        self,
        *,
        epoch_index: int,
        task_index: int,
        tasks_completed: int,
        total_processed: int,
        total_success: int,
        data_files: Optional[List[str]],
        max_tasks: Optional[int],
    ) -> None:
        """
        Save checkpoint + run state after each completed training task.
        """
        if self.mas is None or self.checkpoints_dir is None:
            return

        checkpoint_path = self.checkpoints_dir / f"task_{tasks_completed:05d}.json"
        self.mas.save(str(checkpoint_path))

        next_task_index = task_index + 1
        next_epoch = epoch_index
        if self.tasks and next_task_index >= len(self.tasks):
            next_task_index = 0
            next_epoch = epoch_index + 1

        payload = {
            "version": 1,
            "domain": self.domain_name,
            "timestamp": datetime.now().isoformat(),
            "num_epochs": self.num_epochs,
            "max_steps_per_episode": self.hayek_config.engine.max_steps_per_episode,
            "data_files": data_files,
            "max_tasks": max_tasks,
            "outputs_dir": str(self.outputs_dir) if self.outputs_dir else None,
            "mas_checkpoint_path": str(checkpoint_path),
            "next_epoch": next_epoch,
            "next_task_index": next_task_index,
            "tasks_completed": tasks_completed,
            "total_tasks_processed": total_processed,
            "total_success_count": total_success,
            "population_size": len(self.mas.population) if self.mas else None,
        }

        if self.run_state_path is not None:
            with open(self.run_state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2)

        self._append_jsonl(self.progress_history_path, payload)

    def _get_per_task_log_path(self, task_id: str) -> Optional[Path]:
        """Return the per-task log path for one task."""
        if self.per_task_log_dir is None:
            return None
        return self.per_task_log_dir / f"{task_id}.log"

    def train(
        self,
        data_files: Optional[List[str]] = None,
        max_tasks: Optional[int] = None,
        resume_run_dir: Optional[str] = None,
        resume_from_task: Optional[int] = None,
    ) -> dict:
        """Run training on all tasks for multiple epochs.

        Args:
            data_files: Optional training data file paths.
            max_tasks: Optional cap on the number of loaded tasks.
            resume_run_dir: Optional previous run directory to resume from.
            resume_from_task: Optional task counter to resume after.

        Returns:
            A summary dictionary with aggregate training metrics.
        """
        resume_state = None
        if resume_run_dir:
            # Configure logging first so resume diagnostics are written to logs.
            logger.configure(verbose=self.verbose, log_dir="logs", profile=self.profile)
            self._print_header()
            logger.log("=" * 70)
            logger.log("🔁 RESUMING TRAINING RUN")
            logger.log("=" * 70)
            resume_state = self._restore_from_run_state(
                resume_run_dir=resume_run_dir,
                data_files=data_files,
                max_tasks=max_tasks,
                resume_from_task=resume_from_task,
            )
        elif self.mas is None:
            self.setup(data_files, max_tasks)

        logger.print_mode_banner("TRAIN", f"TRAINING BEGINS — {self.domain_name}")
        logger.log(f"\nTraining on tasks for {self.num_epochs} epoch(s)...")
        logger.log("Each task is one episode in the Hayek economy.\n")

        # Put MAS into training mode
        self.mas.train()

        self.results = []
        total_success_count = int(resume_state["total_success"]) if resume_state else 0
        total_tasks_processed = int(resume_state["total_processed"]) if resume_state else 0
        tasks_completed = int(resume_state["tasks_completed"]) if resume_state else 0
        start_epoch = int(resume_state["start_epoch"]) if resume_state else 0
        start_task_index = int(resume_state["start_task_index"]) if resume_state else 0

        if resume_state is None and self.should_run_pretrain_test():
            test_metrics = self.run_periodic_test(epoch=0, tasks_completed=0)
            self._write_test_metrics(test_metrics)

        for epoch in range(start_epoch, self.num_epochs):
            logger.log("\n" + "▓" * 70)
            logger.log(f"▓ EPOCH {epoch}/{self.num_epochs} ".ljust(69) + "▓")
            logger.log("▓" * 70)

            epoch_success_count = 0

            epoch_start_task = start_task_index if epoch == start_epoch else 0
            for i in range(epoch_start_task, len(self.tasks)):
                task = self.tasks[i]
                task_id = self.get_task_id(task)
                with logger.scoped_task_log(self._get_per_task_log_path(task_id)):
                    # Progress header
                    logger.log(f"\n{'█'*70}")
                    logger.log(f"📋 Epoch {epoch} | Task [{i+1}/{len(self.tasks)}]")
                    logger.log(f"{'─'*70}")
                    logger.log("Task Details\n")
                    self.print_task_details(task)
                    logger.log("\n")

                    env = self.create_env(task)

                    step_ckpt_save_path = None
                    if self.step_ckpt_save_path:
                        step_ckpt_save_path = os.path.join(
                            self.step_ckpt_save_path,
                            f"epoch_{epoch}",
                            f"task_{i}",
                        )
                    self.mas.run_one_episode(env, step_ckpt_save_path=step_ckpt_save_path)

                    success = self.check_success(env, task)

                    self._write_task_outputs(
                        epoch=epoch,
                        task_id=task_id,
                        task=task,
                        success=success,
                    )

                    if success:
                        epoch_success_count += 1
                        total_success_count += 1
                        self.print_success(env, task)
                    else:
                        self.print_failure(env, task)

                total_tasks_processed += 1
                tasks_completed += 1

                self._save_progress(
                    epoch_index=epoch,
                    task_index=i,
                    tasks_completed=tasks_completed,
                    total_processed=total_tasks_processed,
                    total_success=total_success_count,
                    data_files=data_files,
                    max_tasks=max_tasks,
                )

                if self.should_run_periodic_test(tasks_completed):
                    test_metrics = self.run_periodic_test(
                        epoch=epoch,
                        tasks_completed=tasks_completed,
                    )
                    self._write_test_metrics(test_metrics)
                    if test_metrics and self.run_state_path is not None and self.run_state_path.exists():
                        with open(self.run_state_path, "r", encoding="utf-8") as f:
                            current_state = json.load(f)
                        current_state["last_periodic_test"] = test_metrics
                        with open(self.run_state_path, "w", encoding="utf-8") as f:
                            json.dump(current_state, f, ensure_ascii=True, indent=2)

            # Print epoch summary
            epoch_accuracy = 100 * epoch_success_count / len(self.tasks) if self.tasks else 0
            logger.log(f"\n📊 Epoch {epoch} Summary: {epoch_success_count}/{len(self.tasks)} ({epoch_accuracy:.1f}%)")

        self._print_results(total_success_count, total_tasks_processed)

        return {
            "domain": self.domain_name,
            "num_epochs": self.num_epochs,
            "tasks_per_epoch": len(self.tasks),
            "total": total_tasks_processed,
            "success": total_success_count,
            "accuracy": total_success_count / total_tasks_processed if total_tasks_processed else 0,
        }

    def should_run_periodic_test(self, tasks_completed: int) -> bool:
        """Return whether a periodic test evaluation should run after this task."""
        interval = self.hayek_config.evaluation.periodic_test_every_n_tasks
        return (
            self.hayek_config.evaluation.periodic_test_enabled
            and interval > 0
            and tasks_completed % interval == 0
        )

    def should_run_pretrain_test(self) -> bool:
        """Return whether a step-0 test evaluation should run before training."""
        return (
            self.hayek_config.evaluation.periodic_test_enabled
            and self.hayek_config.evaluation.periodic_test_before_training
        )

    def run_periodic_test(
        self,
        *,
        epoch: int,
        tasks_completed: int,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Run a periodic test-set evaluation, one record per configured test file.

        Returns a mapping ``{basename: metrics_record}`` where ``basename`` is the
        test-file filename without its ``.jsonl`` extension. Each metrics record
        is appended to ``test_metrics__<basename>.jsonl`` by the trainer.
        Return ``None`` when no test data is configured.
        """
        _ = epoch
        _ = tasks_completed
        return None

    def _write_test_metrics(
        self, metrics_by_file: Optional[Dict[str, Dict[str, Any]]]
    ) -> None:
        """Append each per-file periodic-test record to its own JSONL file."""
        if not metrics_by_file or self.outputs_dir is None:
            return
        for basename, record in metrics_by_file.items():
            path = self.outputs_dir / f"test_metrics__{basename}.jsonl"
            self._append_jsonl(path, record)

    def save(self, ckpt_save_path: Optional[str] = None):
        """Save the trained model. Uses agent.serialize() for each agent."""
        if self.mas is None:
            raise RuntimeError("No trained model to save. Call train() first.")

        path = ckpt_save_path or self.ckpt_save_path

        logger.log("\n" + "=" * 70)
        logger.log("💾 SAVING MODEL")
        logger.log("=" * 70)

        # mas.save() will use agent.serialize() by default
        self.mas.save(path)

    def _print_header(self):
        """Print training header."""
        logger.print_mode_banner("TRAIN", f"HAYEK MACHINE TRAINING: {self.domain_name}")

    def _print_results(self, success: int, total: int):
        """Print training results."""
        logger.print_mode_banner("TRAIN", "TRAINING COMPLETE")

        accuracy = 100 * success / total if total > 0 else 0
        logger.log(f"\n📊 Success Rate: {success}/{total} ({accuracy:.1f}%)")

        if self.mas:
            self.mas.print_population()

    def _append_jsonl(self, path: Optional[Path], payload: dict) -> None:
        """Append a JSON record to a JSONL file."""
        if path is None:
            return

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _serialize_hayek_config(self) -> dict:
        config = self.hayek_config
        if is_dataclass(config):
            return asdict(config)
        return {}

    def _serialize_llm_client(self) -> dict:
        llm_client = getattr(self, "llm_client", None)
        if llm_client is None:
            return {}

        payload = {
            "client_class": llm_client.__class__.__name__,
            "api_name": getattr(llm_client, "api_name", None),
            "model": getattr(llm_client, "model", None),
            "default_max_tokens": getattr(llm_client, "default_max_tokens", None),
            "default_temperature": getattr(llm_client, "default_temperature", None),
            "extra_kwargs": getattr(llm_client, "extra_kwargs", {}),
        }
        for attr in (
            "top_p",
            "stream",
            "reasoning_enabled",
            "force_openai_routing",
            "drop_params",
            "max_retries",
            "base_delay",
            "api_base",
            "api_url",
        ):
            if hasattr(llm_client, attr):
                payload[attr] = getattr(llm_client, attr)
        return payload

    def _build_experiment_settings(
        self,
        *,
        event: str,
        data_files: Optional[List[str]],
        max_tasks: Optional[int],
        extra: Optional[dict] = None,
    ) -> dict:
        settings = {
            "recorded_at": datetime.now().isoformat(),
            "event": event,
            "domain": self.domain_name,
            "outputs_dir": str(self.outputs_dir) if self.outputs_dir else None,
            "trainer": {
                "num_epochs": self.num_epochs,
                "max_steps_per_episode": self.hayek_config.engine.max_steps_per_episode,
                "verbose": self.verbose,
                "output_path": self.ckpt_save_path,
                "checkpoint_steps_dir": self.step_ckpt_save_path,
                "data_files": data_files,
                "max_tasks": max_tasks,
            },
            "llm": self._serialize_llm_client(),
            "hayek_config": self._serialize_hayek_config(),
        }
        if extra:
            settings["extra"] = extra
        return settings

    def _write_experiment_settings(
        self,
        *,
        event: str,
        data_files: Optional[List[str]],
        max_tasks: Optional[int],
        extra: Optional[dict] = None,
    ) -> None:
        if self.experiment_settings_path is None:
            return

        new_entry = self._build_experiment_settings(
            event=event,
            data_files=data_files,
            max_tasks=max_tasks,
            extra=extra,
        )

        existing: dict = {}
        if self.experiment_settings_path.exists():
            try:
                with open(self.experiment_settings_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}

        history = list(existing.get("history", []))
        history.append(new_entry)

        payload = {
            "latest": new_entry,
            "history": history,
        }
        with open(self.experiment_settings_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _write_task_outputs(
        self,
        *,
        epoch: int,
        task_id: str,
        task: Any,
        success: bool,
    ) -> None:
        """Persist episode-level and agent-level metrics under outputs/."""
        if self.mas is None:
            return

        episode_metrics = dict(getattr(self.mas, "last_episode_metrics", {}) or {})
        agent_records = episode_metrics.pop("agent_records", [])
        task_record = {
            "epoch": epoch,
            "task_id": task_id,
            "success": success,
            "task_description": self.get_task_description(task),
            **episode_metrics,
        }
        self._append_jsonl(self.training_metrics_path, task_record)

        for agent_record in agent_records:
            payload = {
                "epoch": epoch,
                "task_id": task_id,
                "success": success,
                **agent_record,
            }
            self._append_jsonl(self.agent_task_metrics_path, payload)

        if self.mas is not None:
            population_snapshot = {
                "epoch": epoch,
                "task_id": task_id,
                "population_size": len(self.mas.population),
                "agents": [
                    {
                        "id": agent.id,
                        "name": agent.name,
                        "class": agent.__class__.__name__,
                        "wealth": agent.wealth,
                        "capability_score": getattr(agent, "capability_score", 0.0),
                        "bid": agent.get_bid(),
                        "status": getattr(agent.get_status(), "value", str(agent.get_status())),
                        "frozen_system_prompt": getattr(agent, "frozen_system_prompt", None),
                        "trainable_system_prompt": getattr(agent, "trainable_system_prompt", None),
                        "root_ancestor_class": getattr(agent, "root_ancestor_class", None),
                        "father_agent_id": getattr(agent, "father_agent_id", None),
                        "father_agent_name": getattr(agent, "father_agent_name", None),
                        "parent_agent_id": getattr(agent, "parent_agent_id", None),
                        "parent_agent_name": getattr(agent, "parent_agent_name", None),
                        "spawn_method": getattr(agent, "spawn_method", None),
                        "tasks_lived": getattr(agent, "tasks_lived", None),
                    }
                    for agent in self.mas.population.get_all()
                ],
            }
            self._append_jsonl(self.population_metrics_path, population_snapshot)

class Evaluator(ABC):
    """Generic evaluation scaffold shared by adapter runtimes.

    Args:
        checkpoint_path: Path to the trained checkpoint to evaluate.
        verbose: Whether to emit verbose logs.
        hayek_config_overrides: Optional dict of HayekConfig fields to override from checkpoint.
    """
    def __init__(
        self,
        checkpoint_path: str,
        verbose: bool = True,
        hayek_config_overrides: Optional[Dict[str, Any]] = None,
        profile: str = "",
    ):
        """
        Initialize the evaluator.

        Args:
            checkpoint_path: Path to the trained model checkpoint
            verbose: Whether to print detailed logs
            hayek_config_overrides: Optional dict of HayekConfig fields to override from checkpoint
            profile: Experiment profile name, reflected in output dir
        """
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.hayek_config_overrides = hayek_config_overrides or {}
        self.profile = profile

        self.mas = None
        self.tasks: List[Any] = []
        self.results: List[dict] = []

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Return the domain name."""
        pass

    @abstractmethod
    def create_deserializer(self) -> Callable[[Dict], BaseAgent]:
        """
        Create and return a deserializer function for agents.

        Returns:
            A callable that takes a dict and returns a BaseAgent
        """
        pass

    @abstractmethod
    def load_tasks(self, data_files: Optional[List[str]], limit: Optional[int]) -> List[Any]:
        """Load evaluation tasks."""
        pass

    @abstractmethod
    def create_env(self, task: Any) -> BaseEnv:
        """Create an environment from a task."""
        pass

    @abstractmethod
    def check_success(self, env: BaseEnv, task: Any) -> bool:
        """Check if a task was successfully solved."""
        pass

    def get_task_id(self, task: Any) -> str:
        """Get a string identifier for a task."""
        if hasattr(task, "id"):
            return task.id
        return str(task)

    def setup(self, data_files: Optional[List[str]] = None, limit: Optional[int] = None):
        """Set up the evaluator before running evaluation.

        Args:
            data_files: Optional evaluation data file paths.
            limit: Optional cap on the number of loaded tasks.
        """
        from hayekmas.base.mas import HayekMAS

        # Configure logging
        logger.configure(verbose=self.verbose, log_dir="logs", profile=self.profile)

        self._print_header()

        # Load tasks first
        logger.log("=" * 70)
        logger.log("📂 LOADING EVALUATION DATA")
        logger.log("=" * 70)
        self.tasks = self.load_tasks(data_files, limit)
        logger.log(f"\n✅ Loaded {len(self.tasks)} tasks")

        # Load model
        logger.log("\n" + "=" * 70)
        logger.log("📦 LOADING MODEL")
        logger.log("=" * 70)

        checkpoint = Path(self.checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}\n"
                "Train first with: python main.py -c <train-config.json>"
            )

        logger.log(f"\n   Loading: {checkpoint}")
        deserializer = self.create_deserializer()
        self.mas = HayekMAS.load(str(checkpoint), agent_deserializer=deserializer)

        # Apply config overrides from hayek_config_overrides
        for key, value in self.hayek_config_overrides.items():
            if hasattr(self.mas, key):
                old_value = getattr(self.mas, key)
                setattr(self.mas, key, value)
                if hasattr(self.mas.config.engine, key):
                    setattr(self.mas.config.engine, key, value)
                logger.log(f"   Overriding {key}: {old_value} → {value}")

        # Store training history for reference
        training_episodes = self.mas.episode_count
        training_rewards = self.mas.total_rewards
        training_bankruptcies = self.mas.bankruptcy_count
        
        # Display training history
        logger.log(f"\n📊 Training History:")
        logger.log(f"   Episodes trained: {training_episodes}")
        logger.log(f"   Total rewards earned: ${training_rewards:.2f}")
        logger.log(f"   Agent bankruptcies: {training_bankruptcies}")
        
        # Reset episode counter for evaluation (eval episodes start from 1)
        # Note: In eval mode, wealth/rewards are FROZEN - no updates occur
        self.mas.episode_count = 0
        
        # Display loaded agents with details
        logger.log(f"\n✅ Loaded {len(self.mas.population)} agents:")
        logger.log(f"{'─'*70}")
        for i, agent in enumerate(self.mas.population.get_all(), 1):
            bid_str = f"${agent.get_bid():.2f}" if agent.get_bid() is not None else "None"
            agent_type = type(agent).__name__
            status = getattr(agent, 'status', 'N/A')
            
            logger.log(f"\n   Agent {i}: {agent.name}")
            logger.log(f"      Type: {agent_type}")
            logger.log(f"      Status: {status}")
            logger.log(f"      Wealth: ${agent.wealth:.2f}")
            logger.log(f"      Bid: {bid_str}")
        
        logger.log(f"\n{'─'*70}")

    def evaluate(
        self,
        data_files: Optional[List[str]] = None,
        limit: Optional[int] = None,
        max_workers: int = 50,
    ) -> dict:
        """Run evaluation on all tasks.

        Args:
            data_files: Optional evaluation data file paths.
            limit: Optional cap on the number of evaluated tasks.
            max_workers: Max worker count for parallel evaluation.

        Returns:
            A summary dictionary with aggregate evaluation metrics.
        """
        if self.mas is None:
            self.setup(data_files, limit)

        logger.print_mode_banner("EVAL", f"EVALUATION BEGINS — {self.domain_name}")
        n_tasks = len(self.tasks)
        logger.log(f"\nEvaluating on {n_tasks} tasks (parallel)...\n")

        self.mas.eval()
        self.results = []
        if n_tasks <= 1:
            return self._evaluate_sequential()

        lock = threading.Lock()
        indexed_results: List[Optional[dict]] = [None] * n_tasks
        success_count = [0]

        def _eval_one(idx_task):
            idx, task = idx_task
            task_id = self.get_task_id(task)
            env = self.create_env(task)
            env.initialize()
            self.mas._run_auction_action_loop(env)
            success = self.check_success(env, task)
            result = {
                "id": task_id,
                "success": success,
                "steps": env.step_count,
                "final_state": env.get_state_description(),
            }
            with lock:
                indexed_results[idx] = result
                if success:
                    success_count[0] += 1

        with ThreadPoolExecutor(max_workers=min(n_tasks, max_workers)) as pool:
            list(pool.map(_eval_one, enumerate(self.tasks)))

        self.results = [r for r in indexed_results if r is not None]
        self.mas.episode_count += n_tasks
        self._print_results(success_count[0], n_tasks)

        return {
            "domain": self.domain_name,
            "total": n_tasks,
            "success": success_count[0],
            "accuracy": success_count[0] / n_tasks if n_tasks else 0,
        }

    def _evaluate_sequential(self) -> dict:
        self.results = []
        success_count = 0
        for task in self.tasks:
            task_id = self.get_task_id(task)
            env = self.create_env(task)
            self.mas.run_one_episode(env)
            success = self.check_success(env, task)
            if success:
                success_count += 1
            self.results.append({
                "id": task_id,
                "success": success,
                "steps": env.step_count,
                "final_state": env.get_state_description(),
            })
        self._print_results(success_count, len(self.tasks))
        return {
            "domain": self.domain_name,
            "total": len(self.tasks),
            "success": success_count,
            "accuracy": success_count / len(self.tasks) if self.tasks else 0,
        }

    def _print_header(self):
        """Print evaluation header."""
        logger.print_mode_banner("EVAL", f"HAYEK MACHINE: {self.domain_name} Evaluation")

    def _print_results(self, success: int, total: int):
        """Print evaluation results."""
        logger.print_mode_banner("EVAL", "EVALUATION COMPLETE")

        rate = 100 * success / total if total > 0 else 0
        logger.log(f"\n📊 RESULTS:")
        logger.log(f"   Tasks solved: {success}/{total} ({rate:.1f}%)")
        logger.log(f"   Accuracy: {rate:.1f}%")

        if self.mas:
            logger.log(f"\n📈 EVALUATION STATISTICS:")
            logger.log(f"   Tasks evaluated: {self.mas.episode_count}")
            logger.log(f"   🔒 Frozen mode: No wealth updates, no agent changes")

            # Show failed tasks
            failed = [r for r in self.results if not r["success"]]
            if failed:
                logger.log(f"\n❌ FAILED TASKS ({len(failed)}):")
                for r in failed[:5]:
                    logger.log(f"   • {r['id']}: {r['final_state']}")
                if len(failed) > 5:
                    logger.log(f"   ... and {len(failed) - 5} more")

            # Show final agent state (unchanged from start since evaluation is frozen)
            logger.log(f"\n💼 FINAL AGENT STATE (Unchanged - Evaluation Mode):")
            self.mas.print_population()

        logger.log("\n")