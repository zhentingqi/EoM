"""
arch_dse_world adapter runtime.

Multi-layer round-robin training. The agent population persists across
tasks: wealth, experience strings, and per-agent notebooks all carry over
from one layer to the next.
"""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hayekmas.adapters.arch_dse_world.agent import (
    DSEAgentBase,
    HistorianDSEAgent,
    PlannerDSEAgent,
    ExecutorDSEAgent,
    DSE_AGENT_CLASSES,
    build_starter_population,
    deserialize_dse_agent,
)
from hayekmas.adapters.arch_dse_world.env import (
    DSEEnv,
    DSETask,
    load_dse_tasks_from_jsonl,
    load_resnet50_unique_layers,
    DEFAULT_RESNET50_DIR,
    WORKLOADS_DIR,
)
from hayekmas.base.agent import BaseAgent
from hayekmas.base.config import DEFAULT_HAYEK_CONFIG, HayekConfig
from hayekmas.base.mas import HayekMAS
from hayekmas.base.pipeline import Trainer, Evaluator
from hayekmas.utils.llm import LLMClient, get_llm_client
from hayekmas.utils.logger import logger


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DSELLMConfig:
    api: str = "localhost"
    name: str = ""
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    api_base: Optional[str] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DSESplitConfig:
    workload: str = "resnet50"
    layers: Optional[List[str]] = None         # explicit layer names
    files: Optional[List[str]] = None          # OR jsonl files of tasks
    max_tasks: Optional[int] = None
    shuffle: bool = False


@dataclass
class DSEDataConfig:
    train: DSESplitConfig = field(default_factory=DSESplitConfig)
    test: DSESplitConfig = field(default_factory=DSESplitConfig)


@dataclass
class DSERunConfig:
    num_epochs: int = 1
    verbose: bool = True
    output: str = "checkpoints/hayek_dse.json"
    checkpoint: Optional[str] = None
    checkpoint_steps: Optional[str] = None
    resume_run_dir: Optional[str] = None
    resume_from_task: Optional[int] = None
    max_eval_workers: int = 1
    n_historian: int = 3
    n_planner: int = 3
    n_executor: int = 3
    budget_per_task: int = 10
    # Worker-subset support: each parallel worker handles a subset of layers.
    # If None / 1, all layers go to this single worker (no parallelism).
    worker_id: int = 0      # 0..worker_count-1
    worker_count: int = 1   # total parallel workers
    sessions_per_layer: int = 5  # each layer becomes N sequential tasks
    # Optional path to a JSON file mapping role → list of seeded experience
    # priors. Each starter agent in role R gets one prior, rotating. Empty
    # list / missing key → empty experience (current default behavior).
    experience_priors_path: Optional[str] = None
    # If True: prepend an [EXPLORER] hint to slot-0 of each role and an
    # [EXPLOITER] hint to the rest. Combines with experience_priors (the
    # hint sits above the prior in the prompt).
    role_mode_split: bool = False
    # Number of leading sessions per layer to run as Executor-only
    # ("preheat"). Historian and Planner do not wake during these
    # sessions, so the cold-start is a pure single-agent search that
    # builds an informed submit history before the three-role auction
    # kicks in. 0 = no preheat (default).
    preheat_until_session: int = 0
    # Optional list of cached single-agent baseline output roots. For
    # each layer, the env searches these roots in order for
    #   <root>/<layer_id>/workspace/{history.jsonl, best.json}
    # and copies the first match into the workspace before the first
    # session starts. The three-role population thus boots already aware
    # of every baseline submit + simulator result. Costs 0 budget; bounds
    # worst-case at single-agent ReAct quality.
    baseline_seed_dirs: Optional[List[str]] = None


@dataclass
class DSERuntimeConfig:
    domain: str = "arch_dse_world"
    mode: str = "train"
    profile: str = ""
    llm: DSELLMConfig = field(default_factory=DSELLMConfig)
    data: DSEDataConfig = field(default_factory=DSEDataConfig)
    run: DSERunConfig = field(default_factory=DSERunConfig)
    mas: HayekConfig = field(default_factory=lambda: deepcopy(DEFAULT_HAYEK_CONFIG))


_META_KEYS = frozenset({"adapter_config", "profile", "domain"})


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def configure_dse_hayek_config(config: HayekConfig) -> HayekConfig:
    config.terminal.candidate_agent_tags = ("terminal",)
    return config


def load_dse_runtime_config(raw: Dict[str, Any]) -> DSERuntimeConfig:
    if raw.get("domain") != "arch_dse_world":
        raise ValueError(f"Unsupported domain: {raw.get('domain')}")
    profile = raw.get("profile")
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("profile must be a non-empty string")
    profile = profile.strip()

    adapter_path = raw.get("adapter_config")
    if adapter_path:
        with open(Path(adapter_path), "r", encoding="utf-8") as f:
            section = json.load(f)
    else:
        section = {}
    overlay = {k: v for k, v in raw.items() if k not in _META_KEYS}
    merged = _deep_merge(section, overlay)

    data_raw = merged.get("data", {})
    train_raw = data_raw.get("train", {})
    test_raw = data_raw.get("test", {})
    run_raw = merged.get("run", {})
    mas_raw = merged.get("mas", {})
    model_raw = merged.get("model", {})
    extra_kwargs = {k: v for k, v in model_raw.items()
                    if k not in {"api", "name", "api_key", "api_key_env", "api_base"}}

    return DSERuntimeConfig(
        domain="arch_dse_world",
        mode=merged.get("mode", "train"),
        profile=profile,
        llm=DSELLMConfig(
            api=model_raw.get("api", "localhost"),
            name=model_raw.get("name", ""),
            api_key=model_raw.get("api_key"),
            api_key_env=model_raw.get("api_key_env"),
            api_base=model_raw.get("api_base"),
            extra_kwargs=extra_kwargs,
        ),
        data=DSEDataConfig(
            train=DSESplitConfig(
                workload=train_raw.get("workload", "resnet50"),
                layers=train_raw.get("layers"),
                files=train_raw.get("files"),
                max_tasks=train_raw.get("max_tasks"),
                shuffle=train_raw.get("shuffle", False),
            ),
            test=DSESplitConfig(
                workload=test_raw.get("workload", "resnet50"),
                layers=test_raw.get("layers"),
                files=test_raw.get("files"),
                max_tasks=test_raw.get("max_tasks"),
                shuffle=test_raw.get("shuffle", False),
            ),
        ),
        run=DSERunConfig(
            num_epochs=run_raw.get("num_epochs", 1),
            verbose=run_raw.get("verbose", True),
            output=run_raw.get("output", "checkpoints/hayek_dse.json"),
            checkpoint=run_raw.get("checkpoint"),
            checkpoint_steps=run_raw.get("checkpoint_steps"),
            resume_run_dir=run_raw.get("resume_run_dir"),
            resume_from_task=run_raw.get("resume_from_task"),
            max_eval_workers=run_raw.get("max_eval_workers", 1),
            n_historian=int(run_raw.get("n_historian", 3)),
            n_planner=int(run_raw.get("n_planner", 3)),
            n_executor=int(run_raw.get("n_executor", 3)),
            budget_per_task=int(run_raw.get("budget_per_task", 10)),
            worker_id=int(run_raw.get("worker_id", 0)),
            worker_count=int(run_raw.get("worker_count", 1)),
            sessions_per_layer=int(run_raw.get("sessions_per_layer", 5)),
            experience_priors_path=run_raw.get("experience_priors_path"),
            role_mode_split=bool(run_raw.get("role_mode_split", False)),
            preheat_until_session=int(run_raw.get("preheat_until_session", 0)),
            baseline_seed_dirs=run_raw.get("baseline_seed_dirs"),
        ),
        mas=configure_dse_hayek_config(HayekConfig.from_dict(mas_raw)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# FACTORIES
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_tasks(split: DSESplitConfig, *, budget: int,
                   worker_id: int = 0, worker_count: int = 1,
                   sessions_per_layer: int = 1) -> List[DSETask]:
    """Build tasks list. With sessions_per_layer > 1, each layer becomes N
    consecutive tasks (per-layer cross-session). With worker_count > 1,
    layers are sliced contiguously among workers."""
    if split.layers:
        layers = split.layers
    elif split.files:
        # Defer to jsonl loader (no worker-subset for jsonl mode currently)
        return load_dse_tasks_from_jsonl([Path(p) for p in split.files], limit=split.max_tasks)
    else:
        try:
            from hayekmas.adapters.arch_dse_world.env import load_resnet50_unique_layers
            layers = load_resnet50_unique_layers()
        except FileNotFoundError:
            layers = [p.stem for p in sorted((WORKLOADS_DIR / split.workload).glob("*.yaml"))
                      if not p.stem.endswith("_layers") and not p.stem.endswith("_count")
                      and p.stem != "example"]

    # Worker-subset: contiguous chunk of layers
    if worker_count > 1:
        n = len(layers)
        chunk = (n + worker_count - 1) // worker_count
        start = worker_id * chunk
        end = min(start + chunk, n)
        layers = layers[start:end]

    # Expand sessions: each layer becomes N sequential tasks with unique
    # session_idx so workspace dirs don't collide.
    tasks: List[DSETask] = []
    for layer in layers:
        for s in range(max(1, sessions_per_layer)):
            tasks.append(DSETask.from_layer(layer, workload=split.workload,
                                            budget=budget, session_idx=s))
    if split.max_tasks is not None:
        tasks = tasks[: split.max_tasks]
    return tasks


# ═══════════════════════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════════════════════

class DSETrainer(Trainer):

    def __init__(self, llm_client: LLMClient, config: DSERuntimeConfig):
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
        self._notebook_root: Optional[Path] = None

    @property
    def domain_name(self) -> str:
        return "DSE"

    def load_tasks(self, data_files, limit):
        split = self.runtime_config.data.train
        if data_files:
            split = DSESplitConfig(workload=split.workload, files=data_files,
                                   max_tasks=split.max_tasks)
        return _resolve_tasks(
            split,
            budget=self.runtime_config.run.budget_per_task,
            worker_id=self.runtime_config.run.worker_id,
            worker_count=self.runtime_config.run.worker_count,
            sessions_per_layer=self.runtime_config.run.sessions_per_layer,
        )

    def create_agents(self):
        priors = self._resolve_experience_priors()
        role_mode_split = self.runtime_config.run.role_mode_split
        if role_mode_split:
            logger.log("role_mode_split=True: slot-0 agent in each role gets EXPLORER bias, rest get EXPLOITER")
        return build_starter_population(
            backbone_llm=self.llm_client.as_callable(),
            initial_wealth=self.hayek_config.engine.initial_wealth,
            n_historian=self.runtime_config.run.n_historian,
            n_planner=self.runtime_config.run.n_planner,
            n_executor=self.runtime_config.run.n_executor,
            notebook_dir=self._notebook_root,
            logger=logger,
            experience_priors=priors,
            role_mode_split=role_mode_split,
        )

    def _resolve_experience_priors(self) -> Optional[Dict[str, List[str]]]:
        """Load the priors JSON. Two supported formats:
          (1) flat {"historian":[...], "planner":[...], "executor":[...]}
              applies the same priors to ALL workers (legacy).
          (2) per-layer {"_meta":{...}, "layers": {layer_name: {role: [...]}}}
              this worker's layer is determined from worker_id +
              worker_count (matches the SLURM array layout where
              worker_count = #layers, one worker per kernel).
        Returns None if no priors path configured.
        """
        path = self.runtime_config.run.experience_priors_path
        if not path:
            return None
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        # Flat (legacy) form
        if "layers" not in doc and any(k in doc for k in ("historian","planner","executor")):
            logger.log(f"priors (flat): {path} → {[k for k in ('historian','planner','executor') if k in doc]}")
            return doc

        # Per-layer form. CRITICAL: must use the same layer list +
        # slicing as _resolve_tasks() — otherwise the priors loaded for
        # this worker describe a DIFFERENT layer than the one this
        # worker actually processes. The adapter config can override
        # the layer order via data.train.layers, and _resolve_tasks
        # honors that override; we must too.
        layers_table = doc.get("layers", {})
        wid = self.runtime_config.run.worker_id
        wcount = self.runtime_config.run.worker_count
        # Use the same source as _resolve_tasks: cfg.data.train.layers
        # if set, else fall back to the YAML manifest.
        cfg_layers = self.runtime_config.data.train.layers
        if cfg_layers:
            all_layers = list(cfg_layers)
        else:
            try:
                all_layers = load_resnet50_unique_layers()
            except FileNotFoundError:
                all_layers = []
        # SLURM-array layout: chunk(layers, wcount)[wid][0] — same as _resolve_tasks
        if all_layers and wcount >= 1:
            n = len(all_layers)
            chunk = (n + wcount - 1) // wcount
            start = wid * chunk
            end = min(start + chunk, n)
            my_layers = all_layers[start:end]
            assigned = my_layers[0] if my_layers else None
        else:
            assigned = None
        if assigned and assigned in layers_table:
            logger.log(
                f"priors (per-layer): {path}; worker_id={wid}/{wcount} → "
                f"layer={assigned}; "
                f"counts={ {r: len(layers_table[assigned].get(r, [])) for r in ('historian','planner','executor')} }"
            )
            return layers_table[assigned]
        logger.log(
            f"WARN: priors path={path} loaded but no layer entry for worker_id={wid} "
            f"(assigned={assigned}); falling back to no priors"
        )
        return None

    def load_mas_from_checkpoint(self, checkpoint_path):
        return HayekMAS.load(
            checkpoint_path,
            agent_deserializer=self._make_deserializer(),
        )

    def _make_deserializer(self):
        def deserialize(data):
            return deserialize_dse_agent(
                data,
                backbone_llm=self.llm_client.as_callable(),
                logger=logger,
                notebook_dir=self._notebook_root,
            )
        return deserialize

    # ─── role-aware births ──────────────────────────────────────────────────

    def _agents_of_same_role(self, role: str) -> List[DSEAgentBase]:
        if self.mas is None:
            return []
        return [
            a for a in self.mas.population.get_all()
            if isinstance(a, DSEAgentBase) and a.ROLE == role
        ]

    def create_agent_factory_good_birth(self):
        backbone_llm = self.llm_client.as_callable()
        wealth = self.hayek_config.engine.initial_wealth

        def birth_good_agent(parent):
            cls = type(parent)
            siblings = self._agents_of_same_role(parent.ROLE)
            return cls.good_birth(
                parent=parent,
                all_role_agents=siblings,
                backbone_llm=backbone_llm,
                initial_wealth=wealth,
            )
        return birth_good_agent

    def create_agent_factory_bad_birth(self):
        backbone_llm = self.llm_client.as_callable()
        wealth = self.hayek_config.engine.initial_wealth

        def birth_bad_agent(source_agent, task_description="", correct_answer="", failure_trace=""):
            cls = type(source_agent)
            return cls.bad_birth(
                source_agent=source_agent,
                backbone_llm=backbone_llm,
                initial_wealth=wealth,
            )
        return birth_bad_agent

    # ─── env construction with per-task workspace dir ──────────────────────

    def get_task_id(self, task) -> str:
        """Override base class — include session_idx so 5 sessions of same
        layer don't collide in per_task_log/<id>.log or training_metrics rows."""
        return getattr(task, "session_id", task.id)

    def create_env(self, task):
        if self.outputs_dir is None:
            raise RuntimeError("outputs_dir not initialized — setup() must run first.")
        # All 5 sessions of one layer share the SAME workspace dir → cumulative
        # history.jsonl + best.json persist across sessions. Cross-session
        # records monotonically improve. Per-session breakdown is recoverable
        # from history.jsonl call_index ranges (1-10, 11-20, ..., 41-50).
        task_dir = self.outputs_dir / "tasks" / f"{task.workload}_{task.id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        return DSEEnv(
            task=task,
            out_dir=task_dir,
            reward_config=self.hayek_config.reward,
            max_steps=self.hayek_config.engine.max_steps_per_episode,
            preheat_until_session=self.runtime_config.run.preheat_until_session,
            baseline_seed_dirs=self.runtime_config.run.baseline_seed_dirs,
        )

    def check_success(self, env, task):
        if env.best_edp is None:
            return False
        return env.best_edp > 0

    # ─── notebook dir wiring ────────────────────────────────────────────────

    def _ensure_notebook_dirs(self):
        if self.outputs_dir is None:
            return
        self._notebook_root = self.outputs_dir / "agent_notebooks"
        self._notebook_root.mkdir(parents=True, exist_ok=True)
        if self.mas is not None:
            for a in self.mas.population.get_all():
                if isinstance(a, DSEAgentBase):
                    a.notebook_dir = self._notebook_root
                    a.notebook_dir.mkdir(parents=True, exist_ok=True)


# Override `setup` to wire notebook dirs after outputs_dir is created.
_orig_setup = Trainer.setup
def _dse_setup(self, *args, **kwargs):
    _orig_setup(self, *args, **kwargs)
    if isinstance(self, DSETrainer):
        self._ensure_notebook_dirs()
DSETrainer.setup = _dse_setup  # type: ignore[assignment]


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATOR (skeleton; same env, no learning)
# ═══════════════════════════════════════════════════════════════════════════

class DSEEvaluator(Evaluator):

    def __init__(self, llm_client: LLMClient, config: DSERuntimeConfig):
        super().__init__(
            checkpoint_path=config.run.checkpoint or config.run.output,
            verbose=config.run.verbose,
            hayek_config_overrides={
                "max_steps_per_episode": config.mas.engine.max_steps_per_episode,
            },
            profile=config.profile,
        )
        self.runtime_config = config
        self.llm_client = llm_client

    @property
    def domain_name(self) -> str:
        return "DSE"

    def create_deserializer(self):
        def deserialize(data):
            return deserialize_dse_agent(
                data,
                backbone_llm=self.llm_client.as_callable(),
                logger=logger,
                notebook_dir=None,
            )
        return deserialize

    def load_tasks(self, data_files, limit):
        split = self.runtime_config.data.test
        if data_files:
            split = DSESplitConfig(workload=split.workload, files=data_files,
                                   max_tasks=split.max_tasks)
        return _resolve_tasks(split, budget=self.runtime_config.run.budget_per_task)

    def create_env(self, task):
        out_dir = Path("outputs/dse_eval") / f"{task.workload}_{task.id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return DSEEnv(
            task=task,
            out_dir=out_dir,
            reward_config=self.mas.config.reward if self.mas is not None else None,
            max_steps=(self.mas.config.engine.max_steps_per_episode if self.mas else 200),
        )

    def check_success(self, env, task):
        return env.best_edp is not None


# ═══════════════════════════════════════════════════════════════════════════
# LLM CLIENT
# ═══════════════════════════════════════════════════════════════════════════

# Optional JSON file containing {"base_url": "<vllm-server>"}. Set
# GEMMA_ENDPOINT_JSON when a localhost model endpoint may move during a run,
# or put the URL directly in `model.api_base`.
_GEMMA_ENDPOINT_JSON_RAW = os.environ.get("GEMMA_ENDPOINT_JSON", "")
_GEMMA_ENDPOINT_JSON = (
    Path(_GEMMA_ENDPOINT_JSON_RAW) if _GEMMA_ENDPOINT_JSON_RAW else None
)


def _refresh_gemma_endpoint() -> Optional[str]:
    if _GEMMA_ENDPOINT_JSON is None or not _GEMMA_ENDPOINT_JSON.is_file():
        return None
    try:
        data = json.loads(_GEMMA_ENDPOINT_JSON.read_text())
        url = data.get("base_url", "")
        if isinstance(url, str) and url:
            return url.rstrip("/")
    except Exception:
        pass
    return None


def _wrap_with_endpoint_refresh(client: LLMClient) -> LLMClient:
    """Auto-retry on ConnectionError by re-resolving the Gemma endpoint."""
    original_generate = client.generate

    def generate_with_refresh(prompt, **kwargs):
        last_exc = None
        for attempt in range(4):
            try:
                return original_generate(prompt, **kwargs)
            except Exception as e:
                msg = str(e)
                connection_err = (
                    "Connection refused" in msg
                    or "Failed to establish a new connection" in msg
                    or "Max retries exceeded" in msg
                )
                if not connection_err:
                    raise
                last_exc = e
                new_url = _refresh_gemma_endpoint()
                if new_url and new_url != getattr(client, "api_base", "").rstrip("/"):
                    logger.log(
                        f"⚠️  LLM connection lost; switching to {new_url} "
                        f"(attempt {attempt + 1})", indent=1)
                    client.api_base = new_url
                time.sleep(min(60, 5 * (2 ** attempt)))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM call failed after retries")

    client.generate = generate_with_refresh  # type: ignore[method-assign]
    return client


def _build_llm_client(config: DSERuntimeConfig) -> LLMClient:
    logger.log("\n" + "=" * 70)
    logger.log("🤖 SETTING UP LLM BACKEND (DSE)")
    logger.log("=" * 70)
    logger.log(f"   API: {config.llm.api}")
    if config.llm.name:
        logger.log(f"   Model: {config.llm.name}")

    client_kwargs = dict(config.llm.extra_kwargs)
    fresh_url = _refresh_gemma_endpoint() if config.llm.api == "localhost" else None
    if fresh_url:
        logger.log(f"   🔄 Resolved Gemma endpoint: {fresh_url}")
        client_kwargs["api_base"] = fresh_url
    elif config.llm.api_base:
        client_kwargs["api_base"] = config.llm.api_base
    if config.llm.api_key_env:
        client_kwargs["api_key"] = os.getenv(config.llm.api_key_env)
    elif config.llm.api_key is not None:
        client_kwargs["api_key"] = config.llm.api_key

    client = get_llm_client(config.llm.api, model=config.llm.name, **client_kwargs)
    if config.llm.api == "localhost":
        client = _wrap_with_endpoint_refresh(client)
        logger.log("   🛡️  Endpoint-refresh wrapper enabled")
    return client


# ═══════════════════════════════════════════════════════════════════════════
# ENTRYPOINTS
# ═══════════════════════════════════════════════════════════════════════════

def run(config: DSERuntimeConfig) -> None:
    if config.mode not in {"train", "eval"}:
        raise ValueError(f"Unsupported mode: {config.mode}")

    llm_client = _build_llm_client(config)

    if config.mode == "train":
        trainer = DSETrainer(llm_client=llm_client, config=config)
        trainer.train(
            data_files=config.data.train.files,
            max_tasks=config.data.train.max_tasks,
            resume_run_dir=config.run.resume_run_dir,
            resume_from_task=config.run.resume_from_task,
        )
        trainer.save()
        logger.log("\n✨ DSE training complete!")
        logger.close()
        return

    evaluator = DSEEvaluator(llm_client=llm_client, config=config)
    eval_split = config.data.test if (config.data.test.layers or config.data.test.files) else config.data.train
    try:
        evaluator.evaluate(
            data_files=eval_split.files,
            limit=eval_split.max_tasks,
            max_workers=config.run.max_eval_workers,
        )
    except FileNotFoundError as exc:
        logger.log(f"\n❌ {exc}")
        return
    logger.close()


def main(raw_config: Dict[str, Any]) -> None:
    run(load_dse_runtime_config(raw_config))
