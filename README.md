<h1 align="center">Economy of Minds: Emerging Multi-Agent Intelligence with Economic Interactions</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2606.02859"><img src="https://img.shields.io/badge/arXiv-2606.02859-b31b1b.svg" alt="arXiv"></a>
  <a href="https://arxiv.org/pdf/2606.02859"><img src="https://img.shields.io/badge/pdf-2606.02859-b31b1b.svg" alt="pdf"></a>
  <img src="https://img.shields.io/badge/python->=3.10-blue.svg" alt="python">
  <a href="https://zhentingqi.github.io/internal/projects/EoM/"><img src="https://img.shields.io/badge/project-webpage-brightgreen.svg" alt="project webpage"></a>
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="license">
</p>

## Overview

How can a population of agents self-orchestrate and self-adapt into stronger
collective intelligence **without centralized control**? Standard approaches
introduce a central orchestrator to create agents, assign specializations, and
coordinate actions — but this bottlenecks planning at a single coordination
gate and makes learning increasingly inefficient as the system scales.

Inspired by Friedrich Hayek's theory that **prices are a decentralized
coordination signal**, we present **Economy of Minds (EoM)**, a system in which
a population of agents compete via *auctions* for the right to act, exchange
payments through *peer-to-peer transactions*, and accumulate or lose wealth
based on environmental rewards. These simple economic signals induce
decentralized credit assignment and drive planning without global orchestration
or explicit communication protocols. The population *evolves through economic
selection*: agents that consistently contribute to successful trajectories
accumulate wealth and are mutated via **exploitation**, while ineffective agents
go bankrupt and are replaced via **exploration**. Initialized with weak agents,
the economy produces emergent multi-step reasoning strategies and outperforms
stronger monolithic baselines across five agentic tasks — mathematical
reasoning, financial research, scientific research, accelerator design, and
distributed-system optimization.



https://github.com/user-attachments/assets/bf8a5819-75aa-4452-836e-8f6b71967677



## Repository

The repository is organized around a small core engine and domain adapters:

- `main.py`: the global launcher. It reads one JSON config and dispatches to
  the configured adapter runtime.
- `global_configs/`: top-level run configs for the available domains.
- `hayekmas/base/`: the generic auction, training, evaluation, population,
  reward, and configuration machinery.
- `hayekmas/adapters/`: domain-specific environments, agents, prompts, and
  runtime config loaders.
- `hayekmas/utils/`: logging, LLM client construction, data, and visualization
  helpers.
- `third_party/benchmarks/`: benchmark assets used by the adapters.

## Supported Adapters

- `arch_dse_world`: accelerator design-space exploration for ResNet-50 layer
  mapping on a Gemmini-style systolic array. Config:
  `global_configs/train_arch_dse_world.json`.
- `cloudcast`: code evolution for a multi-cloud broadcast routing program.
  Config: `global_configs/train_cloudcast.json`.
- `researchworld`: scientific research reasoning with a rubric-based LLM
  judge. Config: `global_configs/train_research.json`.

## Execution Flow

1. `main.py` loads a JSON config from `global_configs/`.
2. The config's `domain` key selects an adapter runtime.
3. The adapter runtime deep-merges its adapter config, if present, with the
   global config.
4. The runtime builds the configured LLM client, environment, agents, and EoM
   engine.
5. Training or evaluation runs according to the config's `mode`.

## Installation

For general installation:

```bash
pip install -e ".[litellm]"
cd third_party/smolagents
pip install -e ".[toolkit]"
```

Please go to [CloudCast](#cloudcast), [Researchworld](#researchworld), and
[Arch DSE World](#arch-dse-world) for domain-specific installations.

## Running

Run a config through the global launcher:

```bash
python main.py global_configs/train_cloudcast.json
python main.py global_configs/train_research.json
python main.py global_configs/train_arch_dse_world.json
```

Model/API selection lives in each config's `model` section. Common choices:

- `litellm`: cloud APIs and OpenAI-compatible gateways.
- `localhost`: a local OpenAI-compatible server such as vLLM or SGLang.
- `together`: Together AI chat completions.
- `demo`: deterministic toy client for lightweight local checks.
- `vllm` / `sglang`: direct local inference backends.

For `litellm`, set provider-specific environment variables such as
`OPENAI_API_KEY`, or use an OpenAI-compatible gateway with:

```bash
export OPENAI_BASE_URL=<your-base-url>
export OPENAI_API_KEY=<your-key>
```

For `localhost`, set `model.api_base` in the config or:

```bash
export LOCALHOST_BASE_URL=http://127.0.0.1:8000/v1
export LOCALHOST_API_KEY=not-needed
```

If `model.name` is empty, the localhost client attempts to detect the model
from `/v1/models`.

## CloudCast

The `cloudcast` adapter is a code-evolution task. A society of agents edits
a single Python file, `initial_program.py`, that defines a multi-cloud
broadcast routing algorithm. The verifier runs the program on five inter-
and intra-cloud scenarios using a Skyplane-derived cost and throughput
grid and returns the total egress cost; the score is
`max(0, 1 - cost / 1035)`, where `1035` is the cost of the Dijkstra
single-path seed. The workspace persists across episodes.

The task is from [ADRS](https://arxiv.org/pdf/2510.06189).

### Roles

Six fixed roles, defined in `agent.py`:

- `PlannerCloudcastAgent` (`planner`) — proposes the next sub-goal.
- `ReaderCloudcastAgent` (`reader`) — reads files in the workspace.
- `ImplementerCloudcastAgent` (`implementer`) — edits `initial_program.py`.
- `BuilderCloudcastAgent` (`builder`) — runs build / import checks.
- `EvaluatorCloudcastAgent` (`evaluator`) — calls the verifier mid-episode.
- `FinalizerCloudcastAgent` (`finalizer`) — submits the program with `final_answer`.

The auction selects one acting role per step. The last
`mas.terminal.start_on_step_from_end` steps of an episode are restricted
to agents carrying the tags in `mas.terminal.candidate_agent_tags`
(`["terminal"]` by default — only `Finalizer` qualifies).

### Files

- `hayekmas/adapters/cloudcast/` — adapter code (`agent.py`, `env.py`,
  `runtime.py`, `prompts.py`, `tools.py`, `task.py`).
- `third_party/benchmarks/cloudcast-broadcast-opt/` — task directory:
  `instruction.md`, `environment/initial_program.py` with the EVOLVE
  block, `environment/profiles/` cost and throughput grids, and the
  verifier under `tests/`. Runs offline.

Configuration in `hayekmas/adapters/cloudcast/configs/train.json`:

- `run.preserve_workspace_across_episodes` — keep the edited program
  across episodes.
- `run.num_episodes`, `run.max_steps` — episode and step budgets.
- `mas.engine.{min_num_agents, max_num_agents}` — population bounds.
- `mas.terminal.{enabled, start_on_step_from_end, candidate_agent_tags}`
  — restrict the final steps of an episode to the tagged agents.
- `mas.reward.{regression_multiplier, broken_program_penalty,
  path_reward_per_unique_author}` — reward shaping specific to this
  adapter.

## Frontier-Science-Research

The `researchworld` adapter uses tasks from OpenAI's [FrontierScience-Research benchmark](https://openai.com/index/frontierscience/), which targets scientific research reasoning in physics, chemistry and biology.
There are no external tools and the answer is graded by a rubric-based LLM judge.

### Roles

The `researchworld` also uses five specialized agents. All five are defined in `agent.py`; each has a `FROZEN_SYSTEM_PROMPT`(role identity, never mutated) and a `TRAINABLE_SYSTEM_PROMPT`
(strategy, evolved by the Hayek birth loop):

- `LiteratureResearchAgent` (`literature`) — surfaces definitions / theorems / standard formulas; no new derivation.
- `PlannerResearchAgent` (`planner`) — outlines the sub-parts (a), (b), (c)… and the tactic for each.
- `DeriverResearchAgent` (`deriver`) — the workhorse: one concrete derivation/calculation per turn.
- `VerifierResearchAgent` (`verifier`) — sanity-checks the latest contribution (signs, units, limits).
- `AnswerResearchAgent` (`answer`) — emit `<final_answer>…</final_answer>`; emitting it terminates the episode.

Wakeup rules: an agent never acts twice in the same role back-to-back; `literature`/`planner` may self-start an empty episode; `answer` only wakes after a `deriver`/`verifier` turn.


### Rubric reward

The LLM-judger reads the problem, rubric, and candidate answer and replies with `SCORE:` (clamped to`[0, 1]`) and `REASON:`. A task passes when `score >= judge.threshold` (default `0.7`). 

The same model that drives the agents also acts as the judge (`env.llm_fn`). 

Researchworld responsibilities are split by concern:

- `agent.py`: the five research agents, `ResearchAction`, and birth/serialization logic
- `env.py`: `ResearchEnv`, JSONL task loading, and the rubric-based LLM judge
- `runtime.py`: two-layer config parsing, train/eval entrypoints, and periodic-test logic

## Arch DSE World

The `arch_dse_world` adapter runs accelerator design-space exploration: a
ResNet-50 mapping search on a Gemmini-style systolic array, evaluated by
Timeloop + Accelergy (the DOSA paper's pipeline). Relevant files:

- `hayekmas/adapters/arch_dse_world/` holds the adapter code (agent, env,
  runtime) and the bundled simulator helper.
- `hayekmas/adapters/arch_dse_world/simulator/` holds the workspace template
  and ResNet-50 workload YAMLs.
- `hayekmas/adapters/arch_dse_world/configs/` holds adapter-level configs.
- `scripts/arch_dse_world/setup_arch_dse_simulator.sh` installs the Timeloop +
  Accelergy + DOSA simulator backend.
- `scripts/arch_dse_world/dosa_bounded_edps.json` stores cached DOSA baseline
  values.
- `scripts/arch_dse_world/launch_24jobs_perlayer.sh` is the optional per-layer
  launcher for SLURM-style cluster runs.

### Installation

This domain needs two external pieces because the reward comes from a real
hardware simulator.

First, configure an LLM. The default global config uses Together AI:

```bash
export TOGETHER_API_KEY=...
```

For practical throughput, you can instead serve the model yourself with vLLM or
SGLang and set `model.api` to `localhost`, with `LOCALHOST_BASE_URL` or
`model.api_base` pointing at your server. `litellm` with `OPENAI_BASE_URL` and
`OPENAI_API_KEY` also works. In every case, `model.name` must match the exact
model id your backend serves.

Second, install the simulator backend:

```bash
bash scripts/arch_dse_world/setup_arch_dse_simulator.sh
```

The script creates a self-contained conda environment, clones DOSA, builds
Timeloop + Accelergy, and prints the environment variables to export. A run
then looks like:

```bash
export TOGETHER_API_KEY=...
export DSE_CONDA_ENV=/path/to/arch_dse_sim   # printed by the setup script
export DOSA_ROOT=/path/to/dosa               # printed by the setup script
export ARCHGYM_SCRATCH="$(mktemp -d)"
python main.py global_configs/train_arch_dse_world.json
```

Gurobi is not required to run `arch_dse_world`: the eval path only runs
Timeloop on a given hardware/mapping pair. Gurobi is only needed if you
regenerate DOSA's own mapping-search baseline; cached baseline values are in
`scripts/arch_dse_world/dosa_bounded_edps.json`.

For cluster-scale per-layer experiments, use
`scripts/arch_dse_world/launch_24jobs_perlayer.sh` as the starting point and
override the environment variables it documents for your scheduler setup.

## Citation

```bibtex
@misc{qi2026economymindsemergingmultiagent,
      title={Economy of Minds: Emerging Multi-Agent Intelligence with Economic Interactions}, 
      author={Zhenting Qi and Huangyuan Su and Ao Qu and Chenyu Wang and Yu Yao and Han Zheng and Kushal Chattopadhyay and Guowei Xu and Zihan Wang and Weirui Ye and Vijay Janapa Reddi and Ju Li and Paul Pu Liang and Himabindu Lakkaraju and Sham Kakade and Yilun Du},
      year={2026},
      eprint={2606.02859},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.02859}, 
}
```

## License

MIT
