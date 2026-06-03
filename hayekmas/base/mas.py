from __future__ import annotations
from copy import deepcopy
from enum import Enum
import json
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from hayekmas.base.config import HayekConfig
from hayekmas.base.agent import BaseAgent, AgentStatus, set_agent_id_counter
from hayekmas.base.env import BaseEnv
from hayekmas.base.population import Population
from hayekmas.utils.logger import logger


class TerminationReason(Enum):
    """Reason why an episode ended."""
    NONE = "none"
    MAX_STEPS = "max_steps"
    NO_ACTIVE_AGENTS = "no_active_agents"
    GOAL_REACHED = "goal_reached"


class HayekMAS:
    """Core execution engine for the Hayek multi-agent economy.

    Args:
        config: Engine configuration (HayekConfig). All parameters are read from this.
    """
    def __init__(self, config: HayekConfig):
        self.config = deepcopy(config)

        eng = self.config.engine
        self.max_steps_per_episode = eng.max_steps_per_episode
        self.max_trials_per_episode = eng.max_trials_per_episode
        self.birth_interval = eng.birth_interval
        self.num_births_per_interval = eng.num_births_per_interval
        self.min_num_agents = eng.min_num_agents
        self.max_num_agents = eng.max_num_agents

        self.bid_scheme = eng.bid_scheme
        self.reward_scheme = self.config.reward.reward_scheme
        assert self.reward_scheme in ("path_reward_only", "path_reward_and_stepwise_reward"), f"reward_scheme must be 'path_reward_only' or 'path_reward_and_stepwise_reward', got '{self.reward_scheme}'"

        # Bid scheme constants (from config)
        self.base_bid: float = self.config.engine.base_bid
        self.novice_bid_epsilon: float = eng.novice_bid_epsilon
        self.holland_alpha: float = eng.holland_alpha
        self.tycoon_wealth_threshold: float = eng.tycoon_wealth_threshold

        # Population of agents
        self.population: Population = Population()

        # Logger reference (avoids the _log wrapper pattern)
        self.logger = logger

        # Mode flag (True = training, False = evaluation)
        self._training: Optional[bool] = None

        # for statistics
        self.episode_count = 0
        self.total_rewards = 0.0
        self.bankruptcy_count = 0
        self.last_termination_reason: TerminationReason = TerminationReason.NONE
        self.last_episode_metrics: Dict[str, Any] = {}

        # Factories for creating new agents during evolution.
        self._birth_good_agent_factory: Callable[[BaseAgent], BaseAgent] = None
        self._birth_bad_agent_factory: Callable[..., BaseAgent] = None

        # Initial agents kept as templates for replenishment via mutation
        self._initial_agents: List[BaseAgent] = []

    # ═══════════════════════════════════════════════════════════════════════
    # MODE SWITCHING (PyTorch-style)
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def training(self) -> bool:
        """Whether the MAS is in training mode."""
        return self._training

    def train(self) -> "HayekMAS":
        """Switch to training mode. Returns self for chaining."""
        self._training = True
        return self

    def eval(self) -> "HayekMAS":
        """Switch to evaluation mode. Returns self for chaining."""
        self._training = False
        return self

    def run_one_episode(self, env: BaseEnv, step_ckpt_save_path: Optional[str] = None):
        """Dispatch to _train_forward or _eval_forward based on current mode.
        If step_ckpt_save_path is set, step checkpoints are written there after each step."""
        if self._training:
            self._train_forward(env, step_ckpt_save_path=step_ckpt_save_path)
        else:
            self._eval_forward(env, step_ckpt_save_path=step_ckpt_save_path)

    # ═══════════════════════════════════════════════════════════════════════
    # POPULATION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════

    def set_agent_factory(
        self,
        birth_good_agent: Callable[[BaseAgent], BaseAgent],
        birth_bad_agent: Callable[..., BaseAgent],
    ):
        """Set the agent factories for evolution/spawning."""
        self._birth_good_agent_factory = birth_good_agent
        self._birth_bad_agent_factory = birth_bad_agent

    # ─── AGENT CREATION ───

    def birth_good_agent(self, parent: BaseAgent) -> Optional[BaseAgent]:
        """Create a more exploratory child from a strong surviving agent."""
        try:
            new_agent = self._birth_good_agent_factory(parent)
            new_agent.root_ancestor_class = getattr(parent, "root_ancestor_class", parent.__class__.__name__)
            new_agent.father_agent_id = getattr(parent, "father_agent_id", parent.id)
            new_agent.father_agent_name = getattr(parent, "father_agent_name", parent.name)
            new_agent.parent_agent_id = parent.id
            new_agent.parent_agent_name = parent.name
            new_agent.spawn_method = "good_birth"
            new_agent.tasks_lived = 0
            new_agent.bankruptcy_episode = None
            self.logger.log(
                f"🧬 SPAWNING: {new_agent.name} {new_agent.id} via GOOD BIRTH (parent: {parent.name})",
                indent=1, must_print=True,
            )
            return new_agent
        except Exception as e:
            self.logger.log(f"⚠️  Good-birth spawning failed: {e}", indent=1)
            return None

    def birth_bad_agent(self, source_agent: BaseAgent, env: BaseEnv) -> Optional[BaseAgent]:
        """Create a repaired child from a failed or bankrupt source agent."""
        try:
            task_description = env.get_task_description() if hasattr(env, "get_task_description") else ""
            correct_answer = env.get_correct_answer() if hasattr(env, "get_correct_answer") else ""
            new_agent = self._birth_bad_agent_factory(
                source_agent,
                task_description=task_description or getattr(source_agent, "recent_failure_task", ""),
                correct_answer=correct_answer or getattr(source_agent, "recent_failure_answer", ""),
                failure_trace=getattr(source_agent, "recent_failure_trace", "") or source_agent.get_trace_recorded_at_death(),
            )
            new_agent.root_ancestor_class = getattr(
                source_agent, "root_ancestor_class", source_agent.__class__.__name__
            )
            new_agent.father_agent_id = getattr(source_agent, "father_agent_id", source_agent.id)
            new_agent.father_agent_name = getattr(
                source_agent, "father_agent_name", source_agent.name
            )
            new_agent.parent_agent_id = source_agent.id
            new_agent.parent_agent_name = source_agent.name
            new_agent.spawn_method = "bad_birth"
            new_agent.tasks_lived = 0
            new_agent.bankruptcy_episode = None
            self.logger.log(
                f"💀 SPAWNING: {new_agent.name} {new_agent.id} via BAD BIRTH (source: {source_agent.name})",
                indent=1, must_print=True,
            )
            return new_agent
        except Exception as e:
            self.logger.log(f"⚠️  Bad-birth spawning failed: {e}", indent=1)
            return None

    def remove_bankrupt_agents(self, env: Optional[BaseEnv] = None) -> List[BaseAgent]:
        """Remove all bankrupt agents from the population.

        Args:
            env: If provided, records the episode trajectory on each
                bankrupt agent before removal (for death analysis).

        Returns:
            List of agents that were removed.
        """
        bankrupt_agents = [a for a in self.population.get_all() if a.check_bankruptcy()]
        if not bankrupt_agents:
            return []

        self.logger.log(f"💀 {len(bankrupt_agents)} agent(s) went bankrupt", indent=1)
        for bankrupt in bankrupt_agents:
            self.logger.log(
                f"   💀 REMOVING {bankrupt.name} (id={bankrupt.id}, wealth=${bankrupt.wealth:.2f})",
                indent=1,
            )
            bankrupt.bankruptcy_episode = self.episode_count  # for statistics
            if env is not None and env.action_history:
                bankrupt.record_death_trace(env.action_history)
            self.population.remove_agent(bankrupt)
            self.bankruptcy_count += 1  # for statistics
        return bankrupt_agents

    def _select_best_agent_for_good_birth(self) -> Optional[BaseAgent]:
        """Return the richest current agent for good-birth mutation."""
        return self.population.get_richest_agent()

    def _select_worst_agent_for_bad_birth(self) -> Optional[BaseAgent]:
        """Return the poorest current agent for failure-driven repair births."""
        return self.population.get_poorest_agent()

    def _validate_evolution_probabilities(self) -> None:
        """Validate configured birth probabilities."""
        p_a = self.config.evolution.p_a
        p_b = self.config.evolution.p_b
        periodical_good_p = self.config.evolution.periodical_good_p
        if not 0.0 <= p_a <= 1.0:
            raise ValueError(f"evolution.p_a must be within [0, 1], got {p_a}")
        if not 0.0 <= p_b <= 1.0:
            raise ValueError(f"evolution.p_b must be within [0, 1], got {p_b}")
        if p_a + p_b > 1.0:
            raise ValueError(
                f"evolution.p_a + evolution.p_b must be <= 1, got {p_a + p_b:.3f}"
            )
        if not 0.0 <= periodical_good_p <= 1.0:
            raise ValueError(
                f"evolution.periodical_good_p must be within [0, 1], got {periodical_good_p}"
            )

    def _can_add_agent(self) -> bool:
        """Return whether another agent can be added under the population cap."""
        return self.max_num_agents <= 0 or len(self.population) < self.max_num_agents

    def _add_born_agent(self, agent: Optional[BaseAgent]) -> bool:
        """Add a newborn agent when it exists and the population cap allows it."""
        if agent is None:
            return False
        if not self._can_add_agent():
            self.logger.log("⚠️  Population cap reached; skipping birth", indent=1)
            return False
        agent.initialize(initial_wealth=self.config.engine.initial_wealth)
        self.population.add_agent(agent)
        return True

    def _build_agent_failure_trace(self, env: BaseEnv, agent_name: str) -> str:
        """Build a highlighted trajectory for one agent from the current task history."""
        parts: List[str] = []
        for step, action_data in sorted(env.action_history.items()):
            author = action_data.get("author", "")
            text = (action_data.get("text") or "").strip()
            if not text:
                continue
            prefix = ">>>" if author == agent_name else "   "
            parts.append(f"{prefix} [Step {step}] [Message from agent: {author}] {text}")
        return "\n\n".join(parts)

    def _episode_failed(self, env: BaseEnv) -> bool:
        """Best-effort failure detection used for recording recent failure context."""
        if hasattr(env, "is_successful"):
            successful = env.is_successful()
            if successful is not None:
                return not successful
        score = env.get_terminal_score() if hasattr(env, "get_terminal_score") else None
        return score is None or score < 1.0

    def _record_recent_failures(self, env: BaseEnv) -> None:
        """Update each participating agent's latest failure context before bankruptcy checks."""
        if not self._episode_failed(env):
            return
        task_description = env.get_task_description() if hasattr(env, "get_task_description") else ""
        correct_answer = env.get_correct_answer() if hasattr(env, "get_correct_answer") else ""
        for agent in self.population.get_all():
            trace = self._build_agent_failure_trace(env, agent.name)
            if trace:
                agent.record_recent_failure(
                    trace=trace,
                    task_description=task_description,
                    correct_answer=correct_answer,
                )

    def _resolve_bankruptcy_births(
        self,
        all_bankrupts_in_episode: List[BaseAgent],
        env: BaseEnv,
    ) -> None:
        """Handle births caused by bankruptcies after all removals are complete."""
        if not all_bankrupts_in_episode:
            return

        self._validate_evolution_probabilities()
        p_a = self.config.evolution.p_a
        p_b = self.config.evolution.p_b
        for bankrupt in all_bankrupts_in_episode:
            draw = random.random()
            if draw < p_a:
                parent = self._select_best_agent_for_good_birth()
                if parent is not None:
                    self._add_born_agent(self.birth_good_agent(parent))
            elif draw < p_a + p_b:
                self._add_born_agent(self.birth_bad_agent(bankrupt, env))
            else:
                # No birth
                pass 

    def _ensure_necessary_roles_exist(
        self,
        bankrupt_agents: List[BaseAgent],
        env: Optional[BaseEnv] = None,
    ) -> None:
        """Guarantee every role from the initial agents has at least one living
        representative.  Compare current population roles (via ``agent.role``)
        against the original roles, find which are missing, and spawn
        replacements from the corresponding bankrupt agents."""
        if not self._initial_agents:
            return
        original_roles = {a.role for a in self._initial_agents}
        living_roles = {a.role for a in self.population.get_all()}
        missing_roles = original_roles - living_roles
        if not missing_roles:
            return
        bankrupt_by_role: Dict[str, List[BaseAgent]] = {}
        for agent in bankrupt_agents:
            bankrupt_by_role.setdefault(agent.role, []).append(agent)
        for role in missing_roles:
            candidates = bankrupt_by_role.get(role, [])
            assert candidates, (
                f"Role {role!r} is missing from the population but no "
                f"bankrupt agent of that role was found — this should never happen."
            )
            source = random.choice(candidates)
            self.logger.log(
                f"🛡️ Role {role!r} missing — force-spawning from {source.name}",
                indent=1,
            )
            spawned = self._add_born_agent(self.birth_bad_agent(source, env))
            if not spawned:
                raise RuntimeError(
                    f"Role recovery failed for {role!r}: force-spawn from {source.name!r} "
                    "did not produce a live replacement."
                )

    def _get_terminal_score(self, env: BaseEnv) -> float:
        """Absolute terminal score for the completed episode (used for stats/logs)."""
        score = env.get_terminal_score()
        if score is None:
            score = self.config.reward.missing_terminal_score
        return score

    def _get_path_reward_score(self, env: BaseEnv) -> float:
        """Per-episode contribution score used to size path_reward.

        Equals ``get_terminal_score()`` for one-shot domains where each
        episode is a fresh task, so the absolute score is the episode's
        contribution. Adapters with monotonic cross-episode terminal
        scores (cloudcast: workspace + best snapshot persist) override
        ``get_path_reward_score()`` to return the delta — otherwise every
        episode would re-pay path agents for prior episodes' progress.
        """
        score = env.get_path_reward_score()
        if score is None:
            score = self.config.reward.missing_terminal_score
        return score

    def _get_episode_path_agents(self, env: BaseEnv) -> List[BaseAgent]:
        """Return the full agent path for the accepted episode in chronological order.

        Each step produces one entry, so an agent acting multiple times
        appears multiple times in the returned list.
        """
        if not env.action_history:
            return []
        lookup = {agent.name: agent for agent in self.population.get_all()}
        path_agents: List[BaseAgent] = []
        for step, action_data in sorted(env.action_history.items()):
            author = action_data.get("author", "")
            if not author:
                continue
            agent = lookup.get(author)
            if agent is None:
                continue
            path_agents.append(agent)
        return path_agents

    def _apply_path_rewards(self, env: BaseEnv) -> Dict[str, Any]:
        """Apply terminal-derived path rewards to agents on the realized path only.

        Distribution is controlled by config.reward.path_reward_per_unique_author:
          - False (default): reward / N_actions, each occurrence rewarded.
            Suitable when action count IS effort (finance/math).
          - True: reward / N_unique, each agent rewarded once.
            Suitable when upstream roles wake fewer times but contribute
            equal-or-greater strategic value (e.g. Historian/Planner in the
            arch_dse_world H/P/E topology). The legacy `distribution_mode:
            "per_unique_agent"` config key is back-compat translated to this
            in HayekConfig._apply_reward_dict.
        """
        score = self._get_terminal_score(env)             # absolute, for logs/stats
        pay_score = self._get_path_reward_score(env)      # per-episode contribution
        reward_signal = self.config.reward.reward_signal(pay_score)
        path_agents = self._get_episode_path_agents(env)

        # Optionally dedup to unique authors so an agent that won 18 of 22
        # steps doesn't collect 18× share. Total path reward is preserved
        # (denominator = unique count), but each author gets paid once.
        if self.config.reward.path_reward_per_unique_author:
            seen_ids = set()
            unique_agents: List[BaseAgent] = []
            for a in path_agents:
                if a.id in seen_ids:
                    continue
                seen_ids.add(a.id)
                unique_agents.append(a)
            payout_targets = unique_agents
        else:
            payout_targets = path_agents

        per_agent_reward = self.config.reward.path_reward_per_agent(pay_score, len(payout_targets))

        for agent in payout_targets:
            agent.gain_money(per_agent_reward)
            agent.gain_capability(per_agent_reward)

        total_path_reward = per_agent_reward * len(payout_targets)
        self.total_rewards += total_path_reward

        if payout_targets:
            shaping = "centered" if self.config.reward.center_env_reward else "raw"
            pay_note = "" if pay_score == score else f" (delta={pay_score:+.2f})"
            mode_label = "per_unique_agent" if self.config.reward.path_reward_per_unique_author else "per_act"
            self.logger.log(
                f"🛤️  Path reward [{mode_label}]: score={score:.2f}{pay_note}, "
                f"{shaping}_signal={reward_signal:+.2f} "
                f"→ {per_agent_reward:+.2f} × {len(payout_targets)} agents "
                f"(of {len(path_agents)} actions)",
                indent=1,
            )
        else:
            self.logger.log(
                f"🛤️  Path reward skipped: no realized path agents (score={score:.2f})",
                indent=1,
            )

        return {
            "terminal_score": score,
            "center_env_reward": self.config.reward.center_env_reward,
            "reward_signal": reward_signal,
            "centered_score": reward_signal,
            "path_length": len(path_agents),
            "path_reward_per_agent": per_agent_reward,
            "path_reward_total": total_path_reward,
            "path_agent_names": [agent.name for agent in path_agents],
            "terminal_output_reward": (
                env.get_last_final_reward()
                if hasattr(env, "get_last_final_reward")
                else 0.0
            ),
            "terminal_output_author": (
                env.get_final_answer_author()
                if hasattr(env, "get_final_answer_author")
                else None
            ),
        }

    def _is_terminal_mode_step(self, step: int) -> bool:
        """Return True when the current step should enter terminal wrap-up mode."""
        if not self.config.terminal.enabled:
            return False
        start_on_step_from_end = max(0, self.config.terminal.start_on_step_from_end)
        if start_on_step_from_end <= 0:
            return False
        steps_remaining_including_current = self.max_steps_per_episode - step
        return steps_remaining_including_current <= start_on_step_from_end

    def _restrict_agents_for_terminal_mode(
        self,
        candidate_agents: List[BaseAgent],
    ) -> List[BaseAgent]:
        """Restrict the final-step auction to configured wrap-up agent tags when possible."""
        terminal_config = getattr(self.config, "terminal", None)
        if not terminal_config or not terminal_config.enabled:
            return candidate_agents

        allowed_tags = set(getattr(terminal_config, "candidate_agent_tags", ()))
        if not allowed_tags:
            return candidate_agents

        filtered_agents = [
            agent
            for agent in candidate_agents
            if hasattr(agent, "has_any_tag") and agent.has_any_tag(allowed_tags)
        ]
        return filtered_agents

    # ═══════════════════════════════════════════════════════════════════════
    # CORE STEP LOOP (shared by both modes)
    # ═══════════════════════════════════════════════════════════════════════

    def _run_auction_action_loop(self, env: BaseEnv, step_ckpt_save_path: Optional[str] = None):
        """
        Run the auction-action loop. Uses self._training to decide behaviour.

        Training: bids initialized, payments made, rewards applied, novice→veteran.
        Eval: agents act on existing bids, no wealth changes.
        If step_ckpt_save_path is set, writes step_N.json after each step.
        """
        prev_winner: Optional[BaseAgent] = None
        # Rolling window of recent winners for step-reward chain splitting.
        # When enabled (config.reward.step_reward_split_chain), an action
        # that returns positive env reward distributes the reward across the
        # unique members of the last N winners — giving H and P credit for
        # the advice/direction that led to E's successful submit.
        chain_window: List[BaseAgent] = []
        chain_window_size = max(1, int(getattr(self.config.reward, "step_reward_chain_window", 3)))
        split_chain = bool(getattr(self.config.reward, "step_reward_split_chain", False))
        training = self._training

        if step_ckpt_save_path:
            ckpt_path = Path(step_ckpt_save_path)
            ckpt_path.mkdir(parents=True, exist_ok=True)
        else:
            ckpt_path = None

        self.last_termination_reason = TerminationReason.MAX_STEPS
        for step in tqdm(range(self.max_steps_per_episode), disable=not self.logger.should_show_progress_bar()):
            self.logger.log(f"\n┌─ Step {step + 1}/{self.max_steps_per_episode} ─┐", indent=1)

            terminal_mode = self._is_terminal_mode_step(step)
            env.terminal_mode = terminal_mode
            env.allow_abstain_terminal_mode = getattr(self.config.terminal, "allow_abstain_terminal_mode", True)

            # Determine the active agents for the auction
            if terminal_mode:
                all_agents = self.population.get_all()
                active_agents = self._restrict_agents_for_terminal_mode(all_agents)
                if active_agents and len(active_agents) != len(all_agents):
                    allowed_tags = ", ".join(
                        sorted(set(getattr(self.config.terminal, "candidate_agent_tags", ())))
                    )
                    self.logger.log(
                        f"🏁 Terminal mode: bypassing wakeup and restricting final-step auction to tags [{allowed_tags}]",
                        indent=2,
                    )
                else:
                    self.logger.log(
                        "🏁 Terminal mode: bypassing wakeup and using the full population",
                        indent=2,
                    )
            else:
                # ─── ACTIVATION ───
                log_wakeup = getattr(self.config.concurrency, "log_wakeup", True)
                setattr(env, "log_wakeup", log_wakeup)
                active_agents = self.population.get_active(
                    env,
                    parallel=(
                        getattr(self.config.concurrency, "wakeup_parallel_enabled", False)
                        and getattr(self.config.concurrency, "wakeup_max_workers", 1) > 1
                    ),
                    max_workers=max(1, getattr(self.config.concurrency, "wakeup_max_workers", 1)),
                    retry_attempts=max(
                        0,
                        getattr(self.config.concurrency, "wakeup_retry_attempts", 0),
                    ),
                    retry_backoff_seconds=max(
                        0.0,
                        getattr(self.config.concurrency, "wakeup_retry_backoff_seconds", 0.0),
                    ),
                    fail_open=getattr(self.config.concurrency, "wakeup_fail_open", False),
                    log_wakeup=log_wakeup,
                )

            if not active_agents:
                self.logger.log(f"⚠️  No active agents! Ending episode.", indent=2)
                self.last_termination_reason = TerminationReason.NO_ACTIVE_AGENTS
                break

            self.logger.log(f"👥 Active agents: {len(active_agents)}", indent=2)

            # ─── BID INITIALIZATION ───
            if training:
                if self.bid_scheme == "fixed":
                    # Novice enters at a premium (base_bid + ε) to guarantee
                    # first auction win, then locks to base_bid.
                    for agent in active_agents:
                        if agent.get_status() == AgentStatus.NOVICE:
                            agent.set_bid(self.base_bid + self.novice_bid_epsilon)
                            agent.set_status(AgentStatus.VETERAN)
                            self.logger.log(
                                f"🆕 {agent.name} {agent.id} → VETERAN "
                                f"novice_bid={agent.get_bid():.2f}, "
                                f"will settle to base_bid={self.base_bid:.2f}",
                                indent=2,
                            )
                        elif agent.get_status() == AgentStatus.VETERAN:
                            agent.set_bid(self.base_bid)
                elif self.bid_scheme == "fixed_with_eps":
                    # Novice bid = max(veteran bids) + ε to guarantee first
                    # auction win. Once set, bid never changes (accumulated).
                    veteran_bids = [
                        a.get_bid() for a in active_agents
                        if a.get_status() == AgentStatus.VETERAN and a.get_bid() is not None
                    ]
                    high_bid = max(veteran_bids) if veteran_bids else self.base_bid
                    for agent in active_agents:
                        if agent.get_status() == AgentStatus.NOVICE:
                            agent.set_bid(high_bid + self.novice_bid_epsilon)
                            agent.set_status(AgentStatus.VETERAN)
                            self.logger.log(
                                f"🆕 {agent.name} {agent.id} → VETERAN "
                                f"bid={agent.get_bid():.2f} "
                                f"(high_bid={high_bid:.2f} + ε={self.novice_bid_epsilon:.2f})",
                                indent=2,
                            )
                elif self.bid_scheme == "holland":
                    # Holland's wealth-proportional rule: TYCOONs bid b = α × W.
                    # NOVICE entry premium = max(TYCOON bids) + ε so a fresh
                    # agent outbids the richest tycoon on its first auction.
                    # Falls back to base_bid when no tycoons exist yet.
                    # VETERAN → TYCOON is a one-way promotion once wealth
                    # crosses tycoon_wealth_threshold.
                    tycoon_bids = [
                        a.get_bid() for a in active_agents
                        if a.get_status() == AgentStatus.TYCOON and a.get_bid() is not None
                    ]
                    high_bid = max(tycoon_bids) if tycoon_bids else self.base_bid
                    for agent in active_agents:
                        if agent.get_status() == AgentStatus.NOVICE:
                            agent.set_bid(high_bid + self.novice_bid_epsilon)
                            agent.set_status(AgentStatus.VETERAN)
                            self.logger.log(
                                f"🆕 {agent.name} {agent.id} → VETERAN "
                                f"bid={agent.get_bid():.2f} "
                                f"(tycoon_high_bid={high_bid:.2f} + ε={self.novice_bid_epsilon:.2f})",
                                indent=2,
                            )
                        elif agent.get_status() == AgentStatus.VETERAN:
                            if agent.wealth >= self.tycoon_wealth_threshold:
                                agent.set_status(AgentStatus.TYCOON)
                                agent.set_bid(self.holland_alpha * agent.wealth)
                                self.logger.log(
                                    f"💎 {agent.name} {agent.id} VETERAN → TYCOON "
                                    f"wealth=${agent.wealth:.2f} ≥ "
                                    f"threshold=${self.tycoon_wealth_threshold:.2f}, "
                                    f"holland_bid=α×W={agent.get_bid():.2f}",
                                    indent=2,
                                )
                            else:
                                agent.set_bid(self.base_bid)
                        elif agent.get_status() == AgentStatus.TYCOON:
                            agent.set_bid(self.holland_alpha * agent.wealth)
                else:
                    raise ValueError(f"Unknown bid_scheme: {self.bid_scheme}")

            # ─── AUCTION ───
            self.logger.log(f"💵 Agents' bids: {' | '.join([f'{a.name} {a.id}: ${a.get_bid():.2f}' for a in active_agents])}", indent=2)
            assert all(a.get_bid() is not None for a in active_agents), "All agents must have a bid"
            max_bid = max(a.get_bid() for a in active_agents)
            top_bidders = [a for a in active_agents if a.get_bid() == max_bid]
            winner = random.choice(top_bidders)
            self.logger.log(f"🏆 WINNER: {winner.name} (id={winner.id}, bid={winner.get_bid():.2f})", indent=2)

            # ─── PAYMENT (TRAINING ONLY) ───
            if training:
                payment = winner.get_bid()
                winner.lose_money(payment)

                if prev_winner is not None:
                    prev_winner.gain_money(payment)

                    if prev_winner.id == winner.id:
                        self.logger.log(f"💰 Payment: {winner.name} acts again (no payment - same agent)", indent=2)
                        self.logger.log(f"   Balance: {winner.name}=${winner.wealth:.2f}", indent=2)
                    else:
                        self.logger.log(f"💰 Payment: {winner.name} → {prev_winner.name} (${payment:.2f})", indent=2)
                        self.logger.log(f"   Balances: {winner.name}=${winner.wealth:.2f}, {prev_winner.name}=${prev_winner.wealth:.2f}", indent=2)
                else:
                    self.logger.log(f"💰 Payment: {winner.name} → [void] (${payment:.2f}) — first action, no recipient", indent=2)
                    self.logger.log(f"   Balance: {winner.name}=${winner.wealth:.2f}", indent=2)

            # ─── ACTION ───
            self.logger.log(f"🎯 {winner.name} is acting...", indent=2)
            # Track the chain of recent winners for step-reward splitting.
            chain_window.append(winner)
            if len(chain_window) > chain_window_size:
                chain_window = chain_window[-chain_window_size:]
            action = winner.act(env)
            self.logger.log(f"   Action: {action}", indent=2)

            # ─── REWARD ───
            reward = env.apply(action)

            if training:
                if self.reward_scheme == "path_reward_and_stepwise_reward":
                    if split_chain and reward != 0:
                        # Split step reward among unique members of the
                        # recent-winners window (acts as a "credit chain"
                        # over the H→P→E cycle that produced this action).
                        seen_ids = set()
                        chain_uniq: List[BaseAgent] = []
                        for ag in chain_window:
                            if ag.id in seen_ids:
                                continue
                            seen_ids.add(ag.id)
                            chain_uniq.append(ag)
                        share = reward / len(chain_uniq)
                        for ag in chain_uniq:
                            ag.gain_money(share)
                            ag.gain_capability(share)
                        self.total_rewards += reward
                        if reward != 0:
                            emoji = "💵" if reward > 0 else "💸"
                            names = ", ".join(a.name for a in chain_uniq)
                            self.logger.log(
                                f"{emoji} Reward [chain-split]: {reward:+.2f} → {share:+.2f} × {len(chain_uniq)} ({names})",
                                indent=2,
                            )
                    else:
                        winner.gain_money(reward)
                        winner.gain_capability(reward)
                        self.total_rewards += reward
                        if reward != 0:
                            emoji = "💵" if reward > 0 else "💸"
                            self.logger.log(f"{emoji} Reward: {reward:+.2f} → {winner.name} (wealth=${winner.wealth:.2f})", indent=2)

            # ─── CHECK TERMINATION ───
            if env.reach_termination():
                self.logger.log(f"🏁 Episode TERMINATED after {step + 1} steps", indent=2)
                self.last_termination_reason = TerminationReason.GOAL_REACHED
                if ckpt_path:
                    self._write_step_checkpoint(step_ckpt_save_path, step + 1, env)
                break

            # ─── STEP CHECKPOINT ───
            if ckpt_path:
                self._write_step_checkpoint(step_ckpt_save_path, step + 1, env)

            if training:
                prev_winner = winner

    def _write_step_checkpoint(self, step_ckpt_save_path: str, step: int, env: BaseEnv):
        """Write a single step checkpoint JSON for debugging/inspection."""
        try:
            env_summary: Dict[str, Any] = {
                "step_count": env.step_count,
                "terminated": getattr(env, "terminated", None),
            }
            state = getattr(env, "state", "")
            env_summary["state_preview"] = state[-5000:] if len(state) > 5000 else state
            env_summary["action_history"] = env.action_history
            payload = {
                "step": step,
                "env": env_summary,
            }
            out_file = Path(step_ckpt_save_path) / f"step_{step}.json"
            with open(out_file, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            self.logger.log(f"⚠️  Checkpoint write failed: {e}", indent=2)

    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING FORWARD
    # ═══════════════════════════════════════════════════════════════════════

    def _train_forward(self, env: BaseEnv, step_ckpt_save_path: Optional[str] = None):
        """
        Training episode: wealth updates, credit checks, bankruptcy, spawning.
        If step_ckpt_save_path is set, step checkpoints are written there.
        """
        if len(self.population) == 0:
            raise RuntimeError("No agents in population!")

        self.episode_count += 1
        episode_rewards_before = self.total_rewards  # for statistics

        # ─── EPISODE HEADER ───
        self.logger.print_episode_header(
            episode_num=self.episode_count,
            mode="TRAIN",
            problem_desc=env.get_state_description(),
            population_size=len(self.population),
        )

        all_bankrupts_in_episode: List[BaseAgent] = []
        removed_agents_this_episode: List[BaseAgent] = []  # for statistics
        task_start_agent_ids = {agent.id for agent in self.population.get_all()}  # for statistics
        episode_action_history: Dict[str, List[Dict[str, Any]]] = {}  # for statistics
        reward_details: Dict[str, Any] = {}  # for statistics
        for agent in self.population.get_all():
            agent.tasks_lived += 1  # for statistics

        # ─── SNAPSHOT (wealth, bid, status) ───
        pre_episode_snapshot = {
            agent.id: agent.snapshot() for agent in self.population.get_all()
        }
        pre_episode_total_rewards = self.total_rewards
        self.logger.log(f"💾 Snapshot saved (wealth, capability, bid, status)", indent=1)

        # ─── REPLAY LOOP (up to max_trials_per_episode attempts) ───
        replay_count = 0  # for statistics
        for trial in range(self.max_trials_per_episode):
            if trial > 0:
                self.logger.log(f"\n🔁 REPLAY trial {trial + 1}/{self.max_trials_per_episode}", indent=1, must_print=True)
                # Update snapshot to reflect agents removed in prior trials
                pre_episode_snapshot = {
                    agent.id: agent.snapshot() for agent in self.population.get_all()
                }
                pre_episode_total_rewards = self.total_rewards
                replay_count += 1

            # ─── RESET ENVIRONMENT ───
            env.initialize()

            # ─── STEP LOOP ───
            self._run_auction_action_loop(env, step_ckpt_save_path=step_ckpt_save_path)

            if self.reward_scheme in ("path_reward_only", "path_reward_and_stepwise_reward"):
                reward_details = self._apply_path_rewards(env)

            # for statistics: collect per-agent action history
            for step, action_data in sorted(env.action_history.items()):
                author = action_data.get("author", "")
                if not author:
                    continue
                episode_action_history.setdefault(author, []).append(
                    {
                        "step": step,
                        "text": action_data.get("text", ""),
                        "is_final": action_data.get("is_final", False),
                        "is_code_action": action_data.get("is_code_action", False),
                    }
                )

            self._record_recent_failures(env)

            # ─── CREDIT CHECK ───
            agents_status = [
                (agent.name, agent.id, agent.wealth, agent.check_bankruptcy())
                for agent in self.population.get_all()
            ]
            self.logger.print_credit_check(agents_status)

            # ─── BANKRUPTCY HANDLING ───
            bankrupt_agents = self.remove_bankrupt_agents(env)
            self._ensure_necessary_roles_exist(bankrupt_agents, env)

            if bankrupt_agents:
                all_bankrupts_in_episode.extend(bankrupt_agents)  # for statistics
                removed_agents_this_episode.extend(bankrupt_agents)  # for statistics

                assert len(self.population) > 0
                # ─── ROLLBACK: restore surviving agents' wealth/capability only ───
                # Intentionally preserve current bid/status across replay attempts.
                self.logger.log(
                    "🔄 Restoring surviving agents' wealth/capability to snapshot (preserving bid/status)",
                    indent=1,
                )
                for agent in self.population.get_all():
                    if agent.id in pre_episode_snapshot:
                        snapshot = pre_episode_snapshot[agent.id]
                        agent.wealth = snapshot["wealth"]
                        agent.capability_score = snapshot["capability_score"]
                self.total_rewards = pre_episode_total_rewards
                # Continue to next trial (replay the instance)
                continue
            else:
                self.logger.log("✅ No bankruptcies - episode ACCEPTED", indent=1, must_print=True)
                break  # success, no need to replay

        # ─── RENT (before spawning, per Baum 1999) ───
        # v10: skip rent during preheat sessions (env.in_preheat=True). In
        # preheat the non-Executor roles structurally cannot win auctions
        # (their wakeup_condition gates them off), so charging them rent
        # would force bankruptcy before they ever get to participate.
        # Rent resumes once Hayek H/P/E activates at session_idx >=
        # preheat_until_session.
        rent = self.config.engine.rent
        rent_interval = self.config.engine.rent_interval
        in_preheat = getattr(env, "in_preheat", False)
        rent_due = rent > 0 and rent_interval > 0 and self.episode_count % rent_interval == 0
        if rent_due and not in_preheat:
            self.logger.log(f"\n🏠 Charging rent ${rent:.4f} to all {len(self.population)} agents", indent=1)
            for agent in self.population.get_all():
                agent.lose_money(rent)
            rent_bankrupt_agents = self.remove_bankrupt_agents()
            self._ensure_necessary_roles_exist(rent_bankrupt_agents, env)
            all_bankrupts_in_episode.extend(rent_bankrupt_agents)
            removed_agents_this_episode.extend(rent_bankrupt_agents)
        elif rent_due and in_preheat:
            self.logger.log(
                f"\n🏠 Rent skipped (preheat session — H/P cannot earn yet)",
                indent=1,
            )

        self._resolve_bankruptcy_births(all_bankrupts_in_episode, env)

        # ─── BIRTHS (periodical births) ───
        if self.birth_interval > 0 and self.episode_count % self.birth_interval == 0 and self._can_add_agent():
            self.logger.log(f"\n🐣 Periodic birth: spawning {self.num_births_per_interval} new agent(s)", indent=1)
            self._periodic_births(env)

        # Replenish if population dropped below minimum by mutating initial agents
        if self.min_num_agents > 0 and len(self.population) < self.min_num_agents:
            self.logger.log(
                f"\n🐣 Population below minimum ({len(self.population)}/{self.min_num_agents}), "
                f"adding exploratory good-birth agent(s) to replenish",
                indent=1,
            )
            for parent in self._initial_agents:
                if len(self.population) >= self.min_num_agents:
                    break
                source_parent = self._select_best_agent_for_good_birth() or parent
                new_agent = self.birth_good_agent(source_parent)
                if new_agent:
                    self._add_born_agent(new_agent)

        # ─── EPISODE SUMMARY ───
        self.logger.log(f"\n{'─'*70}")
        self.logger.log(f"📊 Episode {self.episode_count} [TRAIN] COMPLETE")
        self.logger.log(f"   Population: {len(self.population)} agents")
        self.logger.log(f"   Bankruptcies: {len(all_bankrupts_in_episode)}")
        self.logger.log(f"   Total rewards this episode: {self.total_rewards - episode_rewards_before:.2f}")

        # for statistics
        self.last_episode_metrics = self._build_episode_metrics(
            env=env,
            replay_count=replay_count,
            task_start_agent_ids=task_start_agent_ids,
            removed_agents=removed_agents_this_episode,
            aggregated_agent_actions=episode_action_history,
            reward_details=reward_details,
        )

    def _periodic_births(self, env: BaseEnv):
        self._validate_evolution_probabilities()
        for _ in range(self.num_births_per_interval):
            if not self._can_add_agent():
                break
            
            if random.random() < self.config.evolution.periodical_good_p:
                parent = self._select_best_agent_for_good_birth()
                if parent:
                    self._add_born_agent(self.birth_good_agent(parent))
            else:
                source_agent = self._select_worst_agent_for_bad_birth()
                if source_agent:
                    self._add_born_agent(self.birth_bad_agent(source_agent, env))

    # ═══════════════════════════════════════════════════════════════════════
    # EVALUATION FORWARD
    # ═══════════════════════════════════════════════════════════════════════

    def _eval_forward(self, env: BaseEnv, step_ckpt_save_path: Optional[str] = None):
        """
        Evaluation episode: frozen population, no wealth changes.
        If step_ckpt_save_path is set, step checkpoints are written there.
        """
        self.episode_count += 1

        if len(self.population) == 0:
            raise RuntimeError("No agents in population! Call add_agent() first.")

        # ─── EPISODE HEADER ───
        self.logger.print_episode_header(
            episode_num=self.episode_count,
            mode="EVAL",
            problem_desc=env.get_state_description(),
            population_size=len(self.population),
        )

        # ─── RESET + RUN (single pass, no credit check) ───
        env.initialize()
        self._run_auction_action_loop(env, step_ckpt_save_path=step_ckpt_save_path)

        self.logger.log(f"\n🔒 EVAL MODE: Population unchanged (frozen)", indent=1, must_print=True)

        # ─── EPISODE SUMMARY ───
        self.logger.log(f"\n{'─'*70}")
        self.logger.log(f"📊 Episode {self.episode_count} [EVAL] COMPLETE")
        self.logger.log(f"   Population: {len(self.population)} agents")

    def _build_episode_metrics(  # for statistics
        self,
        env: BaseEnv,
        replay_count: int,
        task_start_agent_ids: set[int],
        removed_agents: List[BaseAgent],
        aggregated_agent_actions: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        reward_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build episode-level metrics and per-agent task records for analysis."""
        living_agents = self.population.get_all()
        wealths = [agent.wealth for agent in living_agents]
        wealth_highest = max(wealths) if wealths else None
        wealth_average = sum(wealths) / len(wealths) if wealths else None
        wealth_lowest = min(wealths) if wealths else None
        if wealths:
            mean = wealth_average if wealth_average is not None else 0.0
            wealth_variance = sum((value - mean) ** 2 for value in wealths) / len(wealths)
        else:
            wealth_variance = None

        env_metrics = env.build_episode_metrics() if hasattr(env, "build_episode_metrics") else {}

        agent_lookup: Dict[str, BaseAgent] = {agent.name: agent for agent in living_agents}
        for agent in removed_agents:
            agent_lookup[agent.name] = agent

        agent_actions = aggregated_agent_actions or {}

        removed_names = {agent.name for agent in removed_agents}
        agent_records: List[Dict[str, Any]] = []
        for author, actions in agent_actions.items():
            agent = agent_lookup.get(author)
            behavior_trace = ""
            if actions:
                behavior_trace = "\n\n".join(
                    f"[Step {action['step']}] {action['text']}".strip()
                    for action in actions
                    if action.get("text")
                )

            survival_count = -1
            survived_task = author not in removed_names
            if agent is not None and not survived_task:
                survival_count = agent.tasks_lived

            agent_records.append(
                {
                    "agent_id": agent.id if agent is not None else None,
                    "agent_name": author,
                    "agent_class": agent.__class__.__name__ if agent is not None else None,
                    "root_ancestor_class": (
                        getattr(agent, "root_ancestor_class", None) if agent is not None else None
                    ),
                    "father_agent_id": getattr(agent, "father_agent_id", None) if agent is not None else None,
                    "father_agent_name": (
                        getattr(agent, "father_agent_name", None) if agent is not None else None
                    ),
                    "parent_agent_id": getattr(agent, "parent_agent_id", None) if agent is not None else None,
                    "parent_agent_name": getattr(agent, "parent_agent_name", None) if agent is not None else None,
                    "spawn_method": getattr(agent, "spawn_method", None) if agent is not None else None,
                    "wealth_after_task": getattr(agent, "wealth", None) if agent is not None else None,
                    "capability_score_after_task": (
                        getattr(agent, "capability_score", None) if agent is not None else None
                    ),
                    "frozen_system_prompt": (
                        getattr(agent, "frozen_system_prompt", None) if agent is not None else None
                    ),
                    "trainable_system_prompt": (
                        getattr(agent, "trainable_system_prompt", None) if agent is not None else None
                    ),
                    "survived_task": survived_task,
                    "survival_count": survival_count,
                    "tasks_lived_so_far": agent.tasks_lived if agent is not None else None,
                    "behavior_trace": behavior_trace,
                    "actions": actions,
                }
            )

        bankruptcies = len(removed_agents)
        surviving_from_start = sum(1 for agent in living_agents if agent.id in task_start_agent_ids)

        return {
            "episode": self.episode_count,
            "termination_reason": self.last_termination_reason.value,
            "replays": replay_count,
            "bankruptcies": bankruptcies,
            "population_size": len(living_agents),
            "surviving_from_task_start": surviving_from_start,
            "wealth_highest": wealth_highest,
            "wealth_average": wealth_average,
            "wealth_lowest": wealth_lowest,
            "wealth_variance": wealth_variance,
            "step_count": getattr(env, "step_count", None),
            **(reward_details or {}),
            **env_metrics,
            "agent_records": agent_records,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # STATISTICS & DISPLAY
    # ═══════════════════════════════════════════════════════════════════════

    def print_population(self):
        """Print the current population status using rich table."""
        agents = list(self.population.get_best_agents(key=lambda a: a.wealth))
        self.logger.print_population_table(agents)
        self.logger.log(f"Total agents: {len(self.population)} | Total bankruptcies: {self.bankruptcy_count}")

    # ═══════════════════════════════════════════════════════════════════════
    # SAVE / LOAD
    # ═══════════════════════════════════════════════════════════════════════

    def serialize_settings(self) -> Dict[str, Any]:
        """Serialize MAS settings (config and stats) to dictionary."""
        max_id = max(self.population.get_agent_ids(), default=0)
        return {
            "version": "2.2",
            "config": {
                "engine": {
                    "max_steps_per_episode": self.max_steps_per_episode,
                    "max_trials_per_episode": self.max_trials_per_episode,
                    "birth_interval": self.birth_interval,
                    "num_births_per_interval": self.num_births_per_interval,
                    "min_num_agents": self.min_num_agents,
                    "max_num_agents": self.max_num_agents,
                    "bid_scheme": self.bid_scheme,
                    "novice_bid_epsilon": self.novice_bid_epsilon,
                    "holland_alpha": self.holland_alpha,
                    "tycoon_wealth_threshold": self.tycoon_wealth_threshold,
                    "rent": self.config.engine.rent,
                    "rent_interval": self.config.engine.rent_interval,
                },
                "reward": {
                    "reward_scheme": self.reward_scheme,
                    "initial_wealth": self.config.engine.initial_wealth,
                    "path_reward_scale": self.config.reward.path_reward_scale,
                    "terminal_output_bonus_scale": self.config.reward.terminal_output_bonus_scale,
                    "env_reward_scale": self.config.reward.env_reward_scale,
                    "missing_terminal_score": self.config.reward.missing_terminal_score,
                    "center_env_reward": self.config.reward.center_env_reward,
                    "base_bid": self.config.engine.base_bid,
                },
                "evolution": {
                    "p_a": self.config.evolution.p_a,
                    "p_b": self.config.evolution.p_b,
                    "periodical_good_p": self.config.evolution.periodical_good_p,
                },
                "evaluation": {
                    "periodic_test_enabled": self.config.evaluation.periodic_test_enabled,
                    "periodic_test_before_training": (
                        self.config.evaluation.periodic_test_before_training
                    ),
                    "periodic_test_every_n_tasks": self.config.evaluation.periodic_test_every_n_tasks,
                    "periodic_test_parallel_enabled": (
                        self.config.evaluation.periodic_test_parallel_enabled
                    ),
                    "periodic_test_max_workers": self.config.evaluation.periodic_test_max_workers,
                },
                "wakeup": {
                    "wakeup_parallel_enabled": self.config.concurrency.wakeup_parallel_enabled,
                    "wakeup_max_workers": self.config.concurrency.wakeup_max_workers,
                    "wakeup_retry_attempts": self.config.concurrency.wakeup_retry_attempts,
                    "wakeup_retry_backoff_seconds": (
                        self.config.concurrency.wakeup_retry_backoff_seconds
                    ),
                    "wakeup_fail_open": self.config.concurrency.wakeup_fail_open,
                    "log_wakeup": self.config.concurrency.log_wakeup,
                },
                "terminal": {
                    "enabled": self.config.terminal.enabled,
                    "start_on_step_from_end": self.config.terminal.start_on_step_from_end,
                    "candidate_agent_tags": list(self.config.terminal.candidate_agent_tags),
                    "allow_abstain_terminal_mode": self.config.terminal.allow_abstain_terminal_mode,
                },
            },
            "stats": {
                "episodes": self.episode_count,
                "total_rewards": self.total_rewards,
                "bankruptcies": self.bankruptcy_count,
            },
            "max_agent_id": max_id,
        }

    @classmethod
    def deserialize_settings(cls, data: Dict[str, Any]) -> "HayekMAS":
        """Deserialize MAS settings from dictionary."""
        config_raw = data.get("config", {})
        mas_config = HayekConfig.from_dict(config_raw)
        mas = cls(config=mas_config)

        stats = data.get("stats", {})
        mas.episode_count = stats.get("episodes", 0)
        mas.total_rewards = stats.get("total_rewards", 0.0)
        mas.bankruptcy_count = stats.get("bankruptcies", 0)

        max_id = data.get("max_agent_id", 0)
        set_agent_id_counter(max_id + 1)

        return mas

    def save(self, filepath: str, agent_serializer: Optional[Callable[[BaseAgent], Dict]] = None):
        """Save the Hayek Machine state to a JSON file."""
        self.logger.log(f"\n💾 Saving Hayek Machine...")

        data = self.serialize_settings()

        if agent_serializer is None:
            def default_serializer(agent: BaseAgent) -> Dict:
                if hasattr(agent, 'serialize') and callable(agent.serialize):
                    return agent.serialize()
                return {
                    "id": agent.id,
                    "name": agent.name,
                    "wealth": agent.wealth,
                    "capability_score": agent.capability_score,
                    "bid": agent.get_bid(),
                    "status": agent.get_status().value,
                    "type": agent.__class__.__name__,
                }
            agent_serializer = default_serializer

        data["population"] = [agent_serializer(a) for a in self.population.get_all()]
        # Persist initial templates so resumed runs can keep role guarantees.
        data["initial_population"] = [agent_serializer(a) for a in self._initial_agents]

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.log(f"✅ Saved to: {filepath}")
        self.logger.log(f"   📦 {len(self.population)} agents")
        self.logger.log(f"   📚 {self.episode_count} episodes trained")
        self.logger.log(f"   💰 ${self.total_rewards:.2f} total rewards")

    @classmethod
    def load(
        cls,
        filepath: str,
        agent_deserializer: Callable[[Dict], BaseAgent],
    ) -> "HayekMAS":
        """Load a Hayek Machine from a JSON file."""
        logger.log(f"\n📂 Loading Hayek Machine from: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls.load_from_data(data, agent_deserializer=agent_deserializer)

    @classmethod
    def load_from_data(
        cls,
        data: Dict[str, Any],
        agent_deserializer: Callable[[Dict], BaseAgent],
    ) -> "HayekMAS":
        """Load a Hayek Machine from an in-memory serialized dictionary."""
        mas = cls.deserialize_settings(data)

        loaded_population: List[BaseAgent] = []
        for agent_data in data.get("population", []):
            agent = agent_deserializer(agent_data)
            mas.population.add_agent(agent)
            loaded_population.append(agent)

        # Restore initial templates for role-preservation logic.
        # Backward compatibility: if absent, derive one template per role from population.
        initial_population_data = data.get("initial_population", [])
        if initial_population_data:
            by_id: Dict[int, BaseAgent] = {agent.id: agent for agent in loaded_population}
            restored_initial: List[BaseAgent] = []
            for initial_data in initial_population_data:
                template_id = initial_data.get("id")
                if template_id in by_id:
                    restored_initial.append(by_id[template_id])
                else:
                    restored_initial.append(agent_deserializer(initial_data))
            mas._initial_agents = restored_initial
        else:
            seen_roles = set()
            fallback_initial: List[BaseAgent] = []
            for agent in loaded_population:
                if agent.role in seen_roles:
                    continue
                seen_roles.add(agent.role)
                fallback_initial.append(agent)
            mas._initial_agents = fallback_initial

        logger.log(f"✅ Loaded successfully!")
        logger.log(f"   📦 {len(mas.population)} agents restored")
        logger.log(f"   🧩 {len(mas._initial_agents)} initial templates restored")
        logger.log(f"   📚 {mas.episode_count} episodes previously trained")
        logger.log(f"   💰 ${mas.total_rewards:.2f} total rewards accumulated")

        return mas
