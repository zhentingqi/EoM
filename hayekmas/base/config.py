"""
Engine-level configuration objects consumed by `HayekMAS` and adapter runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    """Reward settings used by the engine.

    These values control path rewards and terminal-output scoring behavior.
    """
    reward_scheme: str = "path_reward_only"
    path_reward_scale: float = 1.0
    terminal_output_bonus_scale: float = 0.5
    env_reward_scale: float = 1.0
    missing_terminal_score: float = 0.0
    center_env_reward: bool = True
    # Asymmetric shaping for mid-episode checkpoint deltas. Defaults are
    # backward-compatible (no extra penalty). Cloudcast turns these on to
    # punish "agent ran eval on a regressed/broken program" harder than
    # an even score-axis would.
    regression_multiplier: float = 1.0    # multiplier applied to negative checkpoint deltas
    broken_program_penalty: float = 0.0   # extra flat penalty when verifier returns configs=0
    # When True, path-reward at episode end is split across UNIQUE authors
    # (each appears once), not across every action step. Without this, an
    # agent that wins 18 of 22 steps in an episode collects 18× the share
    # of any agent that won once, even though the work product was joint —
    # which leads to wealth runaway / monopoly. Default False preserves
    # legacy behavior for tasks where action-count IS effort (finance/math).
    # The arch_dse_world adapter sets this to True so the Historian and
    # Planner aren't starved by the Executor's higher action count.
    path_reward_per_unique_author: bool = False
    # Step reward distribution for `reward_scheme="path_reward_and_stepwise_reward"`.
    # By default a positive env reward goes ONLY to the agent that took the
    # action (the auction winner). In H/P/E topologies this concentrates
    # wealth at the Executor (only the Executor acts on the env). Setting
    # this to True splits the step reward among the last
    # `step_reward_chain_window` unique winners, giving the Historian and
    # Planner credit for the advice/direction that enabled the Executor's
    # successful submit.
    step_reward_split_chain: bool = False
    step_reward_chain_window: int = 3

    def centered_score(self, score: float) -> float:
        return 2.0 * score - 1.0

    def reward_signal(self, score: float) -> float:
        """Shaped terminal score for env rewards: optional centering, then ``env_reward_scale``."""
        base = self.centered_score(score) if self.center_env_reward else score
        return self.env_reward_scale * base

    def terminal_output_bonus(self, score: float) -> float:
        return self.terminal_output_bonus_scale * self.reward_signal(score)

    def path_reward_per_agent(self, score: float, path_length: int) -> float:
        if path_length <= 0:
            return 0.0
        return self.path_reward_scale * self.reward_signal(score) / path_length

    def shaped_checkpoint_delta(self, delta: float, was_broken: bool) -> float:
        """Apply asymmetric shaping to a mid-episode checkpoint delta.

        - Broken eval (configs=0) → SUBSTITUTE a flat ``-broken_program_penalty``
          regardless of the raw delta. (The raw delta is meaningless when
          the program crashed; we just want a fixed punishment.)
        - Negative delta (regression) → multiply by ``regression_multiplier``
          to make backsliding hurt more than progress pays.
        - Positive delta → passed through unchanged.

        The returned value is still in delta-space; ``env_reward_scale``
        is applied later by the caller (env.apply_action).
        """
        if was_broken:
            return -self.broken_program_penalty
        if delta < 0.0 and self.regression_multiplier != 1.0:
            return delta * self.regression_multiplier
        return delta


@dataclass
class EvolutionConfig:
    """Evolution settings for good/bad births."""
    p_a: float = 0.0
    p_b: float = 1.0
    periodical_good_p: float = 0.5


@dataclass
class EvaluationConfig:
    """Periodic evaluation settings stored with the engine config."""
    periodic_test_enabled: bool = True
    periodic_test_before_training: bool = False
    periodic_test_every_n_tasks: int = 5
    periodic_test_parallel_enabled: bool = True
    periodic_test_max_workers: int = 4


@dataclass
class ConcurrencyConfig:
    """Concurrency settings used by engine wakeup checks."""
    wakeup_parallel_enabled: bool = False
    wakeup_max_workers: int = 1
    wakeup_retry_attempts: int = 2
    wakeup_retry_backoff_seconds: float = 1.0
    wakeup_fail_open: bool = False
    log_wakeup: bool = True


@dataclass
class TerminalConfig:
    """Terminal-step settings for final wrap-up behavior in the engine."""
    enabled: bool = True
    start_on_step_from_end: int = 1
    candidate_agent_tags: tuple[str, ...] = ("terminal",)
    allow_abstain_terminal_mode: bool = True


@dataclass
class EngineConfig:
    """Core engine parameters for episode execution, population, and bidding."""
    max_steps_per_episode: int = 10
    max_trials_per_episode: int = 4
    birth_interval: int = 5
    num_births_per_interval: int = 2
    min_num_agents: int = 0
    max_num_agents: int = 0
    bid_scheme: str = "fixed"
    base_bid: float = 0.1
    novice_bid_epsilon: float = 0.01
    initial_wealth: float = 0.5
    rent: float = 0.0
    rent_interval: int = 5
    # Holland scheme: VETERAN → TYCOON when wealth ≥ tycoon_wealth_threshold;
    # TYCOON bids b = holland_alpha × wealth.
    holland_alpha: float = 0.1
    tycoon_wealth_threshold: float = 5.0


@dataclass
class HayekConfig:
    """Engine-level configuration container for `HayekMAS`."""
    # Sub-configs
    engine: EngineConfig = field(default_factory=EngineConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    terminal: TerminalConfig = field(default_factory=TerminalConfig)

    @classmethod
    def from_dict(cls, raw: dict) -> "HayekConfig":
        """Build an engine config from a plain dictionary.

        Args:
            raw: Nested config dictionary with optional reward, evolution,
                evaluation, concurrency, and terminal sections.

        Returns:
            A populated `HayekConfig` instance.
        """
        config = cls()
        # Engine fields: look inside an "engine" sub-dict first, fall back to top-level keys
        engine_raw = raw.get("engine", raw)
        config.engine.max_steps_per_episode = engine_raw.get("max_steps_per_episode", config.engine.max_steps_per_episode)
        config.engine.max_trials_per_episode = engine_raw.get("max_trials_per_episode", config.engine.max_trials_per_episode)
        config.engine.birth_interval = engine_raw.get("birth_interval", config.engine.birth_interval)
        config.engine.num_births_per_interval = engine_raw.get("num_births_per_interval", config.engine.num_births_per_interval)
        config.engine.min_num_agents = engine_raw.get("min_num_agents", config.engine.min_num_agents)
        config.engine.max_num_agents = engine_raw.get("max_num_agents", config.engine.max_num_agents)
        config.engine.bid_scheme = engine_raw.get("bid_scheme", config.engine.bid_scheme)
        config.engine.base_bid = engine_raw.get("base_bid", config.engine.base_bid)
        config.engine.novice_bid_epsilon = engine_raw.get("novice_bid_epsilon", config.engine.novice_bid_epsilon)
        config.engine.initial_wealth = engine_raw.get("initial_wealth", config.engine.initial_wealth)
        config.engine.rent = engine_raw.get("rent", config.engine.rent)
        config.engine.rent_interval = engine_raw.get("rent_interval", config.engine.rent_interval)
        config.engine.holland_alpha = engine_raw.get("holland_alpha", config.engine.holland_alpha)
        config.engine.tycoon_wealth_threshold = engine_raw.get(
            "tycoon_wealth_threshold", config.engine.tycoon_wealth_threshold
        )
        reward = raw.get("reward", {})
        config.reward.reward_scheme = reward.get("reward_scheme", config.reward.reward_scheme)
        config.reward.path_reward_scale = reward.get("path_reward_scale", config.reward.path_reward_scale)
        config.reward.terminal_output_bonus_scale = reward.get("terminal_output_bonus_scale", config.reward.terminal_output_bonus_scale)
        config.reward.env_reward_scale = reward.get("env_reward_scale", config.reward.env_reward_scale)
        config.reward.missing_terminal_score = reward.get("missing_terminal_score", config.reward.missing_terminal_score)
        config.reward.center_env_reward = reward.get(
            "center_env_reward", config.reward.center_env_reward
        )
        config.reward.regression_multiplier = reward.get(
            "regression_multiplier", config.reward.regression_multiplier
        )
        config.reward.broken_program_penalty = reward.get(
            "broken_program_penalty", config.reward.broken_program_penalty
        )
        config.reward.path_reward_per_unique_author = reward.get(
            "path_reward_per_unique_author", config.reward.path_reward_per_unique_author
        )
        # Back-compat: accept "distribution_mode": "per_unique_agent" as
        # an alias for path_reward_per_unique_author=True. This lets
        # configs from the original arch_dse_world release keep working.
        _dm = reward.get("distribution_mode")
        if _dm == "per_unique_agent":
            config.reward.path_reward_per_unique_author = True
        config.reward.step_reward_split_chain = reward.get(
            "step_reward_split_chain", config.reward.step_reward_split_chain
        )
        config.reward.step_reward_chain_window = int(reward.get(
            "step_reward_chain_window", config.reward.step_reward_chain_window
        ))
        evolution = raw.get("evolution", {})
        config.evolution.p_a = evolution.get("p_a", config.evolution.p_a)
        config.evolution.p_b = evolution.get("p_b", config.evolution.p_b)
        config.evolution.periodical_good_p = evolution.get(
            "periodical_good_p", config.evolution.periodical_good_p
        )
        evaluation = raw.get("evaluation", {})
        config.evaluation.periodic_test_enabled = evaluation.get("periodic_test_enabled", config.evaluation.periodic_test_enabled)
        config.evaluation.periodic_test_before_training = evaluation.get("periodic_test_before_training", config.evaluation.periodic_test_before_training)
        config.evaluation.periodic_test_every_n_tasks = evaluation.get("periodic_test_every_n_tasks", config.evaluation.periodic_test_every_n_tasks)
        config.evaluation.periodic_test_parallel_enabled = evaluation.get("periodic_test_parallel_enabled", config.evaluation.periodic_test_parallel_enabled)
        config.evaluation.periodic_test_max_workers = evaluation.get("periodic_test_max_workers", config.evaluation.periodic_test_max_workers)
        wakeup = raw.get("wakeup", {})
        config.concurrency.wakeup_parallel_enabled = wakeup.get("wakeup_parallel_enabled", config.concurrency.wakeup_parallel_enabled)
        config.concurrency.wakeup_max_workers = wakeup.get("wakeup_max_workers", config.concurrency.wakeup_max_workers)
        config.concurrency.wakeup_retry_attempts = wakeup.get("wakeup_retry_attempts", config.concurrency.wakeup_retry_attempts)
        config.concurrency.wakeup_retry_backoff_seconds = wakeup.get("wakeup_retry_backoff_seconds", config.concurrency.wakeup_retry_backoff_seconds)
        config.concurrency.wakeup_fail_open = wakeup.get("wakeup_fail_open", config.concurrency.wakeup_fail_open)
        config.concurrency.log_wakeup = wakeup.get("log_wakeup", config.concurrency.log_wakeup)
        terminal = raw.get("terminal", {})
        config.terminal.enabled = terminal.get("enabled", config.terminal.enabled)
        config.terminal.start_on_step_from_end = terminal.get("start_on_step_from_end", config.terminal.start_on_step_from_end)
        config.terminal.candidate_agent_tags = tuple(terminal.get("candidate_agent_tags", config.terminal.candidate_agent_tags))
        config.terminal.allow_abstain_terminal_mode = terminal.get("allow_abstain_terminal_mode", config.terminal.allow_abstain_terminal_mode)
        return config


DEFAULT_HAYEK_CONFIG = HayekConfig()
