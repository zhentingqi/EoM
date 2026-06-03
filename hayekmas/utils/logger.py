"""
Centralized logging utilities for HayekMAS.

This file defines the shared logger used across the engine, pipelines, and
adapters. It supports dual output: Rich console rendering when verbose and a
plain-text log file for every run.
"""

from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import threading
import os
from typing import Optional, TextIO, List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

# ═══════════════════════════════════════════════════════════════════════════
# Theme
# ═══════════════════════════════════════════════════════════════════════════

HAYEK_THEME = Theme({
    "train": "bold green",
    "eval": "bold cyan",
    "bankrupt": "bold red",
    "solvent": "green",
    "header": "bold white",
    "dim": "dim",
})


class HayekLogger:
    """Singleton logger shared by the whole codebase.

    The logger owns console formatting, file logging, per-thread silencing, and
    a few structured output helpers for mode banners, episode headers, and
    population tables.
    """

    _instance: Optional["HayekLogger"] = None

    def __new__(cls):
        """Return the singleton logger instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the singleton logger state once."""
        if self._initialized:
            return

        self.verbose = False
        self.log_file: Optional[TextIO] = None
        self.task_log_file: Optional[TextIO] = None
        self.console = Console(theme=HAYEK_THEME, highlight=False)
        self._lock = threading.RLock()
        self._thread_state = threading.local()
        self._initialized = True

    def configure(self, verbose: bool = False, log_dir: str = "logs", profile: str = ""):
        """
        Configure the logger.

        Args:
            verbose: Whether to print to console
            log_dir: Directory for log files
            profile: Experiment profile name, included in the log file name
        """
        self.verbose = verbose

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_{os.getpid()}"
        log_file_path = log_path / f"{profile}_{run_id}.log"

        if self.log_file:
            self.log_file.close()

        self.log_file = open(log_file_path, "w", encoding="utf-8")
        self.log_file.write(f"Hayek Machine Log - Started at {datetime.now().isoformat()}\n")
        self.log_file.write("=" * 70 + "\n\n")
        self.log_file.flush()

    def _write_plain_text(self, text: str) -> None:
        """Write plain-text log output to the run log and current task log."""
        if self.log_file:
            self.log_file.write(text)
            self.log_file.flush()
        if self.task_log_file:
            self.task_log_file.write(text)
            self.task_log_file.flush()

    @contextmanager
    def scoped_task_log(self, log_path: Optional[Path]):
        """Mirror all logger output into one task-specific log file."""
        if log_path is None:
            yield
            return

        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self.task_log_file:
                self.task_log_file.close()
            self.task_log_file = open(path, "w", encoding="utf-8")
        try:
            yield
        finally:
            with self._lock:
                if self.task_log_file:
                    self.task_log_file.close()
                    self.task_log_file = None

    def should_show_progress_bar(self) -> bool:
        """Whether a tqdm progress bar should be shown.

        Progress bars are shown only in non-verbose mode (they conflict
        with Rich console output).
        """
        return not self.verbose and not self._is_silenced()

    def _is_silenced(self) -> bool:
        """Return whether the current thread has logging temporarily muted."""
        return bool(getattr(self._thread_state, "silenced", 0))

    @contextmanager
    def scoped_silence(self):
        """
        Suppress logger output for the current thread only.

        This is useful for background worker threads that should not write into
        the main interactive training log.
        """
        current = getattr(self._thread_state, "silenced", 0)
        self._thread_state.silenced = current + 1
        try:
            yield
        finally:
            self._thread_state.silenced = current

    # ───────────────────────────────────────────────────────────────────
    # Core log method
    # ───────────────────────────────────────────────────────────────────

    def log(self, msg: str, indent: int = 0, must_print: bool = False):
        """
        Log a message to console (if verbose) and log file (always).

        Args:
            msg: The message to log
            indent: Indentation level (each level = 3 spaces)
            must_print: Force console output even if not verbose
        """
        if self._is_silenced():
            return

        indent_str = "   " * indent
        formatted_msg = f"{indent_str}{msg}"

        with self._lock:
            if self.verbose or must_print:
                self.console.print(formatted_msg)
            self._write_plain_text(formatted_msg + "\n")

    # ───────────────────────────────────────────────────────────────────
    # Structured Rich output methods
    # ───────────────────────────────────────────────────────────────────

    def print_mode_banner(self, mode: str, title: str):
        """
        Print a large Rich Panel banner for mode transitions.

        Args:
            mode: "TRAIN" or "EVAL"
            title: Banner text, e.g. "TRAINING BEGINS"
        """
        if self._is_silenced():
            return
        style = "train" if mode == "TRAIN" else "eval"
        panel = Panel(
            f"[bold]{title}[/bold]",
            style=style,
            border_style=style,
            width=72,
            padding=(1, 2),
        )
        with self._lock:
            if self.verbose:
                self.console.print(panel)
            self._write_plain_text(
                f"\n{'█' * 70}\n"
                f"█ {title.center(66)} █\n"
                f"{'█' * 70}\n\n"
            )

    def print_episode_header(
        self,
        episode_num: int,
        mode: str,
        problem_desc: str,
        population_size: int,
    ):
        """
        Print a Rich Panel header for an episode.

        Args:
            episode_num: Current episode number
            mode: "TRAIN" or "EVAL"
            problem_desc: Short problem description
            population_size: Number of agents
        """
        if self._is_silenced():
            return
        style = "train" if mode == "TRAIN" else "eval"
        frozen_line = "\n🔒 FROZEN: No wealth updates, no payments, no births/bankruptcies" if mode == "EVAL" else ""

        body = (
            f"📚 Episode {episode_num} [{mode}]\n"
            f"   Problem: {problem_desc}\n"
            f"   Population: {population_size} agents"
            f"{frozen_line}"
        )
        panel = Panel(body, border_style=style, width=72)

        with self._lock:
            if self.verbose:
                self.console.print(panel)
            text = (
                f"\n{'═' * 70}\n"
                f"📚 EPISODE {episode_num} [{mode}]\n"
                f"   Problem: {problem_desc}\n"
                f"   Population: {population_size} agents\n"
            )
            if mode == "EVAL":
                text += "   🔒 FROZEN: No wealth updates, no payments, no births/bankruptcies\n"
            text += f"{'═' * 70}\n"
            self._write_plain_text(text)

    def print_population_table(self, agents: list):
        """
        Print a Rich Table of the current population.

        Args:
            agents: List of agents (must have .id, .name, .wealth, .get_bid() methods)
        """
        if self._is_silenced():
            return
        table = Table(title="Population Status", width=72)
        table.add_column("ID", style="dim", width=6)
        table.add_column("Name", width=35)
        table.add_column("Wealth", justify="right", width=12)
        table.add_column("Bid", justify="right", width=10)

        for agent in agents:
            bid_str = f"${agent.get_bid():.2f}" if agent.get_bid() else "-"
            table.add_row(
                str(agent.id),
                agent.name,
                f"${agent.wealth:.2f}",
                bid_str,
            )

        with self._lock:
            if self.verbose:
                self.console.print(table)
            text = (
                f"\n{'═' * 70}\n"
                "📊 POPULATION STATUS\n"
                f"{'═' * 70}\n"
                f"{'ID':<6} {'Name':<35} {'Wealth':>12} {'Bid':>10}\n"
                f"{'─' * 70}\n"
            )
            for agent in agents:
                bid_str = f"${agent.get_bid():.2f}" if agent.get_bid() else "-"
                text += f"{agent.id:<6} {agent.name:<35} ${agent.wealth:>10.2f} {bid_str:>10}\n"
            text += f"{'═' * 70}\n"
            self._write_plain_text(text)

    def print_credit_check(self, agents_status: List[Tuple[str, int, float, bool]]):
        """
        Print a mini Rich Table for the credit check.

        Args:
            agents_status: List of (name, id, wealth, is_bankrupt) tuples
        """
        if self._is_silenced():
            return
        table = Table(title="Credit Check", width=60)
        table.add_column("Agent", width=30)
        table.add_column("Wealth", justify="right", width=12)
        table.add_column("Status", justify="center", width=12)

        for name, agent_id, wealth, is_bankrupt in agents_status:
            status_str = "[bankrupt]💀 BANKRUPT[/bankrupt]" if is_bankrupt else "[solvent]✓ solvent[/solvent]"
            table.add_row(f"{name} {agent_id}", f"${wealth:.2f}", status_str)

        with self._lock:
            if self.verbose:
                self.console.print(table)
            text = "\n🔍 CREDIT CHECK:\n"
            for name, agent_id, wealth, is_bankrupt in agents_status:
                status = "💀 BANKRUPT" if is_bankrupt else "✓ solvent"
                text += f"   {name} {agent_id}: ${wealth:.2f} {status}\n"
            self._write_plain_text(text)

    # ───────────────────────────────────────────────────────────────────
    # Lifecycle
    # ───────────────────────────────────────────────────────────────────

    def close(self):
        """Close the log file."""
        if self.task_log_file:
            self.task_log_file.close()
            self.task_log_file = None
        if self.log_file:
            self.log_file.write(f"\n{'=' * 70}\n")
            self.log_file.write(f"Log ended at {datetime.now().isoformat()}\n")
            self.log_file.close()
            self.log_file = None

    def __del__(self):
        """Best-effort cleanup for the underlying log file handle."""
        self.close()


# Global singleton instance
logger = HayekLogger()
