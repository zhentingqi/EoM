#!/usr/bin/env python3
"""
Global launcher for HayekMAS.

This file is the only top-level entrypoint. It reads a JSON config file
and delegates execution to the requested adapter runtime.
"""

import importlib
import json
from pathlib import Path
import sys


ADAPTER_RUNTIMES = {
    "arch_dse_world": "hayekmas.adapters.arch_dse_world.runtime",
    "cloudcast": "hayekmas.adapters.cloudcast.runtime",
    "researchworld": "hayekmas.adapters.researchworld.runtime",
}


def _load_config(config_path: str) -> dict:
    """Load a JSON config file.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        The configuration dictionary.
    """
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return raw


def main():
    """Launch the configured adapter runtime.

    Usage:
        python main.py <config.json>
    """
    if len(sys.argv) != 2:
        print("Usage: python main.py <config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    raw = _load_config(config_path)

    domain = raw.get("domain")
    if domain not in ADAPTER_RUNTIMES:
        raise ValueError(f"Unknown domain: {domain}")

    runtime_module = importlib.import_module(ADAPTER_RUNTIMES[domain])
    runtime_module.main(raw)


if __name__ == "__main__":
    main()
