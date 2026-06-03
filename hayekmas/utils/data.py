"""
Generic data loading utilities.

Provides generic utilities for loading data from JSONL files.

Usage:
    from hayekmas.utils.data import load_jsonl, find_data_files

    data = load_jsonl("data/some_file.jsonl")
    files = find_data_files("data/finance", pattern="*.jsonl")
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union


def load_jsonl(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.

    Args:
        filepath: Path to the JSONL file

    Returns:
        List of dictionaries, one per line
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Failed to parse line {line_num} in {filepath}: {e}")

    return data


def find_data_files(
    data_dir: Union[str, Path],
    pattern: str = "*.jsonl"
) -> List[Path]:
    """
    Find all data files matching a pattern in a directory.

    Args:
        data_dir: Directory to search
        pattern: Glob pattern (default: *.jsonl)

    Returns:
        List of matching file paths
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        return []

    files = sorted(data_dir.glob(pattern))
    return files
