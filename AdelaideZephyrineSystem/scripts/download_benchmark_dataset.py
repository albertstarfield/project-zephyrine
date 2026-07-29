#!/usr/bin/env python3
"""
Download benchmark datasets from HuggingFace using the datasets library.
Converts any format (parquet, json, csv) to JSONL for the benchmark runner.

Usage:
    python3 download_benchmark_dataset.py <repo_id> <subset> <output_dir> <split>

    repo_id:   HuggingFace repo (e.g. "cais/mmlu")
    subset:    Dataset subset/config (e.g. "abstract_algebra") or "_" for no subset
    output_dir: Directory to store downloaded files
    split:     Dataset split (e.g. "test") or "test" as default

Returns:
    Full path to downloaded JSONL file on success, exits with code 1 on failure.
"""
import json
import sys
from pathlib import Path


def download_dataset(repo_id: str, subset: str, output_dir: str, split: str) -> str:
    """Download a dataset from HuggingFace and convert to JSONL."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[Benchmark-Py] ERROR: datasets library not installed. Run: pip install datasets", file=sys.stderr)
        return ""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build output filename
    subset_tag = subset if subset != "_" else "default"
    output_file = output_path / f"{repo_id.replace('/', '_')}_{subset_tag}_{split}.jsonl"

    # If already cached, return immediately
    if output_file.exists() and output_file.stat().st_size > 0:
        print(f"[Benchmark-Py] Cached: {output_file} ({output_file.stat().st_size} bytes)")
        return str(output_file)

    try:
        print(f"[Benchmark-Py] Loading {repo_id} (subset={subset}, split={split})...")

        # Load dataset
        if subset == "_":
            ds = load_dataset(repo_id, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(repo_id, subset, split=split, trust_remote_code=True)

        print(f"[Benchmark-Py] Loaded {len(ds)} examples. Converting to JSONL...")

        # Write as JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, example in enumerate(ds):
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
                if (i + 1) % 1000 == 0:
                    print(f"[Benchmark-Py]   ... {i + 1}/{len(ds)} examples written")

        print(f"[Benchmark-Py] Done: {output_file} ({output_file.stat().st_size} bytes, {len(ds)} examples)")
        return str(output_file)

    except Exception as e:
        print(f"[Benchmark-Py] ERROR: {e}", file=sys.stderr)
        return ""


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: download_benchmark_dataset.py <repo_id> <subset> <output_dir> [split]")
        sys.exit(1)

    repo_id = sys.argv[1]
    subset = sys.argv[2]
    output_dir = sys.argv[3]
    split = sys.argv[4] if len(sys.argv) > 4 else "test"

    result = download_dataset(repo_id, subset, output_dir, split)
    if result:
        print(result)
        sys.exit(0)
    else:
        sys.exit(1)
