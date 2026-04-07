import argparse
import csv
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = BASE_DIR / "pipeline3"
DEFAULT_MATRIX = PIPELINE_DIR / "label_matrix_500_random_seed456.csv"
DEFAULT_REFERENCE = PIPELINE_DIR / "reference_digits_0_to_9_labeled.png"
BENCHMARK_SCRIPT = BASE_DIR / "pipeline3_openrouter_benchmark.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Run 5 OpenRouter benchmark workers in parallel with different API keys")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--model", type=str, default="google/gemma-4-31b-it", help="Model id")
    parser.add_argument("--allow-any-model", action="store_true", help="Forward override to benchmark script")
    parser.add_argument("--matrix-path", type=str, default=str(DEFAULT_MATRIX), help="Path to source matrix CSV (10 x N)")
    parser.add_argument("--reference-image-path", type=str, default=str(DEFAULT_REFERENCE), help="Path to labeled reference image")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size per benchmark worker")
    parser.add_argument("--resize", type=int, default=224, help="Resize edge")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max completion tokens")
    parser.add_argument("--request-timeout", type=float, default=90.0, help="Request timeout seconds")
    parser.add_argument("--max-retries-timeout", type=int, default=1, help="Retries on network/timeout")
    parser.add_argument("--max-retries-429", type=int, default=2, help="Retries on 429")
    parser.add_argument("--max-retry-wait", type=float, default=6.0, help="Max wait between retries")
    parser.add_argument("--delay", type=float, default=0.4, help="Delay between requests")
    parser.add_argument("--compact-prompt", action="store_true", help="Use compact prompt")
    parser.add_argument("--key-prefix", type=str, default="OPENROUTER_API_KEY", help="API key env var prefix")
    parser.add_argument("--output", type=str, default="parallel5_merged_results.csv", help="Merged per-image output filename under pipeline3/")
    parser.add_argument("--output-batches", type=str, default="parallel5_merged_batches.csv", help="Merged per-batch output filename under pipeline3/")
    return parser.parse_args()


def find_key_vars(prefix: str, workers: int):
    numbered = [f"{prefix}{i}" for i in range(1, workers + 1)]
    found = [name for name in numbered if os.getenv(name, "").strip()]
    if len(found) >= workers:
        return found[:workers]

    # Fallback to include base key if needed.
    base = prefix
    if os.getenv(base, "").strip() and base not in found:
        found.append(base)

    if len(found) < workers:
        raise RuntimeError(
            f"Need {workers} API keys, found {len(found)}. "
            f"Set {prefix}1..{prefix}{workers} (or include {prefix})."
        )

    return found[:workers]


def split_matrix(matrix_path: Path, workers: int, shard_dir: Path):
    matrix = np.loadtxt(matrix_path, delimiter=",", dtype=int)
    if matrix.ndim == 1:
        matrix = matrix.reshape(10, -1)

    cols = matrix.shape[1]
    if cols < workers:
        raise ValueError(f"Matrix has only {cols} columns, cannot split across {workers} workers")

    split_points = np.array_split(np.arange(cols), workers)
    shard_paths = []

    shard_dir.mkdir(parents=True, exist_ok=True)
    for i, idxs in enumerate(split_points, start=1):
        shard = matrix[:, idxs]
        shard_path = shard_dir / f"matrix_shard_{i}.csv"
        np.savetxt(shard_path, shard, fmt="%d", delimiter=",")
        shard_paths.append((shard_path, shard.shape[1]))

    return shard_paths


def run_workers(args, key_vars, shard_info, run_dir: Path):
    procs = []
    worker_outputs = []
    stream_threads = []

    def stream_output(worker_idx, proc):
        prefix = f"[{worker_idx}]"
        if proc.stdout is None:
            return
        for line in proc.stdout:
            text = line.rstrip("\n")
            print(f"{prefix} {text}", flush=True)

    for i in range(args.workers):
        key_var = key_vars[i]
        shard_path, samples_per_digit = shard_info[i]
        worker_out = f"parallel_worker_{i+1}.csv"
        worker_outputs.append(worker_out)

        cmd = [
            sys.executable,
            str(BENCHMARK_SCRIPT),
            "--model",
            args.model,
            "--samples-per-digit",
            str(samples_per_digit),
            "--matrix-path",
            str(shard_path),
            "--reference-image-path",
            args.reference_image_path,
            "--batch-size",
            str(args.batch_size),
            "--resize",
            str(args.resize),
            "--max-tokens",
            str(args.max_tokens),
            "--request-timeout",
            str(args.request_timeout),
            "--max-retries-timeout",
            str(args.max_retries_timeout),
            "--max-retries-429",
            str(args.max_retries_429),
            "--max-retry-wait",
            str(args.max_retry_wait),
            "--delay",
            str(args.delay),
            "--api-key-env-var",
            key_var,
            "--output",
            worker_out,
        ]

        if args.compact_prompt:
            cmd.append("--compact-prompt")
        if args.allow_any_model:
            cmd.append("--allow-any-model")

        print(f"[START] worker {i+1}/{args.workers} key={key_var} samples_per_digit={samples_per_digit}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        procs.append(proc)

        t = threading.Thread(target=stream_output, args=(i + 1, proc), daemon=True)
        t.start()
        stream_threads.append(t)

    exit_codes = []
    for i, proc in enumerate(procs, start=1):
        code = proc.wait()
        exit_codes.append(code)
        print(f"[DONE] worker {i}/{args.workers} exit_code={code}")

    for t in stream_threads:
        t.join(timeout=1.0)

    return worker_outputs, exit_codes


def merge_csvs(input_files, output_path: Path):
    header = None
    rows = []
    for f in input_files:
        path = PIPELINE_DIR / f
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            file_header = next(reader, None)
            if file_header is None:
                continue
            if header is None:
                header = file_header
            rows.extend(list(reader))

    if header is None:
        raise RuntimeError("No worker CSVs were found to merge")

    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    args = parse_args()
    load_dotenv(BASE_DIR / ".env", override=True)

    key_vars = find_key_vars(args.key_prefix, args.workers)

    run_dir = PIPELINE_DIR / "parallel5_run"
    shard_dir = run_dir / "shards"
    run_dir.mkdir(parents=True, exist_ok=True)

    shard_info = split_matrix(Path(args.matrix_path), args.workers, shard_dir)
    worker_outputs, exit_codes = run_workers(args, key_vars, shard_info, run_dir)

    merged_out = PIPELINE_DIR / args.output
    merge_csvs(worker_outputs, merged_out)

    worker_batch_outputs = [f.replace(".csv", "_batches.csv") for f in worker_outputs]
    merged_batches_out = PIPELINE_DIR / args.output_batches
    merge_csvs(worker_batch_outputs, merged_batches_out)

    failed = [c for c in exit_codes if c != 0]
    print("\n===== PARALLEL SUMMARY =====")
    print(f"workers={args.workers}, failed_workers={len(failed)}")
    print(f"merged_saved={merged_out}")
    print(f"merged_batches_saved={merged_batches_out}")


if __name__ == "__main__":
    main()
