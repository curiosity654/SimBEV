#!/usr/bin/env python3
# Academic Software License: Copyright © 2026 Goodarz Mehr.
#
# Multi-worker launcher for SimBEV. This script spawns multiple independent
# SimBEV processes, each assigned a shard of scenes via --num-workers and
# --worker-id. Each worker writes to its own output subdirectory to avoid
# write conflicts.

import argparse
import os
import subprocess
import sys
from typing import List


def _build_worker_command(
    simbev_entry: str,
    config_path: str,
    base_output: str,
    worker_id: int,
    num_workers: int,
    extra_args: List[str],
    render_gpu: int | None = None,
) -> List[str]:
    worker_output = os.path.join(base_output, f"worker_{worker_id}")
    cmd = [
        sys.executable,
        simbev_entry,
        config_path,
        "--path",
        worker_output,
        "--num-workers",
        str(num_workers),
        "--worker-id",
        str(worker_id),
    ]
    if render_gpu is not None:
        cmd.extend(["--render-gpu", str(render_gpu)])
    cmd.extend(extra_args)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch multiple parallel SimBEV workers."
    )
    parser.add_argument("config", help="SimBEV configuration file")
    parser.add_argument(
        "--simbev-entry",
        default=os.path.join(os.path.dirname(__file__), "simbev.py"),
        help="Path to simbev.py entrypoint",
    )
    parser.add_argument(
        "--output-root",
        default="/dataset",
        help="Root output directory; workers write to output-root/worker_{id}",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Total number of parallel workers",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated GPU ids for round-robin assignment (e.g., 0,1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    # Extra args for simbev.py are captured via parse_known_args below.

    args, extra_args = parser.parse_known_args()

    if args.num_workers < 1:
        print("Error: --num-workers must be >= 1", file=sys.stderr)
        return 2

    simbev_entry = os.path.abspath(args.simbev_entry)
    config_path = os.path.abspath(args.config)
    output_root = os.path.abspath(args.output_root)

    if not os.path.exists(simbev_entry):
        print(f"Error: simbev entry not found: {simbev_entry}", file=sys.stderr)
        return 2
    if not os.path.exists(config_path):
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 2

    os.makedirs(output_root, exist_ok=True)

    # If the user passes a lone '--', drop it from passthrough args.
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
        if not gpu_ids:
            print("Error: --gpu-ids must contain at least one GPU id", file=sys.stderr)
            return 2

    procs = []
    for worker_id in range(args.num_workers):
        render_gpu = None
        if gpu_ids:
            render_gpu = gpu_ids[worker_id % len(gpu_ids)]
        cmd = _build_worker_command(
            simbev_entry=simbev_entry,
            config_path=config_path,
            base_output=output_root,
            worker_id=worker_id,
            num_workers=args.num_workers,
            extra_args=extra_args,
            render_gpu=render_gpu,
        )
        print(f"[launcher] worker {worker_id}: {' '.join(cmd)}")
        if not args.dry_run:
            procs.append(subprocess.Popen(cmd))

    if args.dry_run:
        return 0

    # Wait for all workers to finish.
    exit_code = 0
    for proc in procs:
        code = proc.wait()
        if code != 0:
            exit_code = code

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
