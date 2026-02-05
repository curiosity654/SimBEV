#!/usr/bin/env python3
# Academic Software License: Copyright © 2026 Goodarz Mehr.
#
# Merge utility for SimBEV infos JSON files produced by multiple workers.

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List, Tuple

SPLITS = ("train", "val", "test")


def _find_worker_dirs(output_root: str) -> List[str]:
    worker_dirs = []
    if not os.path.isdir(output_root):
        return worker_dirs
    for entry in sorted(os.listdir(output_root)):
        if not entry.startswith("worker_"):
            continue
        worker_dir = os.path.join(output_root, entry)
        if os.path.isdir(worker_dir):
            worker_dirs.append(worker_dir)
    return worker_dirs


def _build_replacements(output_root: str) -> Dict[str, str]:
    target_root = os.path.join(output_root, "simbev")
    replacements = {}
    for worker_dir in _find_worker_dirs(output_root):
        worker_simbev = os.path.join(worker_dir, "simbev")
        if os.path.isdir(worker_simbev):
            replacements[worker_simbev] = target_root
    return replacements


def _rewrite_paths(obj, replacements: Dict[str, str]):
    if isinstance(obj, dict):
        return {k: _rewrite_paths(v, replacements) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite_paths(v, replacements) for v in obj]
    if isinstance(obj, str):
        for old, new in replacements.items():
            if obj.startswith(old):
                return new + obj[len(old) :]
        return obj
    return obj


def _consolidate_worker_outputs(
    output_root: str,
    mode: str,
    allow_overwrite: bool,
    dry_run: bool,
) -> str:
    target_root = os.path.join(output_root, "simbev")
    for worker_dir in _find_worker_dirs(output_root):
        src_root = os.path.join(worker_dir, "simbev")
        if not os.path.isdir(src_root):
            continue
        for root, dirs, files in os.walk(src_root):
            rel = os.path.relpath(root, src_root)
            dst_root = target_root if rel == "." else os.path.join(target_root, rel)
            if not dry_run:
                os.makedirs(dst_root, exist_ok=True)
            for dname in dirs:
                if not dry_run:
                    os.makedirs(os.path.join(dst_root, dname), exist_ok=True)
            for fname in files:
                src = os.path.join(root, fname)
                dst = os.path.join(dst_root, fname)
                if os.path.exists(dst) and not allow_overwrite:
                    raise ValueError(
                        f"Collision while consolidating: {dst} already exists."
                    )
                if dry_run:
                    print(f"[consolidate] {src} -> {dst}")
                else:
                    if mode == "move":
                        shutil.move(src, dst)
                    else:
                        shutil.copy2(src, dst)
    return target_root


def _find_worker_info_paths(output_root: str, split: str) -> List[str]:
    paths = []
    if not os.path.isdir(output_root):
        return paths
    for entry in sorted(os.listdir(output_root)):
        if not entry.startswith("worker_"):
            continue
        worker_dir = os.path.join(output_root, entry)
        info_path = os.path.join(
            worker_dir, "simbev", "infos", f"simbev_infos_{split}.json"
        )
        if os.path.isfile(info_path):
            paths.append(info_path)
    return paths


def _load_infos(path: str) -> Tuple[Dict, Dict]:
    with open(path, "r") as f:
        payload = json.load(f)
    metadata = payload.get("metadata", {})
    data = payload.get("data", {})
    return metadata, data


def _merge_split(output_root: str, split: str) -> Dict:
    info_paths = _find_worker_info_paths(output_root, split)
    if not info_paths:
        return {}

    merged_metadata = {}
    merged_data: Dict[str, Dict] = {}

    for path in info_paths:
        metadata, data = _load_infos(path)
        if not merged_metadata and metadata:
            merged_metadata = metadata

        for scene_key, scene_data in data.items():
            if scene_key in merged_data:
                raise ValueError(
                    f"Duplicate scene key {scene_key} found in {path}. "
                    "Make sure workers write to separate output roots."
                )
            merged_data[scene_key] = scene_data

    # Sort scenes by numeric suffix for stable output.
    sorted_items = sorted(
        merged_data.items(),
        key=lambda kv: int(kv[0].split("_")[1]) if "_" in kv[0] else kv[0],
    )
    merged_data = dict(sorted_items)

    return {"metadata": merged_metadata, "data": merged_data}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge SimBEV infos JSON files from workers."
    )
    parser.add_argument(
        "--output-root",
        default="/dataset",
        help="Root output directory containing worker_* subfolders.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for merged infos (default: output-root/simbev/infos).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any split is missing in all workers.",
    )
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Consolidate worker outputs into output-root/simbev and rewrite paths.",
    )
    parser.add_argument(
        "--consolidate-mode",
        choices=("copy", "move"),
        default="copy",
        help="How to consolidate files: copy (default) or move.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting files during consolidation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print consolidation actions without modifying files.",
    )
    args = parser.parse_args()

    output_root = os.path.abspath(args.output_root)
    output_dir = (
        os.path.abspath(args.output)
        if args.output
        else os.path.join(output_root, "simbev", "infos")
    )

    merged_by_split: Dict[str, Dict] = {}
    for split in SPLITS:
        merged = _merge_split(output_root, split)
        if not merged:
            if args.strict:
                print(f"Error: no infos found for split '{split}'.", file=sys.stderr)
                return 2
            continue
        merged_by_split[split] = merged

    if not merged_by_split:
        print("Error: no worker infos found to merge.", file=sys.stderr)
        return 2

    if args.consolidate:
        _consolidate_worker_outputs(
            output_root=output_root,
            mode=args.consolidate_mode,
            allow_overwrite=args.allow_overwrite,
            dry_run=args.dry_run,
        )
        replacements = _build_replacements(output_root)
        for split in list(merged_by_split.keys()):
            merged_by_split[split] = _rewrite_paths(
                merged_by_split[split], replacements
            )

    os.makedirs(output_dir, exist_ok=True)
    for split in SPLITS:
        if split not in merged_by_split:
            continue
        output_path = os.path.join(output_dir, f"simbev_infos_{split}.json")
        with open(output_path, "w") as f:
            json.dump(merged_by_split[split], f, indent=4)
        print(f"[merge] wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
