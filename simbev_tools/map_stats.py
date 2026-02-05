#!/usr/bin/env python3
"""
Compute CARLA map lane-length statistics.

This script connects to a running CARLA server, loads the requested Town maps,
and approximates the total lane length by sampling map waypoints at a fixed
resolution and summing distances along each unique lane.

Notes:
- Total lane length is the sum of lengths of all lanes (each lane_id counted).
- This is typically more meaningful for scene allocation than map area.
- The estimate depends on the sampling resolution (default: 1.0 m).

Usage examples:
  # Compute stats for a list of Towns
  python -m simbev_tools.map_stats --towns Town01,Town02,Town12

  # Use a config file to extract towns from *_scene_config keys
  python -m simbev_tools.map_stats --config configs/config_2hz_trainval.yaml

  # Save results to JSON
  python -m simbev_tools.map_stats --towns Town01,Town02 --json-out /tmp/map_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import carla
except Exception as exc:  # pragma: no cover - import error display
    print(
        "Failed to import carla. Ensure CARLA Python API is on PYTHONPATH.",
        file=sys.stderr,
    )
    raise

try:
    import yaml
except Exception as exc:  # pragma: no cover
    print("Failed to import yaml. Install pyyaml to use --config.", file=sys.stderr)
    yaml = None


@dataclass
class MapStats:
    town: str
    total_lane_length_m: float
    num_lanes: int
    num_waypoints: int
    resolution_m: float

    def to_dict(self) -> Dict:
        return {
            "town": self.town,
            "total_lane_length_m": self.total_lane_length_m,
            "num_lanes": self.num_lanes,
            "num_waypoints": self.num_waypoints,
            "resolution_m": self.resolution_m,
        }


def _extract_towns_from_config(config_path: str) -> List[str]:
    if yaml is None:
        raise RuntimeError("pyyaml is required to parse --config")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    towns = set()
    for key in ("train_scene_config", "val_scene_config", "test_scene_config"):
        scene_cfg = cfg.get(key)
        if isinstance(scene_cfg, dict):
            towns.update(scene_cfg.keys())
    return sorted(towns)


def _group_waypoints_by_lane(
    waypoints: Iterable["carla.Waypoint"],
) -> Dict[Tuple[int, int, int], List["carla.Waypoint"]]:
    """
    Group waypoints by (road_id, section_id, lane_id).
    """
    groups: Dict[Tuple[int, int, int], List["carla.Waypoint"]] = defaultdict(list)
    for wp in waypoints:
        groups[(wp.road_id, wp.section_id, wp.lane_id)].append(wp)
    return groups


def _lane_length_from_waypoints(waypoints: List["carla.Waypoint"]) -> float:
    """
    Approximate length of a lane by summing distances between sorted waypoints.
    """
    if len(waypoints) < 2:
        return 0.0
    waypoints.sort(key=lambda w: w.s)
    total = 0.0
    prev = waypoints[0].transform.location
    for wp in waypoints[1:]:
        loc = wp.transform.location
        total += loc.distance(prev)
        prev = loc
    return total


def compute_map_stats(
    client: "carla.Client",
    town: str,
    resolution_m: float,
) -> MapStats:
    """
    Load a map and compute total lane length using waypoint sampling.
    """
    client.load_world(town)
    world = client.get_world()
    carla_map = world.get_map()

    waypoints = carla_map.generate_waypoints(resolution_m)
    groups = _group_waypoints_by_lane(waypoints)

    total_lane_length = 0.0
    for lane_wps in groups.values():
        total_lane_length += _lane_length_from_waypoints(lane_wps)

    return MapStats(
        town=town,
        total_lane_length_m=total_lane_length,
        num_lanes=len(groups),
        num_waypoints=len(waypoints),
        resolution_m=resolution_m,
    )


def _print_table(stats: List[MapStats]) -> None:
    """
    Print a simple text table.
    """
    if not stats:
        print("No stats to display.")
        return
    headers = ["Town", "Total lane length (km)", "Lanes", "Waypoints", "Resolution (m)"]
    rows = []
    for s in stats:
        rows.append(
            [
                s.town,
                f"{s.total_lane_length_m / 1000.0:.2f}",
                str(s.num_lanes),
                str(s.num_waypoints),
                f"{s.resolution_m:.2f}",
            ]
        )
    col_widths = [
        max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)
    ]
    header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        print("  ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute CARLA map lane-length statistics."
    )
    parser.add_argument(
        "--host", default="localhost", help="CARLA host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=2000, help="CARLA port (default: 2000)"
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="Client timeout seconds"
    )
    parser.add_argument(
        "--towns",
        default="",
        help="Comma-separated list of towns (e.g., Town01,Town02).",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Path to config YAML; extracts towns from *_scene_config.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Waypoint sampling resolution in meters (default: 1.0)",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write JSON stats output.",
    )

    args = parser.parse_args()

    towns: List[str] = []
    if args.config:
        towns.extend(_extract_towns_from_config(args.config))
    if args.towns:
        towns.extend([t.strip() for t in args.towns.split(",") if t.strip()])
    towns = sorted(set(towns))

    if not towns:
        print("No towns specified. Use --towns or --config.", file=sys.stderr)
        return 2

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    results: List[MapStats] = []
    for town in towns:
        print(f"Loading {town}...")
        stats = compute_map_stats(client, town, args.resolution)
        results.append(stats)

    _print_table(results)

    if args.json_out:
        payload = [s.to_dict() for s in results]
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote JSON stats to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
