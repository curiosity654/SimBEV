# Parallel Generation and Train/Val-Only Runs

This document explains how to run **multi-worker parallel generation** and how to generate **only train + val** splits in SimBEV.

---

## 1) Multi-worker generation (parallel)

SimBEV supports parallel generation by **sharding scenes across workers**. Each worker:
- launches its own CARLA server
- generates a disjoint subset of scenes
- writes to its own output directory to avoid collisions

### Recommended: use the launcher

The launcher starts multiple workers and assigns:
- `--num-workers` (total workers)
- `--worker-id` (worker index)

Each worker writes to:
```
<output-root>/worker_<id>/simbev/...
```

Example (4 workers):
```
python -m simbev.launcher configs/config_2hz.yaml --output-root /dataset --num-workers 4 -- --render
```

### Multi-GPU assignment (round-robin)
If you have multiple GPUs, you can assign workers to GPUs in a round-robin fashion:

```
python -m simbev.launcher configs/config_2hz.yaml --output-root /dataset --num-workers 4 --gpu-ids 0,1 -- --render
```

Notes:
- `--` separates launcher args from `simbev.py` args.
- All scene IDs are distributed by `scene_id % num_workers`.
- `--gpu-ids` applies `--render-gpu` per worker (e.g., workers 0/2 → GPU 0, workers 1/3 → GPU 1).

---

## 2) Consolidate outputs into a single dataset

After all workers finish, you can merge metadata and **consolidate files into a single `simbev/` layout**, identical to the `num_workers=1` structure.

### Copy (keeps worker directories)
```
python -m simbev.merge_infos --output-root /dataset --consolidate --consolidate-mode copy
```

### Move (saves disk space)
```
python -m simbev.merge_infos --output-root /dataset --consolidate --consolidate-mode move
```

This will:
- copy/move all worker files into `/dataset/simbev/...`
- rewrite paths in `simbev_infos_{split}.json` to point to the consolidated root
- write merged `simbev_infos_train/val/test.json` under `/dataset/simbev/infos`

---

## 3) Generate only train + val (skip test)

In `create` mode, SimBEV always loops over `train/val/test`, but **you can disable a split by setting its scene config to empty**.

In your config (YAML):
```yaml
test_scene_config: {}
```
or:
```yaml
test_scene_config: null
```

This makes the `test` split a no-op (no scenes generated, no `simbev_infos_test.json`).

---

## 4) Single worker with sharding flags (optional)

You can also run a specific worker directly without the launcher:

```
python -m simbev.simbev configs/config_2hz.yaml \
  --path /dataset/worker_0 \
  --num-workers 4 \
  --worker-id 0
```

---

## 5) Recommended workflow

1. Run launcher with N workers.
2. Wait for all workers to finish.
3. Consolidate outputs with `merge_infos --consolidate`.

This gives you the **same output layout** as a single-worker run, but with higher throughput.