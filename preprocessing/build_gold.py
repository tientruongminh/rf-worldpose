# preprocessing/build_gold.py

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


POSE_DATASETS = {
    "mmfi",
    "mm-fi",
    "wipose",
    "personwifi",
}

ACTIVITY_DATASETS = {
    "mmfi",
    "wiar",
    "wimans",
    "uthar",
    "wipose",
}

POSE_METADATA_KEYS = {
    "pose_path",
    "keypoint_path",
    "skeleton_path",
}

ACTIVITY_METADATA_KEYS = {
    "action_name",
    "activity_name",
    "label",
}


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def normalize_dataset_name(dataset_name: str) -> str:
    return dataset_name.lower().replace("_", "-")


def contains_any_key(data: Dict[str, Any], keys: set[str]) -> bool:
    return any(key in data and data[key] not in {None, ""} for key in keys)


def has_pose(dataset_name: str, sample: Dict[str, Any]) -> bool:
    dataset = normalize_dataset_name(dataset_name)
    metadata = sample.get("metadata", {})

    if dataset in POSE_DATASETS:
        return True

    if contains_any_key(metadata, POSE_METADATA_KEYS):
        return True

    return False


def has_activity(dataset_name: str, sample: Dict[str, Any]) -> bool:
    dataset = normalize_dataset_name(dataset_name)
    metadata = sample.get("metadata", {})

    if dataset in ACTIVITY_DATASETS:
        return True

    if contains_any_key(metadata, ACTIVITY_METADATA_KEYS):
        return True

    return False


def infer_tasks(dataset_name: str, sample: Dict[str, Any]) -> List[str]:
    """
    Infer all tasks available for one sample.

    A sample can support multiple tasks, for example:

        ["pose", "activity"]

    This replaces the old single-task infer_task() logic.
    """

    tasks = []

    if has_pose(dataset_name, sample):
        tasks.append("pose")

    if has_activity(dataset_name, sample):
        tasks.append("activity")

    if not tasks:
        tasks.append("unknown")

    return tasks


def infer_task(dataset_name: str, sample: Dict[str, Any]) -> str:
    """
    Backward-compatible helper.

    Old code expected one task string.
    New code should use infer_tasks().
    """

    return infer_tasks(dataset_name, sample)[0]


def sample_matches_task_filter(tasks: List[str], task_filter: Optional[str]) -> bool:
    if task_filter is None:
        return True

    return task_filter in tasks


def load_silver_samples(
    silver_root: Path,
    dataset_name: str,
    task_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    manifest_path = silver_root / dataset_name / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing silver manifest: {manifest_path}")

    manifest = load_json(manifest_path)
    samples = []

    for sample in manifest["samples"]:
        tasks = infer_tasks(dataset_name, sample)

        if not sample_matches_task_filter(tasks, task_filter):
            continue

        # Keep old "task" field for compatibility, but add new "tasks" list.
        samples.append(
            {
                "dataset": dataset_name,
                "task": tasks[0],
                "tasks": tasks,
                "path": sample["path"],
                "metadata": sample.get("metadata", {}),
                "x": sample.get("x", {}),
            }
        )

    return samples


def split_samples(
    samples: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    samples = samples[:]
    random.Random(seed).shuffle(samples)

    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }


def build_gold(
    silver_root: str | Path,
    gold_root: str | Path,
    experiment_name: str,
    datasets: List[str],
    task_filter: Optional[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Path:
    silver_root = Path(silver_root)
    gold_dir = Path(gold_root) / experiment_name

    all_samples = []

    for dataset_name in datasets:
        dataset_samples = load_silver_samples(
            silver_root=silver_root,
            dataset_name=dataset_name,
            task_filter=task_filter,
        )

        print(f"{dataset_name}: {len(dataset_samples)} samples")

        all_samples.extend(dataset_samples)

    if len(all_samples) == 0:
        raise RuntimeError("No samples found for gold dataset.")

    splits = split_samples(
        samples=all_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    save_json(gold_dir / "train_manifest.json", splits["train"])
    save_json(gold_dir / "val_manifest.json", splits["val"])
    save_json(gold_dir / "test_manifest.json", splits["test"])

    config = {
        "experiment_name": experiment_name,
        "silver_root": str(silver_root),
        "gold_root": str(gold_dir),
        "datasets": datasets,
        "task_filter": task_filter,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio,
        "seed": seed,
        "num_train": len(splits["train"]),
        "num_val": len(splits["val"]),
        "num_test": len(splits["test"]),
    }

    save_json(gold_dir / "dataset_config.json", config)

    print("\nGold dataset saved to:", gold_dir)
    print("Train:", len(splits["train"]))
    print("Val:", len(splits["val"]))
    print("Test:", len(splits["test"]))

    return gold_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--silver-root", type=str, default="data/silver")
    parser.add_argument("--gold-root", type=str, default="data/gold")
    parser.add_argument("--experiment-name", type=str, required=True)

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset folder names inside data/silver.",
    )

    parser.add_argument(
        "--task-filter",
        type=str,
        default=None,
        choices=["pose", "activity", "unknown"],
        help="Optional task filter. Includes samples that contain this task.",
    )

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_gold(
        silver_root=args.silver_root,
        gold_root=args.gold_root,
        experiment_name=args.experiment_name,
        datasets=args.datasets,
        task_filter=args.task_filter,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()