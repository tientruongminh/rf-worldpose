# preprocessing/preprocess_pipeline.py

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# Change these imports to match your actual folder names.
from datasets.mmfi_loader import MMFiWiFiCSIDataset
from datasets.wifipose_loader import PersonInWiFi3DDataset
from datasets.uthar_loader import UTHARDataset
from datasets.wiar_loader import WiARDataset
from datasets.wimans_loader import WiMANSDataset
from datasets.wipose_loader import WiPoseDataset


LOGGER = logging.getLogger("preprocess")


@dataclass
class PreprocessConfig:
    dataset_name: str
    raw_root: str
    output_root: str
    split: Optional[str] = None
    normalize: bool = True
    overwrite: bool = False
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)


class ProcessedCSIDataset(Dataset):
    """
    Dataset for loading preprocessed .pt samples.

    Each .pt file stores:

        {
            "x": tensor,
            "target": ...,
            "metadata": ...
        }
    """

    def __init__(self, processed_root: str | Path):
        self.processed_root = Path(processed_root)
        self.sample_dir = self.processed_root / "samples"
        self.files = sorted(self.sample_dir.glob("*.pt"))

        if len(self.files) == 0:
            raise RuntimeError(f"No .pt samples found in {self.sample_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return torch.load(self.files[index], map_location="cpu")


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "preprocess.log", encoding="utf-8"),
        ],
    )


def build_dataset(config: PreprocessConfig) -> Dataset:
    name = config.dataset_name.lower()

    kwargs = {
        "root": config.raw_root,
        "normalize": config.normalize,
        **config.dataset_kwargs,
    }

    if name in {"mmfi", "mm-fi"}:
        return MMFiWiFiCSIDataset(**kwargs)

    if name in {"wipose", "wi-pose"}:
        return WiPoseDataset(
            split=config.split,
            **kwargs,
        )

    if name in {"personwifi", "person-wifi", "person-in-wifi", "person-in-wifi-3d"}:
        return PersonInWiFi3DDataset(
            split=config.split or "train",
            single_person_only=True,
            **kwargs,
        )

    if name == "wiar":
        return WiARDataset(**kwargs)

    if name == "wimans":
        return WiMANSDataset(
            single_person_only=True,
            **kwargs,
        )

    if name in {"uthar", "ut-har"}:
        return UTHARDataset(
            split=config.split or "train",
            **kwargs,
        )

    raise ValueError(f"Unknown dataset_name: {config.dataset_name}")


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def tensor_stats(x: torch.Tensor) -> Dict[str, Any]:
    x_float = x.float()

    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "mean": float(x_float.mean()),
        "std": float(x_float.std()),
        "min": float(x_float.min()),
        "max": float(x_float.max()),
    }


def normalize_sample_format(sample: Any, index: int) -> Dict[str, Any]:
    """
    Makes sure every dataset sample becomes:

        {
            "x": tensor,
            "target": ...,
            "metadata": ...
        }

    Your loaders already return this format, but this keeps the pipeline safe.
    """

    if isinstance(sample, dict):
        if "x" not in sample:
            raise KeyError(f"Sample {index} is missing key 'x'.")

        sample.setdefault("target", {})
        sample.setdefault("metadata", {})

        return sample

    if isinstance(sample, tuple) and len(sample) == 2:
        x, target = sample

        return {
            "x": x,
            "target": target,
            "metadata": {"index": index},
        }

    raise TypeError(
        f"Unsupported sample type at index {index}: {type(sample)}. "
        "Expected dict or (x, target)."
    )


def preprocess_dataset(config: PreprocessConfig) -> Path:
    output_dir = Path(config.output_root) / config.dataset_name
    sample_dir = output_dir / "samples"
    manifest_path = output_dir / "manifest.json"

    setup_logging(output_dir)

    if sample_dir.exists() and manifest_path.exists() and not config.overwrite:
        LOGGER.info("Processed cache already exists: %s", output_dir)
        LOGGER.info("Use overwrite=True to regenerate.")
        return output_dir

    sample_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving config")
    save_json(output_dir / "config.json", asdict(config))

    LOGGER.info("Building dataset: %s", config.dataset_name)
    dataset = build_dataset(config)

    LOGGER.info("Dataset size: %d", len(dataset))

    manifest = {
        "dataset_name": config.dataset_name,
        "raw_root": config.raw_root,
        "split": config.split,
        "num_samples": len(dataset),
        "samples": [],
        "failed": [],
    }

    for index in tqdm(range(len(dataset)), desc=f"Preprocessing {config.dataset_name}"):
        try:
            sample = dataset[index]
            sample = normalize_sample_format(sample, index)

            x = sample["x"]

            if not torch.is_tensor(x):
                x = torch.as_tensor(x)

            output_sample = {
                "x": x.cpu(),
                "target": sample["target"],
                "metadata": sample["metadata"],
            }

            output_path = sample_dir / f"{index:08d}.pt"
            torch.save(output_sample, output_path)

            manifest["samples"].append(
                {
                    "index": index,
                    "path": str(output_path),
                    "x": tensor_stats(x),
                    "metadata": sample["metadata"],
                }
            )

        except Exception as exc:
            LOGGER.exception("Failed preprocessing sample index=%d", index)

            manifest["failed"].append(
                {
                    "index": index,
                    "error": repr(exc),
                }
            )

    save_json(manifest_path, manifest)

    LOGGER.info("Finished preprocessing")
    LOGGER.info("Saved samples: %d", len(manifest["samples"]))
    LOGGER.info("Failed samples: %d", len(manifest["failed"]))
    LOGGER.info("Output directory: %s", output_dir)

    return output_dir


def load_config(path: str | Path) -> PreprocessConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return PreprocessConfig(**data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--raw-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="data/silver")
    parser.add_argument("--split", type=str, default=None)

    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config is not None:
        config = load_config(args.config)

    else:
        if args.dataset_name is None:
            raise ValueError("--dataset-name is required if --config is not used.")

        if args.raw_root is None:
            raise ValueError("--raw-root is required if --config is not used.")

        normalize = True

        if args.no_normalize:
            normalize = False

        if args.normalize:
            normalize = True

        config = PreprocessConfig(
            dataset_name=args.dataset_name,
            raw_root=args.raw_root,
            output_root=args.output_root,
            split=args.split,
            normalize=normalize,
            overwrite=args.overwrite,
        )

    output_dir = preprocess_dataset(config)

    print(f"Processed data saved to: {output_dir}")


if __name__ == "__main__":
    main()

