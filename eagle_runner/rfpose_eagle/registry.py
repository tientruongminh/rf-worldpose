from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingPreset:
    config_name: str
    model_family: str
    task: str
    train_module: str
    dataset_version: str
    description: str
    gpus: int = 1
    cpus: int = 8
    mem: str = "64G"
    time_limit: str = "24:00:00"
    partition: str = "tesla"
    gpu_type: str = "h100"
    recommended: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_PRESET = TrainingPreset(
    config_name="wimose_mmfi17j_proto1_eagle",
    model_family="wimose",
    task="pose",
    train_module="rfpose.training.train_wimose",
    dataset_version="rfpose-humanlike-v2-proto1",
    description="Main MM-Fi Protocol 1 pose model: WiMoseNet Proto1, 17 joints, root-relative.",
    gpus=2,
    cpus=8,
    mem="96G",
    time_limit="24:00:00",
    recommended=True,
)


TRAINING_PRESETS: dict[str, TrainingPreset] = {
    DEFAULT_PRESET.config_name: DEFAULT_PRESET,
    "wimose_mmfi17j_proto1_action_only_eagle": TrainingPreset(
        config_name="wimose_mmfi17j_proto1_action_only_eagle",
        model_family="wimose",
        task="action",
        train_module="rfpose.training.train_wimose",
        dataset_version="rfpose-humanlike-v2-proto1",
        description="Action-only fine-tune on Proto1; research baseline, not the best action model.",
        gpus=1,
        time_limit="08:00:00",
    ),
    "rootrel_mmfi_eagle": TrainingPreset(
        config_name="rootrel_mmfi_eagle",
        model_family="rootrel-transformer",
        task="multitask",
        train_module="rfpose.training.train_v2",
        dataset_version="rfpose-unified-v2",
        description="CSITransformerPoseRootRel multitask baseline: pose + action on MM-Fi unified-v2.",
        gpus=1,
        time_limit="24:00:00",
    ),
    "rootrel_mmfi_pose_only_eagle": TrainingPreset(
        config_name="rootrel_mmfi_pose_only_eagle",
        model_family="rootrel-transformer",
        task="pose",
        train_module="rfpose.training.train_v2",
        dataset_version="rfpose-unified-v2",
        description="RootRel pose-only ablation for testing single-task Transformer pose regression.",
        gpus=1,
        time_limit="24:00:00",
    ),
    "rootrel_mmfi_action_only_eagle": TrainingPreset(
        config_name="rootrel_mmfi_action_only_eagle",
        model_family="rootrel-transformer",
        task="action",
        train_module="rfpose.training.train_v2",
        dataset_version="rfpose-unified-v2",
        description="RootRel action-only ablation using the unified-v2 action labels.",
        gpus=1,
        time_limit="24:00:00",
    ),
    "rootrel_mmfi_action_only_from_scratch_eagle": TrainingPreset(
        config_name="rootrel_mmfi_action_only_from_scratch_eagle",
        model_family="rootrel-transformer",
        task="action",
        train_module="rfpose.training.train_v2",
        dataset_version="rfpose-unified-v2",
        description="RootRel action-only model trained from scratch.",
        gpus=1,
        time_limit="24:00:00",
    ),
    "vit2d_mmfi_eagle": TrainingPreset(
        config_name="vit2d_mmfi_eagle",
        model_family="vit2d",
        task="pose",
        train_module="rfpose.training.train_vit2d",
        dataset_version="rfpose-unified-v2",
        description="CSIViT2DPose MM-Fi baseline.",
        gpus=1,
        time_limit="24:00:00",
    ),
    "vit2d_wipose_eagle": TrainingPreset(
        config_name="vit2d_wipose_eagle",
        model_family="vit2d",
        task="pose",
        train_module="rfpose.training.train_vit2d",
        dataset_version="rfpose-unified-v2",
        description="CSIViT2DPose WiPose baseline.",
        gpus=1,
        time_limit="24:00:00",
    ),
    "vit2d_mmfi_augmentation": TrainingPreset(
        config_name="vit2d_mmfi_augmentation",
        model_family="vit2d",
        task="pose",
        train_module="rfpose.training.train_vit2d_augmentation",
        dataset_version="rfpose-unified-v2",
        description="CSIViT2DPose MM-Fi with CSI augmentation.",
        gpus=1,
        time_limit="24:00:00",
    ),
    "vit2d_wipose_augmentation": TrainingPreset(
        config_name="vit2d_wipose_augmentation",
        model_family="vit2d",
        task="pose",
        train_module="rfpose.training.train_vit2d_augmentation",
        dataset_version="rfpose-unified-v2",
        description="CSIViT2DPose WiPose with CSI augmentation.",
        gpus=1,
        time_limit="24:00:00",
    ),
    "vit2d_full_augmentation": TrainingPreset(
        config_name="vit2d_full_augmentation",
        model_family="vit2d",
        task="pose",
        train_module="rfpose.training.train_vit2d_augmentation",
        dataset_version="rfpose-unified-v2",
        description="CSIViT2DPose full-dataset run with CSI augmentation.",
        gpus=1,
        time_limit="24:00:00",
    ),
    "transformer_gold": TrainingPreset(
        config_name="transformer_gold",
        model_family="legacy-transformer",
        task="multitask",
        train_module="rfpose.training.transformer_train",
        dataset_version="rfpose-unified-v2",
        description="Legacy Transformer gold baseline.",
    ),
    "transformer_eagle": TrainingPreset(
        config_name="transformer_eagle",
        model_family="legacy-transformer",
        task="multitask",
        train_module="rfpose.training.transformer_train",
        dataset_version="rfpose-unified-v2",
        description="Legacy Transformer Eagle run.",
    ),
    "ssl_pretrain": TrainingPreset(
        config_name="ssl_pretrain",
        model_family="ssl",
        task="pretrain",
        train_module="rfpose.training.ssl_pretrain",
        dataset_version="rfpose-unified-v2",
        description="Self-supervised CSI encoder pretraining.",
        time_limit="12:00:00",
    ),
    "ssl_eagle": TrainingPreset(
        config_name="ssl_eagle",
        model_family="ssl",
        task="pretrain",
        train_module="rfpose.training.ssl_pretrain",
        dataset_version="rfpose-unified-v2",
        description="Eagle SSL CSI encoder pretraining.",
        time_limit="24:00:00",
    ),
    "quick_test": TrainingPreset(
        config_name="quick_test",
        model_family="legacy-transformer",
        task="smoke",
        train_module="rfpose.training.transformer_train",
        dataset_version="rfpose-unified-v2",
        description="Small smoke test for validating the training stack.",
        mem="32G",
        time_limit="01:00:00",
    ),
    "demo": TrainingPreset(
        config_name="demo",
        model_family="legacy-transformer",
        task="smoke",
        train_module="rfpose.training.transformer_train",
        dataset_version="rfpose-unified-v2",
        description="Demo training config.",
        mem="32G",
        time_limit="01:00:00",
    ),
    "eval_demo": TrainingPreset(
        config_name="eval_demo",
        model_family="evaluation",
        task="eval",
        train_module="rfpose.evaluation.eval_job",
        dataset_version="rfpose-unified-v2",
        description="Evaluation job wrapper.",
        mem="32G",
        time_limit="02:00:00",
    ),
}


def get_preset(config_name: str) -> TrainingPreset:
    return TRAINING_PRESETS.get(
        config_name,
        TrainingPreset(
            config_name=config_name,
            model_family="custom",
            task="custom",
            train_module="rfpose.training.transformer_train",
            dataset_version="rfpose-unified-v2",
            description="Custom config not registered in rfpose_eagle.registry.",
        ),
    )


def list_presets() -> list[dict]:
    return [preset.to_dict() for preset in sorted(TRAINING_PRESETS.values(), key=lambda p: p.config_name)]


def resolve_train_module(config_name: str, explicit_module: str = "") -> str:
    return explicit_module or get_preset(config_name).train_module
