CREATE TABLE IF NOT EXISTS training_configs (
    id            TEXT PRIMARY KEY,
    label         TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    script_path   TEXT NOT NULL DEFAULT 'training/train.py',
    git_repo      TEXT NOT NULL DEFAULT 'https://github.com/tientruongminh/rf-worldpose',
    git_branch    TEXT NOT NULL DEFAULT 'main',
    dataset_hint  TEXT NOT NULL DEFAULT '',
    hyperparams   TEXT NOT NULL DEFAULT '',
    requirements  TEXT NOT NULL DEFAULT 'torch mlflow numpy h5py',
    created_by    TEXT NOT NULL DEFAULT 'system',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO training_configs (id, label, description, script_path, dataset_hint, hyperparams, created_by)
VALUES
  ('wipose-action-baseline',
   'Wi-Pose Action Recognition (baseline)',
   'Baseline action classification on 12 activities using 9-node CSI',
   'training/train_wipose_action.py',
   'wipose-v1',
   'epochs=50
batch_size=128
lr=3e-4
num_nodes=9
window_frames=30
n_subcarriers=30
channels=5
num_classes=12
num_keypoints=18',
   'system'),

  ('wipose-pose2d',
   'Wi-Pose 2D Pose Estimation',
   'Joint regression from CSI amplitude using SkeletonPoints labels',
   'training/train_wipose_pose.py',
   'wipose-v1',
   'epochs=80
batch_size=64
lr=1e-4
num_nodes=9
window_frames=30
n_subcarriers=30
channels=5
num_keypoints=18
pose_weight=1.0',
   'system'),

  ('wiar-action-baseline',
   'WiAR 16-Activity Recognition (baseline)',
   'Baseline activity recognition on 16 activities using Intel 5300 CSI',
   'training/train_wiar_action.py',
   'wiar-v1',
   'epochs=50
batch_size=128
lr=3e-4
num_nodes=9
window_frames=30
n_subcarriers=30
channels=1
num_classes=16',
   'system'),

  ('wiar-action-lora',
   'WiAR LoRA Fine-tune',
   'Fine-tune pretrained model on WiAR with LoRA adapters',
   'training/train_wiar_lora.py',
   'wiar-v1',
   'epochs=30
batch_size=64
lr=1e-4
num_nodes=9
window_frames=30
n_subcarriers=30
channels=1
num_classes=16
lora_rank=8
lora_alpha=16',
   'system')

ON CONFLICT (id) DO NOTHING;
