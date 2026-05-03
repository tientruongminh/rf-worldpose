from __future__ import annotations
import hydra
from omegaconf import DictConfig
from rfpose.training.train import main as train_main
import sys

@hydra.main(version_base=None, config_path='../configs', config_name='rf_worldpose_lora')
def app(cfg: DictConfig):
    dataset = cfg.dataset.path or 'data/gold/stub'
    sys.argv = ['train', '--dataset', dataset, '--epochs', str(cfg.training.epochs), '--batch-size', str(cfg.training.batch_size), '--lr', str(cfg.training.lr)]
    train_main()

if __name__ == '__main__': app()
