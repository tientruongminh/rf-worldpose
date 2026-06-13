"""
augmentations.py
"""

import torch
import torch.nn as nn
import random

class CSISpecAugment(nn.Module):
    def __init__(self, time_mask_param=12, freq_mask_param=20, p=0.5):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x

        B, T, F, C = x.shape
        x_aug = x.clone()

        for i in range(B):
            # Time Masking
            if self.time_mask_param > 0 and random.random() < self.p:
                t = random.randint(1, self.time_mask_param)
                t0 = random.randint(0, T - t)
                x_aug[i, t0:t0+t, :, :] = 0.0

            # Frequency Masking
            if self.freq_mask_param > 0 and random.random() < self.p:
                f = random.randint(1, self.freq_mask_param)
                f0 = random.randint(0, F - f)
                x_aug[i, :, f0:f0+f, :] = 0.0

        return x_aug