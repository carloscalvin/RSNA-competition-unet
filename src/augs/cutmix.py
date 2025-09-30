import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Beta


class CutmixSimple(nn.Module):
    def __init__(self, beta=5.0, dims=(-2,-1)):
        super().__init__()
        assert all(_ < 0 for _ in dims), "dims must be negatively indexed."
        self.beta_distribution = Beta(beta, beta) # beta = 5 = gaussianlike
        self.dims = dims

    def forward(self, X, Y, Z=None):
        b = X.shape[0]
        cut_idx = self.beta_distribution.sample().item()

        perm = torch.randperm(X.size(0))
        X_perm = X[perm]
        Y_perm = Y[perm]

        axis= random.choice(self.dims)

        # Get cut idxs
        cutoff_X = int(cut_idx * X.shape[axis])
        cutoff_Y = int(cut_idx * Y.shape[axis])

        # Apply cut
        if axis == -1:
            X[..., :cutoff_X] = X_perm[..., :cutoff_X]
            Y[..., :cutoff_Y] = Y_perm[..., :cutoff_Y]
        elif axis == -2:
            X[..., :cutoff_X, :] = X_perm[..., :cutoff_X, :]
            Y[..., :cutoff_Y, :] = Y_perm[..., :cutoff_Y, :]
        else:
            raise ValueError("CutmixSimple: Axis not implemented.")

        return X, Y

if __name__ == "__main__":
    import torch
    import numpy as np 
    import matplotlib.pyplot as plt 

    # Create empty RGB images: shape (batch, channels, height, width)
    x = torch.zeros(8, 3, 64, 64).float()
    y = torch.ones(8, 3, 16, 16).float()  # Labels can stay the same for now

    # Assign distinct RGB colors
    x[0] = torch.tensor([255.0,   0.0,   0.0]).view(3, 1, 1)  # Red
    x[1] = torch.tensor([  0.0, 255.0,   0.0]).view(3, 1, 1)  # Green
    x[2] = torch.tensor([  0.0,   0.0, 255.0]).view(3, 1, 1)  # Blue
    x[3] = torch.tensor([255.0, 255.0,   0.0]).view(3, 1, 1)  # Yellow
    x[4] = torch.tensor([  0.0, 255.0, 255.0]).view(3, 1, 1)  # Cyan
    x[5] = torch.tensor([255.0,   0.0, 255.0]).view(3, 1, 1)  # Magenta
    x[6] = torch.tensor([128.0, 128.0, 128.0]).view(3, 1, 1)  # Gray
    x[7] = torch.tensor([  0.0,   0.0,   0.0]).view(3, 1, 1)  # Black

    m= CutmixSimple(beta=3.0, dims=(-2,-1))
    x_before= x.clone()
    x_aug, y_aug= m(x, y)
    x_aug= x_aug.cpu().numpy().astype(np.uint8)
    y_aug= y_aug.cpu().numpy().astype(np.uint8)
    x= x_before.cpu().numpy().astype(np.uint8)
    y= y.cpu().numpy().astype(np.uint8)

    # Visualize original and augmented
    fig, axes = plt.subplots(2, 8, figsize=(10, 5))
    for i in range(x.shape[0]):
        print(x[i].transpose(1,2,0).shape)
        axes[0, i].imshow(x[i].transpose(1,2,0))
        axes[0, i].set_title(f"Original {i}")
        axes[1, i].imshow(x_aug[i].transpose(1,2,0))
        axes[1, i].set_title(f"Cutmix {i}")
    for ax in axes.flatten():
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("cutmix.jpg")
    plt.show()