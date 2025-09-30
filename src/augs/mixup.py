import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Beta

class Mixup(nn.Module):
    def __init__(self, beta, mixadd=False):
        super().__init__()
        self.beta_distribution = Beta(beta, beta)
        self.mixadd = mixadd

    def forward(self, X, Y, Z=None):
        b = X.shape[0]
        coeffs = self.beta_distribution.rsample(torch.Size((b,))).to(X.device)

        X_coeffs = coeffs.view((-1,) + (1,) * (X.ndim - 1))
        Y_coeffs = coeffs.view((-1,) + (1,) * (Y.ndim - 1))
        
        perm = torch.randperm(X.size(0))
        X_perm = X[perm]
        Y_perm = Y[perm]
        
        X = X_coeffs * X + (1 - X_coeffs) * X_perm

        if self.mixadd:
            Y = (Y + Y_perm).clip(0, 1)
        else:
            Y = Y_coeffs * Y + (1 - Y_coeffs) * Y_perm

        if Z is not None:
            return X, Y, Z

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

    m= Mixup(beta=5.0)
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
        axes[1, i].set_title(f"Mixup {i}")
    for ax in axes.flatten():
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("mixup.jpg")
    plt.show()
    
