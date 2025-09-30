import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import _Loss

class SmoothBCE(_Loss):
    def __init__(self, smooth=0.0, pos_weight=None):
        super().__init__()
        assert 0 <= smooth < 1, "smooth must be between 0 and 1."
        self.smooth = smooth

        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight]).float()
            self.register_buffer("pos_weight", pos_weight, persistent=False)

        self.pos_weight = pos_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.smooth > 0:
            target = target * (1 - self.smooth) + (1 - target) * self.smooth

        return F.binary_cross_entropy_with_logits(
            input, target, reduction="mean", pos_weight=self.pos_weight,
        )


if __name__ == "__main__":
    l= SmoothBCE(smooth=0.01)

    x= torch.rand(8,16,16)
    y= torch.rand(8,16,16)

    print(l(x,y))


