
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# Boundary Loss taken from https://github.com/LIVIAETS/boundary-loss/blob/master/losses.py
class SurfaceLoss(nn.Module):
    def __init__(self):
        super(SurfaceLoss, self).__init__()

    def forward(self, logits: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pc = probs.type(torch.float32)
        dc = dist_maps.type(torch.float32)
        loss = torch.einsum("bkwh,bkwh->bkwh", pc, dc).mean()
        return loss

# Combo Loss: Dice Loss and Boundary Loss    
class DL_BL(nn.Module):
    def __init__(self):
        super(DL_BL, self).__init__()
        self.dice_loss = smp.losses.DiceLoss(mode='binary')
        self.surface_loss = SurfaceLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, dist_maps: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * self.dice_loss(logits, labels) + alpha * self.surface_loss(logits, dist_maps)
