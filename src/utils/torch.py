import torch
import torch.nn as nn
import torch.nn.functional as F

def nms_2d(scores: torch.Tensor, nms_radius: int):
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    return torch.where(max_mask, scores, zeros)

def nms_3d(scores: torch.Tensor, nms_radius: int):
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool3d(x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    return torch.where(max_mask, scores, zeros)

def center_of_mass_3d(arr, kernel_size=3):
    """
    Alternative to NMS. Centernet like aggregation
    when segmentation is not precise.
    """
    assert kernel_size%2 == 1, "kernel_size must be odd"
    t,h,w= arr.shape

    # ===== Calculate peak =====
    idx= torch.argmax(arr)
    idx= torch.unravel_index(idx, arr.shape)

    # Pad (to account for peaks on edges)
    half = kernel_size // 2
    pad = (half, half, half, half, half, half)
    arr= F.pad(arr.unsqueeze(0), pad, mode="reflect").squeeze(0)
    idx_padded = (idx[0] + half, idx[1] + half, idx[2] + half) # adjust argmax to after pad

    # Crop local area
    tmp= torch.zeros((kernel_size, kernel_size, kernel_size), device=arr.device)
    tmp= arr[
        idx_padded[0]-half:idx_padded[0]+half+1, 
        idx_padded[1]-half:idx_padded[1]+half+1, 
        idx_padded[2]-half:idx_padded[2]+half+1,
        ]

    # ===== Calculate center of mass =====
    z_coords, y_coords, x_coords = torch.meshgrid(
        torch.arange(tmp.size(0), device=tmp.device), 
        torch.arange(tmp.size(1), device=tmp.device), 
        torch.arange(tmp.size(2), device=tmp.device), 
        indexing='ij',
    )
    
    # Compute weighted averages for each axis
    tmp_total = torch.sum(tmp)
    weighted_z = torch.sum(z_coords * tmp) / tmp_total
    weighted_y = torch.sum(y_coords * tmp) / tmp_total
    weighted_x = torch.sum(x_coords * tmp) / tmp_total
    
    # Calculate offset from center pixel
    # Note: Odd kernel size accounts for +0.5 to center pixel
    z_offset = weighted_z - kernel_size/2 + 1
    y_offset = weighted_y - kernel_size/2 + 1
    x_offset = weighted_x - kernel_size/2 + 1

    # Add offset to argmax
    idx= (idx[0]+z_offset.item(), idx[1]+y_offset.item(), idx[2]+x_offset.item())
    return idx
