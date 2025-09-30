import os
import pickle
import numpy as np
import torch

def _compute_gaussian_kernel_3d(kernel_size, sigma):
    center = kernel_size // 2
    zz, yy, xx = np.meshgrid(
        np.arange(kernel_size) - center,
        np.arange(kernel_size) - center,
        np.arange(kernel_size) - center,
        indexing="ij",
    )
    kernel = np.exp(-(zz**2 + yy**2 + xx**2) / (2 * sigma**2))
    return kernel

def create_heatmap_from_coords(coords_list, target_shape, kernel_size, kernel_sigma):
    num_channels, D, H, W = target_shape
    heatmap = np.zeros(target_shape, dtype=np.float32)
    
    if not coords_list:
        return heatmap

    gaussian_kernel = _compute_gaussian_kernel_3d(kernel_size, kernel_sigma)
    half_ks = kernel_size // 2

    for (c, z_center, y_center, x_center) in coords_list:
        z_min, z_max = max(0, z_center - half_ks), min(D, z_center + half_ks + 1)
        y_min, y_max = max(0, y_center - half_ks), min(H, y_center + half_ks + 1)
        x_min, x_max = max(0, x_center - half_ks), min(W, x_center + half_ks + 1)
        
        kz_min = max(0, half_ks - z_center)
        ky_min = max(0, half_ks - y_center)
        kx_min = max(0, half_ks - x_center)

        kz_max = kernel_size - max(0, (z_center + half_ks + 1) - D)
        ky_max = kernel_size - max(0, (y_center + half_ks + 1) - H)
        kx_max = kernel_size - max(0, (x_center + half_ks + 1) - W)
        
        heatmap_roi = heatmap[c, z_min:z_max, y_min:y_max, x_min:x_max]
        kernel_roi = gaussian_kernel[kz_min:kz_max, ky_min:ky_max, kx_min:kx_max]
        
        heatmap[c, z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(heatmap_roi, kernel_roi)
        
    return heatmap

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, df, mode="train"):
        self.cfg = cfg
        self.mode = mode
        self.uids = df['SeriesInstanceUID'].values
        self.image_dir = cfg.image_dir
        self.coords_dir = cfg.coords_dir

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        series_uid = self.uids[idx]
        image_path = os.path.join(self.image_dir, f"{series_uid}.npy")
        img = np.load(image_path)
        coords_path = os.path.join(self.coords_dir, f"{series_uid}_coords.pkl")
        coords_list = []
        if os.path.exists(coords_path):
            with open(coords_path, 'rb') as f:
                coords_list = pickle.load(f)

        label = create_heatmap_from_coords(
            coords_list,
            target_shape=(self.cfg.seg_classes, *img.shape),
            kernel_size=self.cfg.kernel_size,
            kernel_sigma=self.cfg.kernel_sigma
        )
        
        if self.mode == "train":
            D, H, W = img.shape
            roi_d, roi_h, roi_w = self.cfg.roi_size

            z_start = np.random.randint(0, D - roi_d + 1)
            y_start = np.random.randint(0, H - roi_h + 1)
            x_start = np.random.randint(0, W - roi_w + 1)

            img = img[z_start : z_start + roi_d,
                      y_start : y_start + roi_h,
                      x_start : x_start + roi_w]
            
            label = label[:,
                          z_start : z_start + roi_d,
                          y_start : y_start + roi_h,
                          x_start : x_start + roi_w]

        img = img[np.newaxis, ...].astype(np.float32)
        
        return {
            "input": torch.from_numpy(img),
            "target": torch.from_numpy(label),
        }