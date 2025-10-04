import os
import pickle
import numpy as np
import torch
import scipy

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

            z,y,x= self.cfg.roi_size

            # Random rescale
            if np.random.random() < self.cfg.rescale_p:
                scales = np.random.uniform(low=0.5, high=1.3, size=3)

                img= scipy.ndimage.zoom(img, scales, order=0)
                label= scipy.ndimage.zoom(label, scales, order=0)

            # Crop (to get back to roi_size)
            lz,ly,lx= label.shape
            z_start= np.random.randint(0, max(lz - z, 1))
            y_start= np.random.randint(0, max(ly - y, 1))
            x_start= np.random.randint(0, max(lx - x, 1))
            z_end= z_start + z
            y_end= y_start + y
            x_end= x_start + x
            img= img[z_start:z_end, y_start:y_end, x_start:x_end]
            label= label[z_start:z_end, y_start:y_end, x_start:x_end]

            # Pad (to get back to roi_size)
            lz,ly,lx= label.shape
            pad_z = max(z - lz, 0)
            pad_y = max(y - ly, 0)
            pad_x = max(x - lx, 0)
            pad_zl= np.random.randint(0, pad_z + 1)
            pad_zr= pad_z - pad_zl
            pad_yl= np.random.randint(0, pad_y + 1)
            pad_yr= pad_y - pad_yl
            pad_xl= np.random.randint(0, pad_x + 1)
            pad_xr= pad_x - pad_xl

            pad_width= [(pad_zl, pad_zr), (pad_yl, pad_yr), (pad_xl, pad_xr)]
            label= np.pad(label, pad_width, constant_values=0)
            img= np.pad(img, pad_width, constant_values=np.random.randint(0, 255))

            # Color inversion
            if np.random.random() < 0.25:
                img= 255 - img

        img = img[np.newaxis, ...].astype(np.float32)

        if self.mode == 'train':
            return {
                "input": torch.from_numpy(img),
                "target": torch.from_numpy(label),
            }
        else:
            return {
                "series_uid": series_uid,
                "input": torch.from_numpy(img),
                "target": torch.from_numpy(label),
            }