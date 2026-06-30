import torch
import torch.nn.functional as F
import random

def rotate(x, mask= None, dims= ((-3,-2), (-3,-1), (-2,-1)), p= 1.0):
    """
    Rotate pixels.

    Same rotate for each sample in batch is 
    used for speed. This reduces batch 
    diversity.
    """
    bs= x.shape[0]
    for d in dims:
        if random.random() < p:
            k = random.randint(0,3)
            x = torch.rot90(x, k=k, dims=d)
            if mask is not None:
                mask = torch.rot90(mask, k=k, dims=d) 

    if mask is not None:
        return x, mask
    else:
        return x

def flip_3d(x, mask= None, dims=(-3,-2,-1), p= 0.5):
    """
    Flip along axis.
    """
    axes = [i for i in dims if random.random() < p]
    if axes:
        x = torch.flip(x, dims=axes)
        if mask is not None:
            mask = torch.flip(mask, dims=axes)
        
    if mask is not None:
        return x, mask
    else:
        return x

def swap_dims(x, mask= None, p= 0.5, dims=(-2,-1)):
    """
    Randomly swap dims.
    """
    if random.random() < p:
        swap_dims= list(dims)
        random.shuffle(swap_dims)
        x = x.transpose(*swap_dims)
        if mask is not None:
            mask = mask.transpose(*swap_dims)

    if mask is not None:
        return x, mask
    else:
        return x

def cutmix_3d(x, mask= None, p= 1.0, dims=(-2,-1)):
    """
    Cutmix.
    """

    # Shuffle
    x_mixed = x.roll(1, dims=0)
    if mask is not None:
        mask_mixed = mask.roll(1, dims=0)

    # Shapes
    pb, pc, pz, py, px= x.shape

    for idx in range(pb):
        prob= random.random()
        if prob < p:

            # Get bbox size
            # z_size= int(random.uniform(0.0, 1.0) * pz)
            z_size= pz if -3 in dims else int(random.uniform(0.0, 1.0) * pz)
            y_size= py if -2 in dims else int(random.uniform(0.0, 1.0) * py)
            x_size= px if -1 in dims else int(random.uniform(0.0, 1.0) * px)

            # Get bbox positions
            z_start = random.randint(0, pz - z_size)
            y_start = random.randint(0, py - y_size)
            x_start = random.randint(0, px - x_size)
            z_end= z_start + z_size
            y_end= y_start + y_size
            x_end= x_start + x_size

            # Apply to box
            x[idx, :, z_start:z_end, y_start:y_end, x_start:x_end] = \
            x_mixed[idx, :, z_start:z_end, y_start:y_end, x_start:x_end]

            if mask is not None:
                mask[idx, :, z_start:z_end, y_start:y_end, x_start:x_end] = \
                mask_mixed[idx, :, z_start:z_end, y_start:y_end, x_start:x_end]

    if mask is not None:
        return x, mask
    else:
        return x

def coarse_dropout_3d(x, mask= None, p= 0.5, fill_val=0.0, num_holes=(1,3), hole_range=(8, 64, 64)):

    # Apply with proba
    if torch.rand(1).item() < p:
        zs,ys,xs= x.shape[-3:]

        # Random number of holes
        num_holes= torch.randint(
            low=num_holes[0], 
            high=num_holes[1], 
            size= (1,),
            device="cpu",
            ).item()

        # Dropout coords
        z_start = torch.randint(low=0, high=zs-hole_range[0], size=(num_holes,), device="cpu")#.item()
        y_start = torch.randint(low=0, high=ys-hole_range[1], size=(num_holes,), device="cpu")#.item()
        x_start = torch.randint(low=0, high=xs-hole_range[2], size=(num_holes,), device="cpu")#.item()

        z_size = torch.randint(low=2, high=hole_range[0], size=(num_holes,), device="cpu")
        y_size = torch.randint(low=2, high=hole_range[1], size=(num_holes,), device="cpu")
        x_size = torch.randint(low=2, high=hole_range[2], size=(num_holes,), device="cpu")

        # Apply dropout
        for i in range(num_holes):
            x[..., 
            z_start[i]: z_start[i] + z_size[i], 
            y_start[i]: y_start[i] + y_size[i], 
            x_start[i]: x_start[i] + x_size[i],
            ] = fill_val

    if mask is not None:
        return x, mask
    else:
        return x

def temporal_flip(imgs, coords, masks, rng, p=0.5):
    """Consistent random spatial flip across all W frames; updates coordinates."""
    W, C, Z, Y, X = imgs.shape
    spatial_sizes = [Z, Y, X]
    flip_axes = [2, 3, 4]  # tensor dims for Z, Y, X

    for axis_idx, (dim, ax) in enumerate(zip(spatial_sizes, flip_axes)):
        if rng.random() < p:
            imgs = imgs.flip(dims=[ax])
            for t in range(W):
                if len(coords[t]) > 0:
                    coords[t] = coords[t].clone()
                    coords[t][:, axis_idx] = (dim - 1) - coords[t][:, axis_idx]

    return imgs, coords, masks


def temporal_brightness(imgs, coords, masks, rng, shift_range=0.1, p=0.5):
    """Random additive brightness shift on all frames."""
    if rng.random() < p:
        shift = float(rng.uniform(-shift_range, shift_range))
        imgs = (imgs + shift).clamp(0.0, 4.0)
    return imgs, coords, masks


def temporal_coarse_dropout(imgs, coords, masks, rng, p=0.3,
                             num_holes=(1, 3), hole_range=(4, 16, 16)):
    """Apply random 3D boxes of zeros to all W frames (same location each frame)."""
    if rng.random() < p:
        W, C, Z, Y, X = imgs.shape
        n_holes = int(rng.integers(num_holes[0], num_holes[1] + 1))
        for _ in range(n_holes):
            hz = int(rng.integers(2, max(3, hole_range[0])))
            hy = int(rng.integers(2, max(3, hole_range[1])))
            hx = int(rng.integers(2, max(3, hole_range[2])))
            z0 = int(rng.integers(0, max(1, Z - hz)))
            y0 = int(rng.integers(0, max(1, Y - hy)))
            x0 = int(rng.integers(0, max(1, X - hx)))
            imgs[:, :, z0:z0 + hz, y0:y0 + hy, x0:x0 + hx] = 0.0

    return imgs, coords, masks


if __name__ == "__main__":
    x= torch.ones(1,1,32,32,32)
    mask= torch.ones(1,6,32,32,32)
    print(torch.sum(x))
    x,mask= coarse_dropout_3d(x, mask, p=1.0, num_holes=(1, 3), hole_range=(8, 8, 8))
    print(torch.sum(x))
    print(x.shape, mask.shape)