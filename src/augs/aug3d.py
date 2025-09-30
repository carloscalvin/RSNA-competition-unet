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

if __name__ == "__main__":
    x= torch.ones(1,1,32,32,32)
    mask= torch.ones(1,6,32,32,32)
    print(torch.sum(x))
    x,mask= coarse_dropout_3d(x, mask, p=1.0, num_holes=(1, 3), hole_range=(8, 8, 8))
    print(torch.sum(x))
    print(x.shape, mask.shape)