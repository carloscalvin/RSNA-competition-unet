import json

import torch
import torch.nn as nn
import torch.nn.functional as F 


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_weights(model, wpath):
    state_dict = torch.load(wpath, map_location="cpu", weights_only=True)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print("MISSING_KEYS:")
        print(json.dumps(missing_keys, indent=4))
        
    if unexpected_keys:
        print("UNEXPECTED_KEYS:")
        print(json.dumps(unexpected_keys, indent=4))
    
    return model