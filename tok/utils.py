import torch
import torch.nn as nn


class ScalingLayer(nn.Module):
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        super().__init__()
        self.register_buffer('shift', torch.Tensor(mean)[None, :, None, None])
        self.register_buffer('scale', torch.Tensor(std)[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale
    
    def inv(self, inp):
        return inp * self.scale + self.shift


def load_torch_checkpoint(path_or_obj, map_location="cpu"):
    if not isinstance(path_or_obj, (str, bytes, bytearray)):
        return path_or_obj
    try:
        return torch.load(path_or_obj, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path_or_obj, map_location=map_location)
