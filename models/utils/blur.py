import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from models.backbones.stripformer.Stripformer import Stripformer


def get_blur_vec(model, frames, num):
    frames = rearrange(frames, 'b c d h w -> b d c h w')
    _, d, c, h, w = frames.shape
    with torch.no_grad():
        img_tensor = frames[:, 0:d:int(d / num), :, :, :]
        img_tensor = img_tensor.reshape(-1, c, h, w)

        factor = 8
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        img_tensor = F.pad(img_tensor, (0, padw, 0, padh), 'reflect')
        H, W = img_tensor.shape[2], img_tensor.shape[3]

        output = model(img_tensor)
    return output

def get_blur_net():
    model_g = Stripformer()
    # return nn.DataParallel(model_g)
    return model_g