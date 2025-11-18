"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023)
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""

import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Union
import torch.nn.functional as F

def add_module_prefix(state_dict):
    return {("module." + k if not k.startswith("module.") else k): v for k, v in state_dict.items()}

def loadmodel(model, filename, strict=True):
    if os.path.exists(filename):
        params = torch.load('%s' % filename)
        params = add_module_prefix(params)
        model.load_state_dict(params,strict=strict)
        print('Loading pretrained model... %s ' % filename)
    else:
        print('Pretrained model not Found')
    return model

def mode_change(net, Training):
    if Training == True:
        for param in net.parameters():
            param.requires_grad = True
        net.train()
    if Training == False:
        for param in net.parameters():
            param.requires_grad = False
        net.eval()

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def masking(img, mask):
    # img [B, C, H, W]
    # mask [B, 1, H, W] [0,1]
    img_masked = img * mask.expand((-1, img.shape[1], -1, -1))
    return img_masked

def print_model_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('# parameters: %d' % params)

def make_index_list(maxNumImages, numImageList):
    index = np.zeros((len(numImageList) * maxNumImages), np.int32)
    for k in range(len(numImageList)):
        index[maxNumImages*k:maxNumImages*k+numImageList[k]] = 1
    return index
    
def safe_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        if mode in ["linear", "bilinear", "bicubic", "trilinear"]:
            interpolated_chunks = [
                F.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
            ]
        else:
            interpolated_chunks = [
                F.interpolate(chunk, size=size, mode=mode) for chunk in chunks
            ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        if mode in ["linear", "bilinear", "bicubic", "trilinear"]:
            return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)
        else:
            return F.interpolate(x, size=size, mode=mode)