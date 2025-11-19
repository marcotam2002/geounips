"""
Geometry Meets Light: Leveraging Geometric Priors for Universal Photometric Stereo
under Limited Multi-Illumination Cues (AAAI2026)
# Copyright (c) 2025 King-Man Tam
# All rights reserved.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import *
from . import transformer
from ..utils import gauss_filter
from ..utils.ind2sub import *

from .vggt.models.aggregator import Aggregator
from .vggt.heads.dpt_head import DPTHead

import time
 
class GLC_Upsample(nn.Module):
    def __init__(self, input_nc, num_enc_sab=1, dim_hidden=256, dim_feedforward=1024, use_efficient_attention=False):
        super(GLC_Upsample, self).__init__()       
        self.comm = transformer.CommunicationBlock(input_nc, num_enc_sab = num_enc_sab, dim_hidden=dim_hidden, ln=True, dim_feedforward = dim_feedforward,use_efficient_attention=False)
       
    def forward(self, x):
        x = self.comm(x)        
        return x

class GLC_Aggregation(nn.Module):
    def __init__(self, input_nc, num_agg_transformer=2, dim_aggout=384, dim_feedforward=1024, use_efficient_attention=False):
        super(GLC_Aggregation, self).__init__()              
        self.aggregation = transformer.AggregationBlock(dim_input = input_nc, num_enc_sab = num_agg_transformer, num_outputs = 1, dim_hidden=dim_aggout, dim_feedforward = dim_feedforward, num_heads=8, ln=True, attention_dropout=0.1, use_efficient_attention=use_efficient_attention)

    def forward(self, x):
        x = self.aggregation(x)      
        return x

class Regressor(nn.Module):
    def __init__(self, input_nc, num_enc_sab=1, use_efficient_attention=False, dim_feedforward=256, output='normal'):
        super(Regressor, self).__init__()     
        # Communication among different samples (Pixel-Sampling Transformer)
        # not sure here
        self.dim_hidden = 384
        self.comm = transformer.CommunicationBlock(input_nc, num_enc_sab = num_enc_sab, dim_hidden=self.dim_hidden, ln=True, dim_feedforward = dim_feedforward, use_efficient_attention=use_efficient_attention)   
        self.prediction_normal = PredictionHead(self.dim_hidden, 3)
        self.target = output
        if output == 'brdf':   
            self.prediction_base = PredictionHead(self.dim_hidden, 3) # No urcainty
            self.prediction_rough = PredictionHead(self.dim_hidden, 1)
            self.prediction_metal = PredictionHead(self.dim_hidden, 1)

    def forward(self, x, num_sample_set):
        """Standard forward
        INPUT: img [Num_Pix, F]
        OUTPUT: [Num_Pix, 3]"""  


        if x.shape[0] % num_sample_set == 0:
            x_ = x.reshape(-1, num_sample_set, x.shape[1])
            x_ = self.comm(x_)            
            x = x_.reshape(-1, self.dim_hidden)
        else:
            ids = list(range(x.shape[0]))
            num_split = len(ids) // num_sample_set
            x_1 = x[:(num_split)*num_sample_set, :].reshape(-1, num_sample_set, x.shape[1])
            x_1 = self.comm(x_1).reshape(-1, self.dim_hidden)
            x_2 = x[(num_split)*num_sample_set:,:].reshape(1, -1, x.shape[1])
            x_2 = self.comm(x_2).reshape(-1, self.dim_hidden)
            x = torch.cat([x_1, x_2], dim=0)

        x_n = self.prediction_normal(x)        
        if self.target == 'brdf':
            x_brdf = (self.prediction_base(x), self.prediction_rough(x), self.prediction_metal(x))
        else:
            x_brdf = []
        return x_n, x_brdf, x
    
class PredictionHead(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(PredictionHead, self).__init__()
        modules_regression = []
        modules_regression.append(nn.Linear(dim_input, dim_input//2))
        modules_regression.append(nn.ReLU())
        modules_regression.append(nn.Linear(dim_input//2, dim_output))
        self.regression = nn.Sequential(*modules_regression)

    def forward(self, x):
        return self.regression(x)

class Net(nn.Module):
    def __init__(self, pixel_samples, output, device):
        super().__init__()
        self.device = device
        self.target = output
        self.pixel_samples = pixel_samples
        self.glc_smoothing = True

   
        """ Geometric Encoder """
        self.aggregator = Aggregator(aa_order=["frame", "global"]).to("cuda")

        self.light_aggregator = Aggregator(
                                    depth=12,
                                    patch_embed="conv",
                                    aa_order=["frame", "light"]
                                ).to("cuda").to(self.device) 

        patch_size = 14
        features = 128
        embed_dim = 1024

        """ Geometric Feature """
        self.geometric_extractor = DPTHead(
            dim_in=2 * embed_dim,
            patch_size=patch_size,
            features=features,
            feature_only=True,  # Only output features, no activation
            down_ratio=2,  # Reduces spatial dimensions by factor of 2
            pos_embed=False,
        ).to(self.device) 

        """ Photometric Feature """
        self.photometric_extractor = DPTHead(
            intermediate_layer_idx=[2,5,8,11],
            dim_in=2 * embed_dim,
            patch_size=patch_size,
            features=features,
            feature_only=True,  # Only output features, no activation
            down_ratio=2,  # Reduces spatial dimensions by factor of 2
            pos_embed=False,
        ).to(self.device) 

        # for low-res structure (H/4 x W/4)
        self.glc_upsample_base = GLC_Upsample(256, num_enc_sab=1, dim_hidden=256, dim_feedforward=1024, use_efficient_attention=True).to(self.device) 
        self.glc_aggregation_base = GLC_Aggregation(256, num_agg_transformer=2, dim_aggout=384, dim_feedforward=1024, use_efficient_attention=False).to(self.device) 
        self.regressor_base = Regressor(384, num_enc_sab=1, use_efficient_attention=True, dim_feedforward=1024, output=self.target).to(self.device) 

        # for high-res strucure (H x W)
        self.glc_upsample = GLC_Upsample(256+256, num_enc_sab=1, dim_hidden=256, dim_feedforward=1024, use_efficient_attention=True).to(self.device) 
        self.glc_aggregation = GLC_Aggregation(256+256, num_agg_transformer=2, dim_aggout=384, dim_feedforward=1024, use_efficient_attention=False).to(self.device) 
        self.regressor = Regressor(384+3, num_enc_sab=1, use_efficient_attention=True, dim_feedforward=1024, output=self.target).to(self.device) 

        self.img_embedding = nn.Sequential(
            nn.Linear(3,32),
            nn.LeakyReLU(),
            nn.Linear(32, 256)
        )
        

    def no_grad(self):
        mode_change(self.aggregator, False)
        mode_change(self.light_aggregator, False)
        mode_change(self.geometric_extractor, False)
        mode_change(self.photometric_extractor, False)
        mode_change(self.img_embedding, False)
        mode_change(self.glc_upsample_base, False)
        mode_change(self.glc_aggregation_base, False)
        mode_change(self.regressor_base, False)
        mode_change(self.glc_upsample, False)
        mode_change(self.glc_aggregation, False)
        mode_change(self.regressor, False)
    
    @torch.no_grad()
    
    def forward(self, I, M, nImgArray, decoder_resolution, canonical_resolution):     
        
        # --- Parse decoder & canonical resolutions ---
        decoder_resolution = decoder_resolution[0, 0].cpu().numpy().astype(np.int32).item()
        canonical_resolution = canonical_resolution[0, 0].cpu().numpy().astype(np.int32).item()

        # --- Setup ---
        B, C, H, W, Nmax = I.shape
        start_enc = time.time()

        # -------------------------------------------------------
        #  Stage 1: Coarse Normal Prediction (Encoder)
        # -------------------------------------------------------
        agg_resolution = 518

        # Reorder to (B, Nmax, C, H, W)
        img = I.permute(0, 4, 1, 2, 3)

        # Build valid-image index mask
        img_index = make_index_list(Nmax, nImgArray)

        # Apply mask
        mask = M.unsqueeze(1).expand(-1, Nmax, 3, -1, -1)
        img = img * mask

        # Merge batch * views
        img_merged = img.reshape(-1, C, H, W)

        # Resize for decoder and aggregator
        I_dec = safe_interpolate(img_merged, size=(decoder_resolution, decoder_resolution),
                                mode='bilinear', align_corners=False)
        img_agg = safe_interpolate(img_merged, size=(agg_resolution, agg_resolution),
                                mode='bilinear', align_corners=False)

        # Restore (B, Nmax, C, ...)
        img_agg = img_agg.view(B, Nmax, C, agg_resolution, agg_resolution)

        # --- Geometric Features ---
        tokens_list, ps_idx = self.aggregator(img_agg)
        geometric_maps = self.geometric_extractor(tokens_list, img_agg, ps_idx)
        del tokens_list, ps_idx

        # --- Photometric Features ---
        tokens_list, ps_idx = self.light_aggregator(img_agg)
        photometric_maps = self.photometric_extractor(tokens_list, img_agg, ps_idx)
        del tokens_list, ps_idx, img_agg

        # --- Resize GLC Features ---
        geometric_maps = geometric_maps.reshape(-1, 128, 259, 259)
        photometric_maps = photometric_maps.reshape(-1, 128, 259, 259)

        geometric_maps = safe_interpolate(geometric_maps, size=(256, 256),
                                        mode='bilinear', align_corners=False)
        photometric_maps = safe_interpolate(photometric_maps, size=(256, 256),
                                            mode='bilinear', align_corners=False)

        # GLC = Geometry + Light
        glc = torch.cat([geometric_maps, photometric_maps], dim=1)
        del geometric_maps, photometric_maps

        enc_time = time.time() - start_enc
        start_dec = time.time()


        # -------------------------------------------------------
        #  Stage 2: High-Resolution Normal Prediction (Decoder)
        # -------------------------------------------------------

        # Prepare input for decoder
        img_full = I.permute(0, 4, 1, 2, 3).to(self.device)     # (B, Nmax, C, H, W)
        mask_full = M

        # Filter only valid views
        img_full = img_full.reshape(-1, C, H, W)
        img_full = img_full[img_index == 1]

        # Resize for decoder
        img_dec = F.interpolate(img_full, size=(decoder_resolution, decoder_resolution),
                                mode='bilinear', align_corners=False)
        M_dec = F.interpolate(mask_full, size=(decoder_resolution, decoder_resolution),
                            mode='nearest')

        # Decoder image sizes
        C = img_full.shape[1]
        H = decoder_resolution
        W = decoder_resolution

        # Output buffer
        nout = torch.zeros(B, H * W, 3).to(self.device)

        # Optional GLC smoothing
        if self.glc_smoothing:
            f_scale = decoder_resolution // canonical_resolution
            smoothing = gauss_filter.gauss_filter(glc.shape[1],
                                                10 * f_scale + 1,
                                                1).to(glc.device)
            glc = smoothing(glc)

        p = 0
        for b in range(B):

            # indices of valid images for this sample
            target = range(p, p + nImgArray[b])
            p += nImgArray[b]

            # Mask (flatten)
            m_flat = M_dec[b].reshape(-1, H * W).permute(1, 0)
            ids = np.nonzero(m_flat > 0)[:, 0]
            ids = ids[np.random.permutation(len(ids))]

            # Split into pixel batches
            if len(ids) > self.pixel_samples:
                num_split = len(ids) // self.pixel_samples + 1
                idset = np.array_split(ids, num_split)
            else:
                idset = [ids]

            # Gather multi-view observations
            o_full = img_dec[target].reshape(nImgArray[b], C, H * W).permute(2, 0, 1)

            # ---- Pixel Sampling Loop ----
            for ids in idset:

                o_ids = o_full[ids]

                # GLC feature sampling
                coords = ind2coords(np.array((H, W)), ids).expand(Nmax, -1, -1, -1)
                glc_ids = F.grid_sample(glc[target], coords.to(glc.device),
                                        mode='bilinear', align_corners=False)
                glc_ids = glc_ids.reshape(len(target), -1, len(ids)).permute(2, 0, 1)

                # Convert coords again
                coords = ind2coords(np.array((H, W)), ids)

                # -------------------------------------------------------
                #  Base Normal (canonical 256x256)
                # -------------------------------------------------------
                x_base = self.glc_upsample_base(glc_ids)
                x_base = self.glc_aggregation_base(x_base)
                x_n_base, _, _ = self.regressor_base(x_base, len(ids))
                x_n_base = F.normalize(x_n_base, dim=1)

                # -------------------------------------------------------
                #  Fine Normal (original resolution)
                # -------------------------------------------------------
                o_ids_embed = self.img_embedding(o_ids)
                x = torch.cat([o_ids_embed, glc_ids], dim=2)

                glc_up = self.glc_upsample(x)
                x = torch.cat([o_ids_embed, glc_up], dim=2)

                x = self.glc_aggregation(x)
                x_n, _, _ = self.regressor(torch.cat([x, x_n_base], dim=1), len(ids))
                x_n = F.normalize(x_n, dim=1)

                # Store normal
                if self.target == 'normal':
                    nout[b, ids] = x_n.detach()

        # Restore (B, 3, H, W)
        nout = nout.permute(0, 2, 1).reshape(B, 3, H, W)

        dec_time = time.time() - start_dec
        print(f"[Timing] Encoder: {enc_time:.3f}s, Decoder: {dec_time:.3f}s")

        return nout