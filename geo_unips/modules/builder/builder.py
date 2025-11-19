"""
Geometry Meets Light: Leveraging Geometric Priors for Universal Photometric Stereo
under Limited Multi-Illumination Cues (AAAI2026)
# Copyright (c) 2025 King-Man Tam
# All rights reserved.
"""

import torch
import torch.nn.functional as F
import numpy as np
from modules.model import model, decompose_tensors
from modules.model.model_utils import *
from modules.utils import compute_mae
import cv2
import glob
import time


class builder():
    def __init__(self, args, device):
   
        self.device = device
        self.args = args               
        """Load pretrained model"""
        if 'normal' in args.target:
            model_dir = f'{args.checkpoint}/normal'
            self.net_nml = model.Net(args.pixel_samples, 'normal', device).to(self.device)
            self.net_nml = torch.nn.DataParallel(self.net_nml)
            self.net_nml = self.load_models(self.net_nml, model_dir)
            self.net_nml.module.no_grad()

        print('')

        print(f"canonical resolution: {self.args.canonical_resolution} x {self.args.canonical_resolution}  ")
        print(f"pixel samples: {self.args.pixel_samples}\n") 

    def separate_batch(self, batch):
        I = batch[0].to(self.device) # [B, 3, H, W]
        N = batch[1].to(self.device) # [B, 1, H, W]
        M = batch[2].to(self.device) # [B, 1, H, W]
        nImgArray = batch[3]
        roi = batch[4]
        return I, N, M, nImgArray, roi

    def load_models(self, model, dirpath):
        pytmodel = "".join(glob.glob(f'{dirpath}/*.pytmodel'))
        model = loadmodel(model, pytmodel, strict=True)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[Params] Total: {total_params / 1e6:.2f} M")
        return model

    def run(self, 
            canonical_resolution = None,
            testdata = None,
            max_image_resolution = None,
            ):
        
        testdata.max_image_resolution = max_image_resolution
        test_data_loader = torch.utils.data.DataLoader(testdata, batch_size = 1, shuffle=False, num_workers=0, pin_memory=True)

        mae_list = []
        time_list = []
        memory_list = []


        for batch_test in test_data_loader:                  
            start_time = time.time()

            I, N, M, nImgArray, roi = self.separate_batch(batch_test)


            roi = roi[0].numpy()
            h_ = roi[0]
            w_ = roi[1]
            r_s = roi[2]
            r_e = roi[3]
            c_s = roi[4]
            c_e = roi[5]
            B, C, H, W, Nimg = I.shape

            """Scalable Reconstruction"""   
            with torch.no_grad():       
                if self.args.scalable:    
                    patch_size = 512               
                    patches_I = decompose_tensors.divide_tensor_spatial(I.permute(0,4,1,2,3).reshape(-1, C, H, W), block_size=patch_size, method='tile_stride')
                    patches_I = patches_I.reshape(B, Nimg, -1, C, patch_size, patch_size).permute(0, 2, 3, 4, 5, 1)
                    sliding_blocks = patches_I.shape[1]
                    patches_M = decompose_tensors.divide_tensor_spatial(M, block_size=patch_size, method='tile_stride')
                    
                    patches_nml = []

                    for k in range(sliding_blocks):
                        print(f"Recovering {self.args.target} map(s): {k+1} / {sliding_blocks}")
                        if torch.sum(patches_M[:, k, :, :, :]) > 0:
                            pI = patches_I[:, k, :, :, :,:]
                            pI = F.interpolate(pI.permute(0,4,1,2,3).reshape(-1, pI.shape[1],pI.shape[2],pI.shape[3]), size=(patch_size, patch_size), mode='bilinear', align_corners=True).reshape(B, Nimg, C, patch_size, patch_size).permute(0,2,3,4,1)
                            pM = F.interpolate(patches_M[:, k, :, :, :], size=(patch_size, patch_size), mode='bilinear', align_corners=True)
                            nout = torch.zeros((B, 3, patch_size, patch_size))
                            if 'normal' in self.args.target:
                                nout = self.net_nml(pI, pM, nImgArray.reshape(-1,1), decoder_resolution = patch_size * torch.ones(pI.shape[0],1), canonical_resolution=canonical_resolution* torch.ones(pI.shape[0],1))
                                nout = (F.interpolate(nout, size=(patch_size, patch_size), mode='bilinear', align_corners=True) * pM).cpu()
    
                            patches_nml.append(nout)
                        else:
                            patches_nml.append(torch.zeros((B, 3, patch_size, patch_size)))   
           
                    patches_nml = torch.stack(patches_nml, dim=1)
                    merged_tensor_nml = decompose_tensors.merge_tensor_spatial(patches_nml.permute(1,0,2,3,4), method='tile_stride')
                    nml = merged_tensor_nml.squeeze().permute(1,2,0)

                else:
                    print(f"Recovering {self.args.target} map(s) 1 / 1")
                    if 'normal' in self.args.target:
                        nout = self.net_nml(I, M, nImgArray.reshape(-1,1), decoder_resolution=testdata.data.h * torch.ones(I.shape[0],1), canonical_resolution=canonical_resolution* torch.ones(I.shape[0],1))
                        nml = (nout * M.to(nout.device)).squeeze().permute(1, 2, 0).cpu().detach()
                        del nout
                    
                
                # save normal of original resolution
                if 'normal' in self.args.target:
                    nml = nml.cpu().numpy()                
                    nml = cv2.resize(nml, dsize=(c_e-c_s, r_e-r_s), interpolation=cv2.INTER_CUBIC)
                    mask = np.float32(np.abs(1 - np.sqrt(np.sum(nml * nml, axis=2))) < 0.5)
                    nml = np.divide(nml, np.linalg.norm(nml, axis=2, keepdims=True) + 1.0e-12)
                    nml = nml * mask[:, :, np.newaxis]
                    nout = np.zeros((h_, w_, 3), np.float32)
                    nout[r_s:r_e, c_s:c_e,:] = nml

                    if torch.sum(N) > 0:
                        n_true = N.permute(0,2,3,1).squeeze().cpu().numpy()
                        mask = np.float32(np.abs(1 - np.sqrt(np.sum(n_true * n_true, axis=2))) < 0.5)
                        mae, emap = compute_mae.compute_mae_np(nout, n_true, mask = mask)
                        mae_list.append(mae)
                        print(f"Mean Angular Error (MAE) is {mae:.3f}\n")                        
                        emap = emap.squeeze()
                        thresh = 90
                        emap[emap>=thresh] = thresh
                        emap = emap/thresh
                        cv2.imwrite(f'{testdata.data.data_workspace}/error.png', 255*emap)     
                    
                    cv2.imwrite(f'{testdata.data.data_workspace}/normal.png', 255*(0.5 * (1+nout[:,:,::-1])))     

                    cv2.imwrite(f'{testdata.data.data_workspace}/mask.png', 255*mask)                                  


            end_time = time.time() 
            elapsed_time = end_time - start_time
            time_list.append(elapsed_time)

            torch.cuda.synchronize()
            mem_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            memory_list.append(mem_used)
            print(f"[Memory] Peak GPU memory usage: {mem_used:.2f} MB")

            print(f"[test] time for this object: {elapsed_time:.2f} sec")


        print(f"[test] avg_mae: {np.mean(mae_list):.3f}")
        print(f"[test] avg_time: {np.mean(time_list):.2f} sec over {len(time_list)} objects")
        if memory_list:
            avg_mem = np.mean(memory_list)
            max_mem = np.max(memory_list)
            print(f"[test] avg_memory: {avg_mem:.2f} MB over {len(memory_list)} objects")
            print(f"[test] max_memory: {max_mem:.2f} MB")
