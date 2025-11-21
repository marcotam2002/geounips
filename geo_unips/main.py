"""
Geometry Meets Light: Leveraging Geometric Priors for Universal Photometric Stereo
under Limited Multi-Illumination Cues (AAAI2026)
# Copyright (c) 2025 King-Man Tam
# All rights reserved.
"""

from __future__ import print_function, division
from modules.model.model_utils import *
from modules.builder import builder
from modules.io import dataio
import sys
import argparse
import time
import torch

sys.path.append('..')  # add parent directly for importing

# Argument parser
parser = argparse.ArgumentParser()

# Properties
parser.add_argument('--session_name', default='geo_unips')
parser.add_argument('--checkpoint', default='checkpoint')

# Data Configuration
parser.add_argument('--max_image_res', type=int, default=512)
parser.add_argument('--max_image_num', type=int, default=4)
parser.add_argument('--test_ext', default='.data')
parser.add_argument('--test_dir', default='assets')
parser.add_argument('--test_prefix', default='L*')
parser.add_argument('--mask_margin', type=int, default=8)

# Network Configuration
parser.add_argument('--canonical_resolution', type=int, default=256)
parser.add_argument('--pixel_samples', type=int, default=10000)
parser.add_argument('--scalable', action='store_true')


def main():
    args = parser.parse_args()
    print(f'\nStarting a session: {args.session_name}')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    geo_unips = builder.builder(args, device)
    test_data = dataio.dataio(args)

    start_time = time.time()
    geo_unips.run(testdata=test_data,
                  max_image_resolution=args.max_image_res,
                  canonical_resolution=args.canonical_resolution,
                  )
    end_time = time.time()
    print(f"Prediction finished (Elapsed time is {end_time - start_time:.3f} sec)")
    print("\nNormal estimation completed. The predicted normal maps are saved at:\n")
    print(f"        ./{args.session_name}/results/{test_data.data.objname}\n")



if __name__ == '__main__':
    main()
