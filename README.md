<p align="center">
 <h2 align="center"> Geometry Meets Light: Leveraging Geometric Priors for Universal Photometric Stereo Under Limited Multi-Illumination Cues </h2>
<p align="center">
    King-Man Tam<sup>1</sup> · 
    Satoshi Ikehata<sup>2,3</sup> · 
    Yuta Asano<sup>2</sup> · 
    Zhaoyi An<sup>1</sup> · 
    Rei Kawakami<sup>1</sup>
</p>

<p align="center">
    <b>
        <sup>1</sup>Institute of Science Tokyo &nbsp;&nbsp;
        <sup>2</sup>National Institute of Informatics &nbsp;&nbsp;
        <sup>3</sup>Denso IT Laboratory
    </b>
</p>
<p align="center">
    <b>AAAI 2026 Oral</b>
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2511.13015"><img src="https://img.shields.io/badge/arXiv-2511.13015-b31b1b.svg" alt="arXiv Paper"></a>
    <a href="https://marcotam2002.github.io/geounips/"><img src="https://img.shields.io/badge/Project-Page-brightgreen.svg" alt="Project Page"></a>
    <a href="https://huggingface.co/marcotam2002/geounips"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange.svg" alt="Hugging Face Model"></a>
    <a href="https://drive.google.com/drive/folders/1w9-pFxQbmcGbCkKHSWsPFC3E6HafjBbh?usp=drive_link"><img src="https://img.shields.io/badge/Google%20Drive-Dataset-4285F4.svg?logo=googledrive&logoColor=white" alt="Dataset"></a>
</p>

![Cover](/figures/teaser.png)

## Overview

GeoUniPS is a universal photometric stereo network that combines synthetic supervision with high-level geometric priors from large-scale 3D reconstruction models pretrained on in-the-wild data. These reconstruction models act as visual-geometry foundation models and already encode rich geometric knowledge of real scenes, which helps when only a few illumination conditions are available.

![Pipeline](/figures/pipeline.png)

GeoUniPS achieves state-of-the-art performance on multiple datasets, both quantitatively and qualitatively, and is particularly strong on complex in-the-wild scenes.

![Result](/figures/multi_ill.png)

### Updates

- **2026-04-04**: Added a detailed description of PS-Perp.
- **2026-03-20**: Released the PS-Perp training dataset.
- **2025-11-20**: Released the core code, a test subset, and pretrained models.
- **2025-11-18**: Paper available on arXiv.

## Installation

```bash
git clone https://github.com/marcotam2002/geounips.git
cd geounips

conda create -n geounips python=3.10.19
conda activate geounips

pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install einops opencv-python
```

Download the checkpoint from [Hugging Face](https://huggingface.co/marcotam2002/geounips) or [Google Drive](https://drive.google.com/file/d/1GWcdvsLMpjfaqWEvAskf7Tn1Nvv-KeKE/view?usp=sharing) and place it at `checkpoint/ckpt.pytmodel`.

## Inference

```bash
python geo_unips/main.py --session_name "test_data" --test_dir "assets" --checkpoint checkpoint --max_image_num 4 --max_image_res 512 --scalable
```

`--test_dir` is the input image folder and `--session_name` is the name of the output folder.

## PS-Perp Dataset

The PS-Perp training dataset (about 1.1 TB) is available on [Google Drive](https://drive.google.com/drive/folders/1w9-pFxQbmcGbCkKHSWsPFC3E6HafjBbh?usp=drive_link). It is split into 12 archives, `part_00.tar.gz` to `part_11.tar.gz`. Scenes are distributed evenly across the parts, so a few parts are enough for quick experiments and more can be added for full training.

After extraction, each `.data` folder holds one scene:

```text
DATA_PATH/
├── A.data/
├── B.data/
├── C.data/
└── ...
```

All scene folders share the same layout:

```text
<scene>.data/
├── basecolor.png          # albedo
├── metallic.png           # metallic map
├── roughness.png          # roughness map
├── normal.tif             # ground-truth normal
├── depth.exr              # depth map
├── position.exr           # 3D position map (see note below)
├── camera.txt             # camera parameters
├── mask.png               # optional foreground mask
├── image_00000_Point.tif
├── image_00001_Point.tif
├── image_00002_DirEnv.tif
├── ...
└── image_00009_PointEnv.tif
```

The suffix of each `image_XXXXX_*.tif` gives its lighting condition: `Point` (point light), `Dir` (directional light), `Env` (environment light), or a combination such as `PointEnv` and `DirEnv`.

**Known issue:** the provided `position.exr` is problematic and was not used in the paper. If you need a point map, recompute it from `depth.exr` and the camera parameters in `camera.txt`.

## Citation

```bibtex
@inproceedings{kmtam2026geounips,
  title={Geometry Meets Light: Leveraging Geometric Priors for Universal Photometric Stereo Under Limited Multi-Illumination Cues},
  author={Tam, King-Man and Ikehata, Satoshi and Asano, Yuta and An, Zhaoyi and Kawakami, Rei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
