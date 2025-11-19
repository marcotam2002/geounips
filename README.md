<p align="center">
 <h2 align="center"> Geometry Meets Light: Leveraging Geometric Priors for Universal Photometric Stereo under Limited Multi-Illumination Cues </h2>
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

 </p>

<p align="center">
    <a href="https://arxiv.org/abs/2511.13015">
        <img src="https://img.shields.io/badge/arXiv-2511.13015-b31b1b.svg" alt="arXiv Paper">
    </a>
</p>

---

## 📌 **Highlights**  
✅ **Robust Fine-Detail Normal Recovery**: GeoUniPS reconstructs sharp and accurate surface normals even with limited multi-illumination cues, including biased lighting, shadows, and self-occlusions in challenging in-the-wild scenes.
✅ **Address the limitations of the orthographic projection assumption**: Introduces PS-Perp, a dataset with realistic perspective projection that enables learning of spatially varying view directions beyond the orthographic assumption.
✅ **State-of-the-Art Performance**: Achieves superior quantitative and qualitative results across multiple datasets, particularly in complex in-the-wild scenarios.
✅ **Open Source**: Code is publicly available for research and development.

---

## ⏳ **Timeline**  

- ✅ **2025-11-17** - 🛠️ Repository initialized with documentation.  
- ✅ **2025-11-18** - 📄 Paper available on arXiv.  
- ✅ **2025-11-20** - 🚀 Provide core codebase, testing subset, and pre-trained models for evaluation.  
- TODO: Provide detailed testing instructions.

---

## 🚀 **Installation & Usage**

### 1. Clone the Repository

```bash
git clone https://github.com/marcotam2002/geounips.git
cd geounips
```

### 2. Environment Setup

```bash
conda create -n geounips python=3.10.19
conda activate geounips

pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

pip install einops
pip install opencv-python
```

### 3. Download the Model Weights

Download the checkpoint file from [this link](https://drive.google.com/file/d/1GWcdvsLMpjfaqWEvAskf7Tn1Nvv-KeKE/view?usp=sharing) and place it at `checkpoint/normal/ckpt.pytmodel`.

### 4. 🏁 **Quick Start** 

Once everything is set up, run the demo script with the following command. The --session_name argument specifies the output folder, and the --test_dir argument specifies the input image folder:

```bash
python geo_unips/main.py --session_name "path_to_output" --test_dir "path_to_input" --checkpoint checkpoint --max_image_num 4 --max_image_res 2048 --scalable --target normal
```

---

## 📜 Citation

If you find our work helpful, please consider citing:

```bibtex
@misc{tam2025geounips,
      title={Geometry Meets Light: Leveraging Geometric Priors for Universal Photometric Stereo under Limited Multi-Illumination Cues}, 
      author={King-Man Tam and Satoshi Ikehata and Yuta Asano and Zhaoyi An and Rei Kawakami},
      year={2025},
      eprint={2511.13015},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```

---

## 📝 License

This project is released under the [MIT License](LICENSE).

If you find this repository useful, please consider **starring ⭐** and **forking 🍴** it!
