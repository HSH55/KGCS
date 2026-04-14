# KGCS: Zero-Annotation Expert-Knowledge Injection for Object Detection in Aerial Images

[![DOI](https://img.shields.io/badge/DOI-10.1109/TGRS.2026.3670221-blue)](https://doi.org/10.1109/TGRS.2026.3670221)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)

📄 **Paper**: [https://doi.org/10.1109/TGRS.2026.3670221](https://doi.org/10.1109/TGRS.2026.3670221)

**K**nowledge-**G**uided **C**ollaborative **S**ystem (**KGCS**) is a zero-annotation framework for aerial object detection that integrates structured expert knowledge with frozen foundation models.

> 🚧 **Code Coming Soon**
> The repository is under active development. Please ⭐ Star and 👀 Watch this repo for updates!

---

## 📌 Highlights

* 🚫 **Zero Annotation**: No training data or manual labels required
* 🧠 **Expert Knowledge Injection**: Bridges domain gaps in aerial imagery
* 🤖 **Foundation Model Collaboration**: Combines SAM, CLIP, and GPT-4o
* 🧩 **Three-Stage Pipeline**: Modular and interpretable design
* ⚡ **Efficient Inference**: Runs on a single GPU with no fine-tuning

---

## 🏗 Architecture Overview

KGCS is built on a three-stage collaborative pipeline:

1. **Scene Description Module (SDM)**
   Constructs a hierarchical semantic dictionary using expert knowledge and LLM reasoning.

2. **Object Proposal Module (OPM)**
   Generates structure-aware region proposals to address fragmentation in aerial scenes.

3. **Image-Text Similarity Module (ISM)**
   Performs cross-modal alignment and adaptive filtering for robust classification.

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/HSH55/KGCS.git
cd KGCS
```

### 2. Environment Setup

```bash
conda create -n kgcs python=3.8
conda activate kgcs
pip install -r requirements.txt
```

### 3. Run Pipeline (Coming Soon)

```python
from kgcs.pipeline import KGCSPipeline

pipeline = KGCSPipeline(config_path="config/kgcs_config.yaml")
results = pipeline.detect("path/to/image.jpg")
```

---

## 📊 Experiments

KGCS is evaluated under strict zero-annotation settings on DIOR and DOTA datasets.

| Dataset | Setting | Recall@100 | mAP   |
| ------- | ------- | ---------- | ----- |
| DIOR    | Novel   | 42.9%      | 8.4%  |
| DIOR    | All     | 48.8%      | 15.4% |
| DOTA    | Novel   | 52.6%      | 5.3%  |
| DOTA    | All     | 51.0%      | 5.4%  |

---

## 📄 Paper

📄 **IEEE Version**: [https://doi.org/10.1109/TGRS.2026.3670221](https://doi.org/10.1109/TGRS.2026.3670221)

---

## ✒️ Citation

If you find this work useful, please cite:

```bibtex
@ARTICLE{11419164,
  author={Hu, Wei and Hu, Suhang and Ma, Fei and Zhao, Qihao and Zhang, Fan},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={KGCS: Zero-Annotation Expert-Knowledge Injection for Object Detection in Aerial Images},
  year={2026},
  volume={64},
  pages={1-16},
  doi={10.1109/TGRS.2026.3670221}
}
```

---

## 🤝 Contributing

We welcome contributions including:

* 📖 Documentation improvements
* 🐛 Bug reports
* 💡 Feature suggestions
* 🔧 Pull requests

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 🙏 Acknowledgments

This work builds upon several influential models and datasets:

* Segment Anything Model (SAM)
* CLIP
* GPT-4o
* DIOR & DOTA datasets

---

⭐ If you find this project useful, please consider giving it a star!
