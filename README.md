# KGCS: Zero-Annotation Expert-Knowledge Injection for Object Detection in Aerial Images

<div align="center">
📄 Our Paper Address: (Your paper link here)
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)

**K**nowledge-**G**uided **C**ollaborative **S**ystem for Zero-Annotation Aerial Object Detection.

This project introduces **KGCS**, a novel zero-annotation framework that integrates domain expert knowledge with frozen foundation models (SAM, CLIP, GPT-4o) to achieve accurate aerial object detection without any training data. The system features a three-stage modular pipeline designed to overcome structural ambiguity, semantic discrimination difficulty, and annotation scarcity in remote sensing imagery.

> **Code is Coming Soon!**  
> The project is under active development, and code will be released soon. Please click "Watch" and "Star" in the upper right corner to stay updated!

## 📋 Table of Contents

* [Architecture Overview](#-architecture-overview)
* [Features](#-features)
* [Installation](#-installation)
* [🚀 Quick Start](#-quick-start)
  * [1. Dataset Format](#1-dataset-format)
  * [2. Scene Description Module (SDM)](#2-scene-description-module-sdm)
  * [3. Object Proposal Module (OPM)](#3-object-proposal-module-opm)
  * [4. Image-Text Similarity Module (ISM)](#4-image-text-similarity-module-ism)
  * [5. Full Pipeline Inference](#5-full-pipeline-inference)
* [Experiments](#-experiments)
* [Contributing](#-contributing)
* [License](#-license)
* [Citation](#citation)
* [Acknowledgments](#-acknowledgments)

## 🏗 Architecture Overview

The KGCS framework is structured as a three-stage pipeline that collaboratively leverages frozen foundation models guided by structured expert knowledge:

1. **Scene Description Module (SDM)**: Constructs a hierarchical semantic dictionary by integrating expert knowledge and LLM reasoning to enhance CLIP’s fine-grained discrimination.
2. **Object Proposal Module (OPM)**: Generates high-quality region proposals via a structure-aware dual-path strategy to overcome SAM’s fragmentation in aerial scenes.
3. **Image-Text Similarity Module (ISM)**: Performs cross-modal alignment with adaptive two-stage filtering to ensure robust classification under zero-annotation constraints.

## ✨ Features

* **Zero Annotation Required**: No training data, manual labels, or model fine-tuning needed.
* **Expert Knowledge Injection**: Integrates domain-specific priors to bridge the aerial domain gap.
* **Multi-Model Collaboration**: Orchestrates SAM, CLIP, and GPT-4o in a frozen, interpretable pipeline.
* **Structure-Aware Design**: Dual-path proposal strategy handles both fixed-boundary and composite targets.
* **Cost-Efficient Deployment**: One-time dictionary construction and lightweight inference enable scalable deployment on single-GPU systems.

## ⚙️ Installation

Ensure your environment meets the following requirements:

* Python 3.8+
* PyTorch (with CUDA if available)
* HuggingFace Transformers
* OpenCV, PIL
* (Optional) GPT-4o API access for SDM

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/KGCS.git
   cd KGCS
   ```

2. **Create and activate a Python environment**:

   ```bash
   conda create -n kgcs python=3.8
   conda activate kgcs
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### 1. Dataset Format

KGCS is designed for aerial imagery datasets such as **DIOR** and **DOTA**. Images should be in standard format (e.g., JPEG, PNG) with optional JSON annotations for evaluation only.

For zero-annotation inference, only images are required.

### 2. Scene Description Module (SDM)

SDM builds a discriminative semantic dictionary using expert knowledge and GPT-4o.

```python
from kgcs.sdm import SceneDescriptionModule

sdm = SceneDescriptionModule(expert_knowledge_path="config/expert_descriptions.json")
dictionary = sdm.build_dictionary(image_list, target_categories)
```

*Dictionary includes target descriptions, distractors, and contour features.*

### 3. Object Proposal Module (OPM)

OPM uses SAM and a dual-path strategy to generate candidate regions.

```python
from kgcs.opm import ObjectProposalModule

opm = ObjectProposalModule()
proposals_clear, proposals_ambiguous = opm.generate_proposals(image)
```

*Proposals are categorized into contour-clear and boundary-ambiguous targets.*

### 4. Image-Text Similarity Module (ISM)

ISM aligns visual proposals with semantic descriptions via CLIP and applies adaptive filtering.

```python
from kgcs.ism import ImageTextSimilarityModule

ism = ImageTextSimilarityModule(dictionary)
detections = ism.filter_and_classify(proposals_clear, proposals_ambiguous)
```

*Output includes bounding boxes, labels, and confidence scores.*

### 5. Full Pipeline Inference

Run the complete KGCS pipeline on an aerial image:

```python
from kgcs.pipeline import KGCSPipeline

pipeline = KGCSPipeline(config_path="config/kgcs_config.yaml")
results = pipeline.detect("path/to/aerial_image.jpg")
```

## 📊 Experiments

KGCS is evaluated under strict zero-annotation protocols on DIOR and DOTA datasets.

| Dataset | Setting | Recall@100 | mAP |
|---------|---------|------------|-----|
| DIOR    | Novel   | 42.9%      | 8.4% |
| DIOR    | All     | 48.8%      | 15.4% |
| DOTA    | Novel   | 52.6%      | 5.3% |
| DOTA    | All     | 51.0%      | 5.4% |

*For detailed results and comparisons, refer to the paper.*

## 🤝 Contributing

We welcome contributions including:

* 📖 Documentation improvements
* 🐛 Bug reports
* 💡 Feature suggestions
* 🔧 Pull requests

Please see [CONTRIBUTING.md](CONTRIBUTING.md) (to be created) for guidelines.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## ✒️ Citation

```bibtex
@article{hu2025kgcs,
  title={KGCS: Zero-Annotation Expert-Knowledge Injection for Object Detection in Aerial Images},
  author={Hu, Wei and Hu, Suhang and Ma, Fei and Zhao, Qihao and Zhang, Fan},
  journal={arXiv preprint},
  year={2025}
}
```

## 🙏 Acknowledgments

This work builds upon several foundational models and open-source efforts:

* **Segment Anything Model (SAM)** for zero-shot segmentation
* **CLIP** for vision-language alignment
* **GPT-4o** for semantic reasoning and description generation
* **DIOR & DOTA Datasets** for aerial imagery evaluation
* The open-source remote sensing and computer vision communities for continuous inspiration and support.
