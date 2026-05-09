# Pixels-to-Predictions: Multi-Modal Science QA with SmolVLM

A research-oriented repository exploring parameter-efficient fine-tuning and inference strategies for multi-modal science question answering using vision-language models.

This project investigates how lightweight adaptation methods such as LoRA and DoRA can improve reasoning performance on image-based science multiple-choice tasks while remaining computationally efficient.

---

# Features

- Multi-modal science question answering
- LoRA and DoRA fine-tuning
- Attention + MLP adapter experiments
- Native-resolution and high-resolution image training
- Metadata-aware prompting
- Ensemble inference strategies
- Confidence-based routing
- Test-time augmentation (TTA)
- Candidate-token constrained decoding
- Extensive ablation studies

---

# Model

Base model used throughout the experiments:

```text
HuggingFaceTB/SmolVLM-500M-Instruct
```

---

# Repository Structure

```text
.
├── notebooks/
├── checkpoints/
├── submissions/
├── report/
├── requirements.txt
└── README.md
```

## notebooks/
Contains training, inference, and ablation notebooks.

## checkpoints/
Contains final trained adapters and lightweight model checkpoints.

## submissions/
Contains generated prediction CSV files.

## report/
Contains the final report and LaTeX source.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Krishan101/dl-pixels-to-predictions.git
cd dl-pixels-to-predictions
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Requirements

Main dependencies:

```text
transformers==4.57.1
peft==0.18.0
bitsandbytes>=0.45.0
accelerate>=1.7.0
torch>=2.0
pandas==2.2.2
Pillow==11.3.0
numpy
scikit-learn
kagglehub
```

---

# Dataset Setup

Dataset download is handled using `kagglehub`.

```python
import kagglehub

kagglehub.login()

COMP_DIR = kagglehub.competition_download(
    "pixels-to-predictions"
)
```

Expected structure:

```text
pixels-to-predictions/
├── train.csv
├── val.csv
├── test.csv
├── sample_submission.csv
├── images/
```

---

# Main Experiments

The repository includes notebooks and submission files for:

- zero-shot baselines
- LoRA fine-tuning
- DoRA fine-tuning
- attention-only adaptation
- attention + MLP adaptation
- metadata prompting
- inference ablations
- ensemble methods
- high-resolution image training
- native-resolution image loading
- hard-sample fine-tuning

---

# Running Training

Example training notebook:

```text
notebooks/LoRA-v9-r8-all7-flatLR-trainval.ipynb
```

Main training strategies explored:

- rank-8 LoRA
- DoRA adapters
- attention + MLP targets
- metadata-aware prompts
- train+validation combined training
- cosine and flat LR schedules

---

# Running Inference

Inference uses constrained candidate-token scoring and log-probability ranking.

Example workflow:

```python
pred, scores = predict_score_one(row)
```

Submission generation:

```python
submission.to_csv("submission.csv", index=False)
```

---

# Ensemble Inference

The repository includes several ensemble strategies:

- score averaging
- prompt ensembling
- metadata-aware ensembling
- diverse-seed ensembles

Best ensemble submission:

```text
submissions/sub_score_average.csv
```

---

# Hardware

Experiments were conducted using:

- NVIDIA A100 40GB GPUs
- Google Colab environments

Approximate total compute:
- ~120 GPU-hours

---

# Notes

- Intermediate checkpoints are intentionally excluded to reduce repository size.
- The repository focuses on reproducibility and experimental organization.
- Final lightweight adapters and notebooks are included.

---

# Repository

https://github.com/Krishan101/dl-pixels-to-predictions