<div align="center">

# ArogyaSutra

### A Multi-Agent Framework for Multimodal Medical Reasoning in Indic Languages

[![HuggingFace](https://img.shields.io/badge/🤗_Models_&_Datasets-yellow?style=flat-square)](https://huggingface.co/collections/iit-patna-cse-ai/arogyasutra)
[![Paper](https://img.shields.io/badge/📃_Paper-Coming_Soon-grey?style=flat-square)](#)
[![Project Page](https://img.shields.io/badge/🌐_Project_Page-iitp--cse.github.io-green?style=flat-square)](https://iitp-cse.github.io/ArogyaSutra/)
[![IJCAI 2026](https://img.shields.io/badge/IJCAI-2026-blue?style=flat-square)](https://2026.ijcai.org/)

</div>

---

## Abstract

Existing MLLMs, predominantly trained on English-centric data, struggle to support healthcare use cases in low-resource Indic language settings. We address this with two contributions: **ArogyaBodha**, a multilingual multimodal medical QA dataset of 40,857 samples from 8 sources covering 31 body systems, 6 imaging modalities, and 21 clinical domains across English and 7 Indian languages; and **ArogyaSutra**, an actor–critic multi-agent framework combining tool-based visual grounding with dual-memory mechanisms for step-wise, error-aware medical reasoning. Experiments show consistent improvements over strong baselines across all Indic languages.

---

## Contributions

- **ArogyaBodha** — 40,857 expert-verified multimodal medical QA samples across English, Bengali, Hindi, Assamese, Tamil, Telugu, Punjabi, and Marathi, curated from 8 heterogeneous medical sources.
- **ArogyaSutra Framework** — An actor–critic multi-agent system with tool-based image grounding, code-switching, and dual short/long-term memory for iterative error correction.
- **Actor–Critic Training Data** — Automated rollout pipeline generating critic-refined supervision data (`ArogyaBodha_ActCri`) for fine-tuning.
- **Fine-tuned Model** — `arogyasutraV1E3.5`, a Qwen2.5-VL-7B model distilled from actor–critic rollouts, achieving 43.40% avg accuracy across 7 Indic languages.

---

## Framework

ArogyaSutra uses an actor–critic loop where both agents share the **Qwen2.5-VL-7B** backbone. The Actor grounds reasoning in medical images via four vision tools; the Critic evaluates outputs and either approves or triggers error-specific feedback.

**Vision Tools:** Object Detection · Zoom/Crop · Edge Detection · Depth Estimation

**Memory:** Short-term captures the latest error; long-term summarizes all prior errors across the reasoning trajectory. At inference, only the distilled Actor is used — no Critic needed.

![ArogyaSutra Framework](https://raw.githubusercontent.com/IITP-CSE/ArogyaSutra/web/web/actor-critic_2.jpg)

---

## Performance

Zero-shot evaluation on 130 samples/language (910 total). Metric: choice accuracy.

### Indic Language Benchmark

| Model | AS | BN | HI | MR | PB | TA | TE | **Avg** |
|---|---|---|---|---|---|---|---|---|
| GPT-4.0 | 38.40 | 44.60 | 40.10 | 37.80 | 39.20 | 36.90 | 39.10 | 39.30 |
| Mistral-Small-3.2-24B | 40.90 | 42.90 | 44.10 | 43.20 | 40.90 | 41.40 | 42.50 | 42.27 |
| Qwen2.5-VL-7B (base) | 32.30 | 36.50 | 33.84 | 36.12 | 32.30 | 35.38 | 33.07 | 34.21 |
| MedGemma-4B-it | 38.50 | 35.10 | 37.40 | 34.50 | 35.70 | 36.70 | 34.90 | 36.11 |
| MedVLM-R1 | 23.50 | 22.80 | 25.60 | 22.80 | 24.20 | 25.40 | 22.20 | 23.79 |
| **ArogyaSutra (7B)** | **47.69** | **45.38** | **42.30** | **42.30** | **43.07** | **41.53** | **41.53** | **43.40** |

> ArogyaSutra outperforms GPT-4.0 by **+4.1 pts** and its own base model by **+9.2 pts**.

### Out-of-Distribution (100 real clinical queries)

| Model | Accuracy |
|---|---|
| Qwen2.5-VL-7B (base) | 35.0 |
| **ArogyaSutra (7B)** | **50.4** |

---

## Ablation Study Results

### Component Ablation

| Configuration | **Avg Accuracy** |
|---|---|
| **Full ArogyaSutra** | **43.40** |
| w/o Critic & Image Grounding | 33.43 |
| w/o Critic | 33.71 |
| w/o Image Grounding | 26.86 |
| w/o Code-switching | 26.86 |

### Image Grounding × Memory

| Image Grounding | Memory | **Avg** |
|---|---|---|
| ✗ | ✓ | 26.86 |
| ✓ | ✗ | 30.57 |
| ✓ | ✓ | **43.40** |

Removing image grounding or code-switching causes the steepest drops (−16.5 pts each). Memory alone is insufficient without grounding; together they are complementary and essential.

---

## Repository Structure

```
ArogyaSutra/
├── gen/                    # Actor-critic rollout & data generation
├── train/                  # Fine-tuning scripts (Unsloth + Qwen2.5)
├── test/                   # Evaluation scripts
├── tools/
│   ├── Depth-Anything-V2/  # Depth estimation service
│   └── LLMDet/             # Object detection service
├── verl/                   # RL utilities
├── gpt-4o-mini/MedQA/      # GPT-4o-mini baseline results
├── qwen/MedQA/             # Qwen baseline results
├── generate.sh             # Generate actor-critic training data
├── train.sh                # Fine-tune model
├── test.sh                 # Run evaluation
├── tools.sh                # Start tool services
├── tserv.sh                # Start vLLM server
└── setup.py
```

---

## Pipeline Overview

```
[1] Data Prep → [2] Tool Services → [3] Generate ActCri Data → [4] Fine-tune → [5] Evaluate
```

1. **Data Prep** — Download `ArogyaBodha` from Hugging Face.
2. **Tool Services** — Start Depth-Anything-V2 and LLMDet as background microservices.
3. **Generate ActCri Data** — Run actor–critic rollouts to produce `ArogyaBodha_ActCri` supervision data.
4. **Fine-tune** — LoRA fine-tune Qwen2.5-VL-7B on the critic-refined traces (3 epochs, ~12–15 hrs on A100).
5. **Evaluate** — Serve the distilled model via vLLM and run inference on the test set.

---

## Quick Start

### 1. Clone & Install Core Environment

```bash
git clone https://github.com/IITP-CSE/ArogyaSutra.git && cd ArogyaSutra

conda create -n asenv python=3.10 -y && conda activate asenv
pip install torch==2.6.0 torchvision==0.21.0
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -e ".[vllm]"
conda install -c conda-forge compilers
```

### 2. Install Tools Environment

```bash
conda create -n tools python=3.10 -y && conda activate tools
pip install unsloth==2025.9.7
pip install transformers==4.42.0 fastapi uvicorn matplotlib opencv-python python-multipart
conda install -c conda-forge compilers

# Download Depth-Anything-V2 checkpoint
cd tools/Depth-Anything-V2 && mkdir checkpoints && cd checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
cd ../../..
```

### 3. Run the Pipeline

```bash
# Start tool services (in a tmux session)
tmux new -s tools && bash tools.sh

# Generate actor-critic training data
conda activate asenv && bash generate.sh

# Fine-tune Qwen2.5
bash train.sh

# Start vLLM inference server
bash tserv.sh

# Evaluate
bash test.sh
```

---

## Citation

```bibtex
@inproceedings{halder2026arogyasutra,
  title     = {ArogyaSutra: A Multi-Agent Framework for Multimodal Medical Reasoning in Indic Languages},
  author    = {Halder, Tanmoy Kanti and Ghosh, Akash and Baidya, Subhadip and Roy, Arijit and Saha, Sriparna},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence (IJCAI)},
  year      = {2026},
  url       = {https://github.com/IITP-CSE/ArogyaSutra}
}
```

---

<div align="center">
<strong>IIT Patna (CSE) · IIT Kanpur · Prasannadeb Women's College</strong><br>
Supported by the Aryabhatta Supercomputing Centre, IIT Patna — National Supercomputing Mission, Govt. of India
</div>
