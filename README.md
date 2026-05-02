# Ferrari: Federated Feature Unlearning: Proof of Concept

> **NeurIPS 2024** | Responsible & Explainable AI | FAST NUCES Islamabad | Spring 2026
> Katrina Bodani (22i0545, AI-8A) & Syed Zain Abbas Zaidi (22i1905, DS-8A)

---

## Overview

This repository is a proof-of-concept (POC) implementation of **Ferrari: Federated Feature Unlearning via Optimizing Feature Sensitivity** (NeurIPS 2024) by Win Kent Ong and Chee Sang Chan et al. Ferrari is the first federated feature unlearning framework that removes targeted feature knowledge from a global FL model without requiring participation from other clients.

We reproduce and demonstrate three unlearning scenarios:

| Scenario | Dataset | What is Unlearned | POC Status |
|---|---|---|---|
| **Backdoor** | MNIST | Pixel-pattern trigger | Working |
| **Bias** | Colored MNIST (CMNIST) | Background color bias | Working |
| **Sensitive** | Diabetes (Tabular) | Pregnancies feature | Working |

---

## Credits

**Original Paper:**
> Gu, H., Ong, W. K., Chan, C. S., & Fan, L. (2024). Ferrari: Federated Feature Unlearning via Optimizing Feature Sensitivity. *Advances in Neural Information Processing Systems*, 37, 24150–24180.
> 📄 [Official PDF](https://papers.nips.cc/paper_files/paper/2024/file/2b09bb02b90584e2be94ff3ae09289bc-Paper-Conference.pdf)

**Original Repository:**
> GitHub: [OngWinKent/Federated-Feature-Unlearning](https://github.com/OngWinKent/Federated-Feature-Unlearning)
> Authors: WinKent Ong, Hanlin Gu, Chee Seng Chan, Lixin Fan
> License: BSD-3

---

## Environment Setup

### Prerequisites
- Windows 10/11
- Python 3.9.12 (install separately — do NOT use Python 3.10+)
- NVIDIA GPU with CUDA support (tested on RTX 4050 6GB)

### Step 1 — Install Python 3.9.12
Download from [python.org](https://www.python.org/downloads/release/python-3912/) — use the Windows 64-bit installer. **Do NOT add to PATH** if you have another Python version installed.

### Step 2 — Clone the repository
```bash
git clone https://github.com/OngWinKent/Federated-Feature-Unlearning.git
cd Federated-Feature-Unlearning
```

### Step 3 — Create virtual environment with Python 3.9
```bash
py -3.9 -m venv ferrari_env
ferrari_env\Scripts\activate
```

### Step 4 — Install PyTorch (CUDA 11.8)
```bash
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118 --timeout 1000
```

### Step 5 — Install dependencies
```bash
pip install opencv-python==4.8.0.76 numpy==1.24.4
pip install -r requirement.txt
```

### Step 6 — Verify installation
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
Expected output:
```
2.0.0+cu118
True
NVIDIA GeForce RTX 4050 Laptop GPU
```

### Step 7 — Create data directory
```bash
mkdir data
mkdir experiments
```

---

## Scenario 1 — Backdoor Unlearning (MNIST)

A malicious federated client poisons the global model with a pixel-pattern trigger. Any image containing the trigger (5×5 white square at position (2,2)) gets misclassified as digit 0. Ferrari removes this backdoor without retraining.

### Training (200 global epochs, ~6 hours)
```bash
python fl_training_main.py -gpu -train_mode backdoor -dataset MNist -root "data" -checkpoint "experiments" -trigger_label 0 -trigger_size 5 -global_epochs 200 -local_epochs 5 -batch_size 128 -lr 0.0001 -client_num 10 -frac 0.4 -momentum 0.5 -optimizer sgd -seed 0 -report_training -save_model
```
### Unlearning (~30 seconds)
```bash
python unlearn_main.py -gpu -unlearning_scenario backdoor -dataset MNist -root "data" -checkpoint "experiments" -trigger_label 0 -trigger_size 5 -sample_number 20 -min_sigma 0.5 -max_sigma 1.0 -lr 0.000003 -client_num 10 -batch_size 128 -seed 0 -save_model
```

---

## Scenario 2 — Bias Unlearning (Colored MNIST)

The model is trained on colored MNIST where the **background** of digit 3 images is blue and digit 8 images is green. The model learns to classify by background color instead of digit shape. Ferrari removes this color bias.

### Training (~2 hours)
```bash
python fl_training_main.py -gpu -train_mode bias -dataset MNist -root "data" -checkpoint "experiments" -mnist_mode background -bias_ratio 0.8 -global_epochs 200 -local_epochs 5 -batch_size 128 -lr 0.0001 -client_num 10 -frac 0.4 -momentum 0.5 -optimizer sgd -seed 0 -report_training -save_model
```

### Unlearning
```bash
python unlearn_main.py -gpu -unlearning_scenario bias -dataset MNist -root "data" -checkpoint "experiments" -mnist_mode background -bias_ratio 0.8 -sample_number 20 -min_sigma 0.05 -max_sigma 1.0 -sigma 0.5 -lr 0.0000005 -client_num 10 -batch_size 128 -seed 0 -save_model
```
---

## Scenario 3 — Sensitive Feature Unlearning (Diabetes)

A federated client trains the model on medical records containing a sensitive attribute (number of pregnancies). Ferrari makes the model insensitive to this feature, predictions no longer change when pregnancies is perturbed.

### Training (~30 seconds)
```bash
python fl_training_main.py -gpu -train_mode sensitive -dataset diabetes -root "data" -checkpoint "experiments" -hidden_layer_num 64 -global_epochs 500 -local_epochs 10 -batch_size 16 -lr 0.01 -client_num 3 -frac 0.8 -momentum 0.9 -optimizer sgd -seed 0 -report_training -save_model
```

### Unlearning
```bash
python unlearn_main.py -gpu -unlearning_scenario sensitive -dataset diabetes -root "data" -checkpoint "experiments" -hidden_layer_num 64 -sample_number 20 -min_sigma 0.5 -max_sigma 1.0 -sigma 0.5 -lr 0.000003 -client_num 3 -batch_size 16 -seed 0 -save_model
```
---

## POC Dashboard

The dashboard is a Flask web application demonstrating all three unlearning scenarios interactively.

### Setup
```bash
pip install flask pillow
```

Place these files in the repo root:
```
Federated-Feature-Unlearning/
├── ferrari_dashboard.py
├── dashboard_static/
    └── index.html
```

### Run
```bash
python ferrari_dashboard.py
```

Open `http://localhost:5000` in your browser.

### Dashboard Tabs

**Tab 1 — Backdoor Unlearning**
- Upload any MNIST image.
- Shows clean vs triggered image side by side
- 4-panel prediction comparison (baseline vs unlearned)
- Trigger size auto-detected from filename

**Tab 2 — Bias Unlearning**
- Upload any coloured MNIST image.
- Shows correct-color vs color-swapped image
- Demonstrates baseline fooled by color, unlearned model uses shape

**Tab 3 — Sensitive Feature Unlearning**
- Enter health record values
- Shows prediction before/after pregnancies perturbation
- Baseline changes prediction, unlearned model stays consistent

---

## Repository Structure

```
Federated-Feature-Unlearning/
├── configs/                    # Training & unlearning argument configs
├── datasets/                   # Dataset construction (backdoor, bias, sensitive)
├── experiments/                # Saved model checkpoints
│   ├── MNist/backdoor/        # baseline.pth, unlearn.pth
│   ├── MNist/bias/            # baseline.pth, unlearn.pth
│   └── diabetes/sensitive/    # baseline.pth, unlearn.pth
├── fl_strategies/              # Federated learning training loop
├── model/                      # ResNet18, LinearModelTabular
├── unlearn_strategies/         # Lipschitz unlearning algorithm
├── dashboard_static/           # Dashboard frontend (index.html)
├── ferrari_dashboard.py        # Flask dashboard backend
├── fl_training_main.py         # Main FL training script
└── unlearn_main.py             # Main unlearning script
```

---

## Citation

```bibtex
@inproceedings{ferrari,
    title={Ferrari: Federated Feature Unlearning via Optimizing Feature Sensitivity},
    author={Hanlin Gu and WinKent Ong and Chee Seng Chan and Lixin Fan},
    booktitle={Advances in Neural Information Processing Systems},
    volume={37},
    pages={24150--24180},
    year={2024}
}
```
