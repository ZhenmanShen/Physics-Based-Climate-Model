# Physics-Based Climate Model Emulator (CSE 151B - Spring 2025)

This repository contains our final submission for the CSE 151B Climate Emulation competition. We designed a deep learning pipeline to emulate a physics-based climate model, progressively improving baseline models with hybrid architectures, temporal memory, and data-aware engineering.

---

## 🌍 Overview

The task is to emulate monthly global climate outputs—surface air temperature (`tas`) and precipitation (`pr`)—using input forcings (e.g., `CO2`, `SO2`, `CH4`, `rsdt`, `BC`) across various emissions scenarios (SSPs). The dataset includes ensemble members for internal variability.

---

## 📁 Project Structure
```
├── configs/                      # Hydra config files
│   ├── data/
│   │   ├── default.yaml
│   │   ├── data_ensemble.yaml
│   │   └── data_final.yaml
│   ├── model/
│   │   ├── SimpleCNN.yaml
│   │   ├── cnn_transformer.yaml
│   │   └── unet_convlstm_attention.yaml
│   ├── trainer/
│   │   ├── trainer/default.yaml
│   ├── training/
│   │   ├── training/default.yaml
│   └── main_config.yaml
├── data/processed.zarr/          # Dataset 
├── notebooks/                    # Data exploration notebook
│   └── data-exploration-basic.ipynb
├── src/                          # Model architecture implementations & utilities
│   ├── cnn_transformer.py
│   ├── unet_convlstm_attention.py
│   ├── models.py                 # SimpleCNN implementation resides in models.py
│   ├── utils_baseline.py         # Utility functions for baseline/ensemble
│   ├── utils_final.py            # Utility functions for final model (include updated normalization)
│   └── ...
├── main_baseline.py              # Starter model (no changes)
├── main_ensemble.py              # Uses all ensemble members and SSPs
├── main_final.py                 # Final model with attention + temporal modeling
├── README.md
├── requirements.txt
└── Makefile
```

---

## 🧠 Model Architectures

| Model Name                  | Description |
|----------------------------|-------------|
| **cnn_baseline**           | Basic convolutional model from starter code (defined in `utils_baseline.py`). No temporal modeling. |
| **cnn_transformer**        | Adds global spatial awareness via Transformer layers after a CNN encoder. |
| **unet_convlstm_attention** | Final model: U-Net (for spatial features) + ConvLSTM (for temporal memory) with attention and seasonality. |

> **Note:** Only `unet_convlstm_attention` expects a temporal input (e.g. `seq_len=6`). Others use single frames.

---

## 🔧 Model Configuration Files

Each model has a dedicated YAML config in `configs/model/`, where you can easily change hyperparameters like number of layers or embedding size.

| YAML File                          | Associated Model                | Key Settings You Can Tune                        |
|-----------------------------------|----------------------------------|--------------------------------------------------|
| `SimpleCNN.yaml`                  | `SimpleCNN`                      | Input/output channels, kernel size              |
| `cnn_transformer.yaml`            | `cnn_transformer`                | `embed_dim`, `n_heads`, `depth`, `mlp_dim`      |
| `unet_convlstm_attention.yaml`    | `unet_convlstm_attention`        | `base` channel size, `seq_len`, temporal depth  |

To switch models, update `configs/main_config.yaml`:
```yaml
defaults:
  - data: default
  - model: cnn_transformer  # change to the desired model name
  - training: default
  - trainer: default
  - _self_
```
For the **final model**, change ```data``` to ```data_final```
```yaml
defaults:
  - data: data_final
  - model: unet_convlstm_attention
  - training: default
  - trainer: default
  - _self_
```

---

## 🧰 Utility Scripts
| File               | Purpose                                                                                         |
|--------------------|-------------------------------------------------------------------------------------------------|
| `utils_baseline.py` | Used by `main_baseline.py` and `main_ensemble.py`. Handles standard data loading, metrics, plotting. |
| `utils_final.py`    | Used by `main_final.py`. Adds variable-specific normalization, sliding window construction, and seasonal embeddings. |

---

## ⚙️ How to Run

### Step 1 – Install Dependencies

Set up your environment and install all required packages:

```bash
pip install -r requirements.txt
```

### Step 2 Run a Model

You can run any of the three main scripts directly:

```
# Baseline CNN-Transformer (starter code + utils_baseline.py)
python main_baseline.py

# Ensemble-augmented CNN-Transformer (uses all SSPs + member IDs)
python main_ensemble.py

# Final model (U-Net + ConvLSTM with normalization and seasonality)
python main_final.py
```

⚠️ Important: Before running a model, make sure ```configs/main_config.yaml``` is properly configured.

For ```main_baseline.py```, use 
```
defaults:
  - data: default
  - model: cnn_transformer  # or cnn_baseline, cnn_transformer_attention
  - training: default
  - trainer: default
  - _self_
```

For ```main_ensemble.py```, use 
```
defaults:
  - data: data_ensemble
  - model: cnn_transformer  # or cnn_baseline, cnn_transformer_attention
  - training: default
  - trainer: default
  - _self_
```

For ```main_final.py```, use 
```
defaults:
  - data: data_final
  - model: unet_convlstm_attention
  - training: default
  - trainer: default
  - _self_
```
