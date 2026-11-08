#!/usr/bin/env bash
set -euo pipefail
# Requires python3-venv already installed (Lambda images usually have it).
# Run beside requirements.txt. CUDA wheel defaults to cu128; override with cu126
# only if your driver supports CUDA 12.6 but not 12.8.
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install torch==2.7.0 --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_WHEELS:-cu128}"
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name threeways-causal --display-name 'Python (Three Ways Causal)'
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: check NVIDIA driver/wheel compatibility"; print(torch.cuda.get_device_name(0))'
