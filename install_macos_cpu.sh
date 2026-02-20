#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-wham-mac}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH. Install Miniconda/Anaconda first."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/6] Creating conda env: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.10 -y

echo "[2/6] Upgrading pip/setuptools/wheel"
conda run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel

echo "[3/6] Installing PyTorch CPU"
conda run -n "$ENV_NAME" python -m pip install torch torchvision torchaudio

echo "[4/6] Installing FFmpeg + chumpy from conda-forge"
conda install -n "$ENV_NAME" -c conda-forge ffmpeg chumpy -y

echo "[5/6] Installing WHAM deps (excluding chumpy/setuptools pin)"
grep -Ev '^(chumpy|setuptools==)' requirements.txt > requirements_no_chumpy.txt
conda run -n "$ENV_NAME" python -m pip install -r requirements_no_chumpy.txt

echo "[6/6] Setup complete"
echo "Activate with: conda activate $ENV_NAME"
echo "Then fetch models: bash fetch_demo_data.sh"
