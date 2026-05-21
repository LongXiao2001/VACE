#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m pip install --upgrade pip
python3 -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install -r requirements.txt
python3 -m pip install "wan@git+https://github.com/Wan-Video/Wan2.1"
python3 -m pip install safetensors huggingface_hub

echo "Environment is ready under: $ROOT_DIR"
