#!/usr/bin/env bash
# set -euo pipefail

ROOT_DIR="$(pwd)"
MODEL_DIR="${ROOT_DIR}/models"
BENCH_DIR="${ROOT_DIR}/benchmarks"

mkdir -p "$MODEL_DIR" "$BENCH_DIR"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  python3 -m pip install huggingface_hub
fi

echo "Downloading VACE base model..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-VACE-14B', local_dir='${MODEL_DIR}/Wan2.1-VACE-14B', local_dir_use_symlinks=False)"

echo "Downloading optional benchmark examples..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ali-vilab/VACE-Benchmark', repo_type='dataset', local_dir='${BENCH_DIR}/VACE-Benchmark', local_dir_use_symlinks=False)" || true

cat <<'EOF'

Base assets are ready.

If you already have a CausVid LoRA checkpoint, place it anywhere local and pass:
  --lora_path /absolute/path/to/your_causvid_lora.safetensors

If your CausVid adapter is stored as a .pt file, this project can also load it directly.
EOF
