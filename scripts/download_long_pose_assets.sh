#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT_DIR}/models"
BENCH_DIR="${ROOT_DIR}/benchmarks"

mkdir -p "$MODEL_DIR" "$BENCH_DIR"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  python3 -m pip install huggingface_hub
fi

echo "Downloading VACE base model..."
huggingface-cli download Wan-AI/Wan2.1-VACE-14B \
  --local-dir "${MODEL_DIR}/Wan2.1-VACE-14B" \
  --local-dir-use-symlinks False

echo "Downloading optional benchmark examples..."
huggingface-cli download ali-vilab/VACE-Benchmark \
  --repo-type dataset \
  --local-dir "${BENCH_DIR}/VACE-Benchmark" \
  --local-dir-use-symlinks False || true

cat <<'EOF'

Base assets are ready.

If you already have a CausVid LoRA checkpoint, place it anywhere local and pass:
  --lora_path /absolute/path/to/your_causvid_lora.safetensors

If your CausVid adapter is stored as a .pt file, this project can also load it directly.
EOF
