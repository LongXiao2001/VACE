#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

: "${CKPT_DIR:=models/Wan2.1-VACE-1.3B}"
: "${POSE_VIDEO:=benchmarks/VACE-Benchmark/assets/examples/pose/src_video.mp4}"
: "${PROMPT:=在一个热带的庆祝派对上，一家人围坐在椰子树下的长桌旁。桌上摆满了异国风味的美食。长辈们愉悦地交谈，年轻人兴奋地举杯碰撞，孩子们在沙滩上欢乐奔跑。背景中是湛蓝的海洋和明亮的阳光，营造出轻松的气氛。镜头以动态中景捕捉每个开心的瞬间，温暖的阳光映照着他们幸福的面庞。}"

python vace/vace_wan_long_inference.py \
  --ckpt_dir "$CKPT_DIR" \
  --src_video "$POSE_VIDEO" \
  --prompt "$PROMPT" \
  --window_frames 81 \
  --overlap_frames 17 \
  --strategy keyframe \
  --anchor_stride 2 \
  --blend_frames 8 \
  --sample_steps 40 \
  --sample_shift 16 \
  --sample_guide_scale 5.0 \
  --offload_model true \
  --layer_vram_management true \
  --num_persistent_param_in_dit 6000000000 \
  --layer_vram_verbose true \
  --memory_log true \
  --memory_log_prefix "VACE-LongPose"
