# Long Pose Guide

This extension adds a long pose-driven video workflow on top of VACE-Wan.

## What was added

- `vace/vace_wan_long_inference.py`
  - `strategy=keyframe`: first generate sparse anchor windows, then refine all windows with anchor first/last-frame references.
  - `strategy=autoregressive`: continue segment by segment with the previous generated tail frame as reference.
- LoRA runtime support in `WanVace`
  - accepts `.safetensors` and `.pt`
  - can apply/remove LoRA on demand during denoising
- VRAM optimizations
  - `--layer_vram_management true`: Pusa/DiffSynth-style layer wrapper offload
  - `--num_persistent_param_in_dit 6000000000`: how many DiT parameters may stay GPU-persistent
  - `--layer_vram_verbose true`: print wrapped layer onload/offload traces
  - `--module_vram_management true`: DiffSynth-style per-module load/unload scheduling
  - `--module_vram_empty_cache true`: clear allocator cache after unloading managed modules
  - `--module_vram_verbose true`: print each module load/unload together with live/max VRAM
  - `--block_swap true`: swap transformer blocks between CPU and GPU during forward
  - `--vace_hint_cpu_offload true`: keep VACE control hints on CPU between block calls
  - VACE control path no longer keeps the stacked `c` tensor structure across all blocks
  - `--memory_log true`: print stage-level and denoising-step VRAM stats

## Setup

```bash
cd /Users/longxiao/Documents/projects/video_gen/VACE
bash scripts/setup_long_pose_env.sh
bash scripts/download_long_pose_assets.sh
```

## Run

```bash
cd /Users/longxiao/Documents/projects/video_gen/VACE
bash run_vace_wan_long_pose.sh
```

## Common custom commands

Keyframe two-pass long pose:

```bash
python vace/vace_wan_long_inference.py \
  --ckpt_dir models/Wan2.1-VACE-1.3B \
  --src_video /absolute/path/to/pose.mp4 \
  --prompt "your prompt" \
  --window_frames 81 \
  --overlap_frames 17 \
  --strategy keyframe \
  --anchor_stride 2 \
  --layer_vram_management true \
  --num_persistent_param_in_dit 6000000000 \
  --layer_vram_verbose true \
  --offload_model true \
  --memory_log true
```

Autoregressive continuation:

```bash
python vace/vace_wan_long_inference.py \
  --ckpt_dir models/Wan2.1-VACE-1.3B \
  --src_video /absolute/path/to/pose.mp4 \
  --prompt "your prompt" \
  --window_frames 81 \
  --overlap_frames 17 \
  --strategy autoregressive \
  --layer_vram_management true \
  --num_persistent_param_in_dit 6000000000 \
  --layer_vram_verbose true \
  --offload_model true \
  --memory_log true
```

With a CausVid-style LoRA adapter:

```bash
python vace/vace_wan_long_inference.py \
  --ckpt_dir models/Wan2.1-VACE-1.3B \
  --src_video /absolute/path/to/pose.mp4 \
  --prompt "your prompt" \
  --lora_path /absolute/path/to/causvid_lora.safetensors \
  --lora_scale 1.0 \
  --lora_dyn_offload true \
  --strategy keyframe \
  --layer_vram_management true \
  --num_persistent_param_in_dit 6000000000 \
  --layer_vram_verbose true \
  --offload_model true \
  --memory_log true
```

## Notes

- The current long video entry is tuned for single GPU execution.
- `window_frames` must satisfy `4n+1`.
- `overlap_frames` should stay modest. `17` is a practical default for `81` frame windows.
- If you want the closest behavior to `Pusa-VidGen/PusaV1/diffsynth`, prefer `--layer_vram_management true` and tune `--num_persistent_param_in_dit`.
- The newer layer-level mode and the older module/block-level mode should not be mixed unless you are debugging.
- If you want to debug actual VRAM peaks on a 4090, add both `--module_vram_verbose true` and `--memory_log true`.
- If you want to inspect all temporary pose windows and intermediate anchor references, pass `--keep_temps`.
