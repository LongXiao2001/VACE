# -*- coding: utf-8 -*-
import argparse
import logging
import math
import os
import shutil
import tempfile
from datetime import datetime

import numpy as np
import torch
from PIL import Image

from annotators.utils import read_video_frames, save_one_video
from vace_wan_inference import (
    _init_logging,
    get_parser as get_base_parser,
    validate_args,
)
from models.wan import WanVace
from models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS


def get_parser():
    parser = argparse.ArgumentParser(
        description="Long pose-driven video generation based on VACE-Wan"
    )
    base_parser = get_base_parser()
    for action in base_parser._actions:
        if action.dest != "help":
            parser._add_action(action)

    parser.set_defaults(src_mask=None)
    parser.add_argument(
        "--window_frames",
        type=int,
        default=81,
        help="Frames per generation window. Must be 4n+1.",
    )
    parser.add_argument(
        "--overlap_frames",
        type=int,
        default=17,
        help="Overlapped frames between adjacent windows.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="keyframe",
        choices=["keyframe", "autoregressive"],
        help="Long video strategy: two-pass keyframe refinement or autoregressive continuation.",
    )
    parser.add_argument(
        "--anchor_stride",
        type=int,
        default=2,
        help="Generate one anchor segment every N windows during keyframe first pass.",
    )
    parser.add_argument(
        "--blend_frames",
        type=int,
        default=8,
        help="Blend this many frames when stitching adjacent windows.",
    )
    parser.add_argument(
        "--keep_temps",
        action="store_true",
        help="Keep temporary segment inputs/outputs for debugging.",
    )
    return parser


def _validate_long_args(args):
    if args.src_video is None:
        raise ValueError("`--src_video` is required for pose long video generation.")
    if (args.window_frames - 1) % 4 != 0:
        raise ValueError("`--window_frames` must satisfy 4n+1.")
    if args.overlap_frames >= args.window_frames:
        raise ValueError("`--overlap_frames` must be smaller than `--window_frames`.")
    if args.overlap_frames < 1:
        raise ValueError("`--overlap_frames` must be positive.")
    return args


def _make_windows(total_frames, window_frames, overlap_frames):
    stride = window_frames - overlap_frames
    starts = list(range(0, max(total_frames - window_frames, 0) + 1, stride))
    last_start = max(total_frames - window_frames, 0)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _pad_frames(frames, target_length):
    if len(frames) >= target_length:
        return frames[:target_length]
    pad_frame = frames[-1]
    return frames + [pad_frame.copy() for _ in range(target_length - len(frames))]


def _save_video_segment(segment_frames, fps, file_path):
    save_one_video(file_path, segment_frames, fps=fps, quality=8, macro_block_size=None)


def _save_image(frame, file_path):
    Image.fromarray(frame.astype(np.uint8)).save(file_path)


def _tensor_to_uint8_frames(video_tensor):
    video_tensor = video_tensor.detach().cpu().float().clamp(-1, 1)
    video_tensor = ((video_tensor + 1.0) * 127.5).round().byte()
    return [frame.permute(1, 2, 0).numpy() for frame in video_tensor.permute(1, 0, 2, 3)]


def _blend_videos(base_frames, next_frames, overlap_frames, blend_frames):
    if not base_frames:
        return list(next_frames)

    blend_frames = min(blend_frames, overlap_frames, len(base_frames), len(next_frames))
    output = list(base_frames[:-overlap_frames])
    if blend_frames > 0:
        for idx in range(overlap_frames):
            prev = base_frames[-overlap_frames + idx].astype(np.float32)
            curr = next_frames[idx].astype(np.float32)
            if idx < blend_frames:
                alpha = float(idx + 1) / float(blend_frames + 1)
                mixed = np.clip(prev * (1.0 - alpha) + curr * alpha, 0, 255).astype(np.uint8)
            else:
                mixed = curr.astype(np.uint8)
            output.append(mixed)
    else:
        output.extend(next_frames[:overlap_frames])
    output.extend(next_frames[overlap_frames:])
    return output


def _build_segment_refs(segment_index, segment_outputs, temp_dir, strategy, input_refs=None, anchor_stride=2):
    refs = []
    if input_refs:
        refs.extend(input_refs)

    if strategy == "autoregressive":
        if segment_index > 0 and segment_outputs[segment_index - 1] is not None:
            prev_last = segment_outputs[segment_index - 1][-1]
            prev_path = os.path.join(temp_dir, f"seg_{segment_index:04d}_prev.png")
            _save_image(prev_last, prev_path)
            refs.insert(0, prev_path)
        return refs or None

    prev_anchor = None
    next_anchor = None
    for idx in range(segment_index, -1, -1):
        if segment_outputs[idx] is not None:
            prev_anchor = idx
            break
    for idx in range(segment_index, len(segment_outputs)):
        if segment_outputs[idx] is not None:
            next_anchor = idx
            break

    if prev_anchor is not None:
        prev_last = segment_outputs[prev_anchor][-1]
        prev_path = os.path.join(temp_dir, f"seg_{segment_index:04d}_anchor_prev.png")
        _save_image(prev_last, prev_path)
        refs.insert(0, prev_path)
    if next_anchor is not None and next_anchor != prev_anchor:
        next_first = segment_outputs[next_anchor][0]
        next_path = os.path.join(temp_dir, f"seg_{segment_index:04d}_anchor_next.png")
        _save_image(next_first, next_path)
        refs.append(next_path)
    return refs or None


def _run_single_segment(
    wan_vace,
    cfg,
    args,
    pose_segment_path,
    segment_refs,
    device,
):
    src_video, src_mask, src_ref_images = wan_vace.prepare_source(
        [pose_segment_path],
        [None],
        [segment_refs],
        args.window_frames,
        SIZE_CONFIGS[args.size],
        device,
    )
    video = wan_vace.generate(
        args.prompt,
        src_video,
        src_mask,
        src_ref_images,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.window_frames,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model,
    )
    return _tensor_to_uint8_frames(video)


def main(args):
    args = argparse.Namespace(**args) if isinstance(args, dict) else args
    args = validate_args(args)
    args = _validate_long_args(args)
    args.frame_num = args.window_frames

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)
    if world_size != 1:
        raise NotImplementedError("Long pose inference currently targets single-GPU execution.")

    cfg = WAN_CONFIGS[args.model_name]
    logging.info("Creating WanVace pipeline for long pose generation.")
    wan_vace = WanVace(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )
    wan_vace.configure_runtime(
        block_swap=args.block_swap,
        vace_hint_cpu_offload=args.vace_hint_cpu_offload,
        lora_path=args.lora_path,
        lora_scale=args.lora_scale,
        lora_dyn_offload=args.lora_dyn_offload,
        module_vram_management=args.module_vram_management,
        module_vram_empty_cache=args.module_vram_empty_cache,
        module_vram_verbose=args.module_vram_verbose,
        module_vram_keep_resident=args.module_vram_keep_resident,
    )

    frames, fps, _, _, total_frames = read_video_frames(args.src_video, use_type="cv2", info=True)
    windows = _make_windows(total_frames, args.window_frames, args.overlap_frames)
    logging.info("Loaded pose video with %s frames, split into %s windows.", total_frames, len(windows))

    if args.save_dir is None:
        save_dir = os.path.join(
            "results",
            args.model_name,
            "long_pose_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        )
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix="vace_long_pose_", dir=save_dir)
    input_refs = None if args.src_ref_images is None else [p.strip() for p in args.src_ref_images.split(",") if p.strip()]
    segment_outputs = [None for _ in windows]

    try:
        segment_video_paths = []
        for seg_idx, start in enumerate(windows):
            clip_frames = _pad_frames(frames[start:start + args.window_frames], args.window_frames)
            pose_segment_path = os.path.join(temp_dir, f"pose_segment_{seg_idx:04d}.mp4")
            _save_video_segment(clip_frames, fps, pose_segment_path)
            segment_video_paths.append(pose_segment_path)

        if args.strategy == "keyframe":
            anchor_indices = sorted(set(list(range(0, len(windows), args.anchor_stride)) + [len(windows) - 1]))
            logging.info("First pass anchor windows: %s", anchor_indices)
            for seg_idx in anchor_indices:
                refs = _build_segment_refs(seg_idx, segment_outputs, temp_dir, "autoregressive", input_refs=input_refs)
                segment_outputs[seg_idx] = _run_single_segment(
                    wan_vace, cfg, args, segment_video_paths[seg_idx], refs, device
                )

            logging.info("Second pass refinement over all windows.")
            for seg_idx in range(len(windows)):
                if seg_idx in anchor_indices:
                    continue
                refs = _build_segment_refs(
                    seg_idx,
                    segment_outputs,
                    temp_dir,
                    "keyframe",
                    input_refs=input_refs,
                    anchor_stride=args.anchor_stride,
                )
                segment_outputs[seg_idx] = _run_single_segment(
                    wan_vace, cfg, args, segment_video_paths[seg_idx], refs, device
                )
        else:
            logging.info("Running autoregressive long video generation.")
            for seg_idx in range(len(windows)):
                refs = _build_segment_refs(seg_idx, segment_outputs, temp_dir, "autoregressive", input_refs=input_refs)
                segment_outputs[seg_idx] = _run_single_segment(
                    wan_vace, cfg, args, segment_video_paths[seg_idx], refs, device
                )

        merged_frames = []
        for segment_frames in segment_outputs:
            merged_frames = _blend_videos(
                merged_frames,
                segment_frames,
                overlap_frames=args.overlap_frames,
                blend_frames=args.blend_frames,
            )

        output_file = args.save_file or os.path.join(save_dir, "out_long_pose.mp4")
        save_one_video(output_file, merged_frames, fps=fps, quality=8, macro_block_size=None)
        logging.info("Saved long pose video to %s", output_file)
        return {
            "out_video": output_file,
            "save_dir": save_dir,
            "num_segments": len(windows),
        }
    finally:
        if not args.keep_temps and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main(get_parser().parse_args())
