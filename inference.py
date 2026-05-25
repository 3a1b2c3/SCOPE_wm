# Copyright 2025 SCOPE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SCOPE inference script for action-conditioned game world video generation.

This script generates first-person game videos conditioned on player actions
(keyboard/controller buttons + joystick/mouse movement). It supports both
single-image and batch-directory inference modes.

Example usage:
    # Single image inference
    python inference.py \\
        --model_dir ./SCOPE \\
        --input_image examples/example_0/image.png \\
        --action_path examples/example_0/action.parquet \\
        --prompt "First-person shooter perspective in a toy garden"

    # Batch inference
    python inference.py \\
        --model_dir ./SCOPE \\
        --input_image_dir ./images \\
        --action_path action.parquet \\
        --prompt "First-person view" \\
        --output_dir ./outputs
"""

import argparse
import glob
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from diffsynth.models.scope_dit import WanModel as SCOPEDiT
from diffsynth.pipelines.scope_pipeline import ModelConfig
from diffsynth.pipelines.scope_pipeline import WanVideoPipeline
from diffsynth.utils.data import crop_and_resize
from diffsynth.utils.data import save_video

logger = logging.getLogger(__name__)

# ============================================================
# Model Configuration Constants
# ============================================================

# Controller button columns in the action parquet file.
# Each button is binary (0 = released, 1 = pressed).
BUTTON_COLS = [
    "right_trigger",  # Fire (RT)
    "left_trigger",   # Aim Down Sights (LT)
    "south",          # Jump (A)
    "right_thumb",    # Melee (R3)
    "west",           # Reload (X)
    "north",          # Weapon Switch (Y)
]

# ActionModule hyperparameters controlling how raw action signals are processed.
ACTION_CONFIG = {
    "mouse_dim_in": 4,                  # 2D left stick + 2D right stick
    "keyboard_dim_in": len(BUTTON_COLS),  # 6 binary buttons
    "dim": 3072,                        # Must match DiT hidden dim
    "num_heads": 24,                    # Must match DiT num_heads
    "vae_time_compression_ratio": 4,    # Wan2.2 VAE temporal compression
    "windows_size": 4,                  # Sliding window for temporal context
}

# Full DiT architecture configuration for SCOPE.
DIT_CONFIG = {
    "enable_action": True,
    "action_config": ACTION_CONFIG,
    "has_image_input": False,
    "patch_size": [1, 2, 2],
    "in_dim": 48,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "eps": 1e-06,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
}

# Negative prompt to guide generation away from common artifacts.
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


# ============================================================
# Helper Functions
# ============================================================


def load_actions(
    parquet_path: str,
    num_frames: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads keyboard and mouse actions from a parquet file.

    Reads the action parquet, pads if necessary, and returns tensors ready
    for the SCOPE pipeline.

    Args:
        parquet_path: Path to the `.parquet` file containing per-frame actions.
        num_frames: Number of frames to extract (pads with last row if short).
        device: Target device for the output tensors.

    Returns:
        A tuple of (keyboard, mouse) tensors:
            - keyboard: Shape [1, num_frames, 6], dtype bfloat16.
            - mouse: Shape [1, num_frames, 4], dtype bfloat16.
    """
    df = pd.read_parquet(parquet_path)

    # Pad with last row if parquet has fewer frames than requested.
    if len(df) < num_frames:
        pad = pd.concat(
            [df.iloc[[-1]]] * (num_frames - len(df)), ignore_index=True
        )
        df = pd.concat([df, pad], ignore_index=True)

    # Extract button presses as binary tensor.
    buttons = df[BUTTON_COLS].values[:num_frames].astype(np.float32)
    keyboard = torch.tensor(buttons).unsqueeze(0).to(device, torch.bfloat16)

    # Extract joystick axes (2D left + 2D right = 4D total).
    j_left = np.array(df["j_left"].tolist())[:num_frames].astype(np.float32)
    j_right = np.array(df["j_right"].tolist())[:num_frames].astype(np.float32)
    mouse = (
        torch.tensor(np.concatenate([j_left, j_right], axis=-1))
        .unsqueeze(0)
        .to(device, torch.bfloat16)
    )

    return keyboard, mouse


def compute_num_frames(
    max_frames: int = 81,
    divisor: int = 4,
    remainder: int = 1,
) -> int:
    """Computes valid frame count satisfying temporal alignment constraints.

    The Wan2.2 VAE requires frame counts of the form `n % divisor == remainder`.
    This function finds the largest valid `n <= max_frames`.

    Args:
        max_frames: Upper bound on frame count.
        divisor: Temporal divisibility factor (from VAE compression).
        remainder: Required remainder after division.

    Returns:
        The largest integer n <= max_frames where n % divisor == remainder.
    """
    n = max_frames
    while n > 1 and n % divisor != remainder:
        n -= 1
    return n


def find_model_files(model_dir: str) -> dict[str, any]:
    """Auto-discovers model files in the unified model directory.

    Expected layout::

        model_dir/
        +-- model-00001-of-00003.safetensors   # SCOPE DiT shards
        +-- model-00002-of-00003.safetensors
        +-- model-00003-of-00003.safetensors
        +-- models_t5_umt5-xxl-enc-bf16.pth    # Text encoder
        +-- Wan2.2_VAE.pth                     # VAE
        +-- google/umt5-xxl/                   # Tokenizer

    Args:
        model_dir: Root directory containing all model weights.

    Returns:
        Dictionary with keys: ``scope_shards``, ``t5_path``, ``vae_path``,
        ``tokenizer_path``, ``base_dit_shards``.

    Raises:
        FileNotFoundError: If required model files are missing.
    """
    # SCOPE DiT checkpoint shards.
    scope_shards = sorted(
        glob.glob(os.path.join(model_dir, "model-*-of-*.safetensors"))
    )
    if not scope_shards:
        single = os.path.join(model_dir, "SCOPE.safetensors")
        if os.path.isfile(single):
            scope_shards = [single]
        else:
            raise FileNotFoundError(
                f"No SCOPE checkpoint found in {model_dir}. "
                "Expected 'model-*-of-*.safetensors' or 'SCOPE.safetensors'."
            )

    # Text encoder (UMT5-XXL).
    t5_path = os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    if not os.path.isfile(t5_path):
        raise FileNotFoundError(f"Text encoder not found: {t5_path}")

    # Video VAE.
    vae_path = os.path.join(model_dir, "Wan2.2_VAE.pth")
    if not os.path.isfile(vae_path):
        raise FileNotFoundError(f"VAE not found: {vae_path}")

    # Tokenizer directory.
    tokenizer_path = os.path.join(model_dir, "google/umt5-xxl")
    if not os.path.isdir(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    # Base DiT (optional; used by pipeline model registry).
    base_dit_shards = sorted(
        glob.glob(
            os.path.join(
                model_dir, "diffusion_pytorch_model-*-of-*.safetensors"
            )
        )
    )

    return {
        "scope_shards": scope_shards,
        "t5_path": t5_path,
        "vae_path": vae_path,
        "tokenizer_path": tokenizer_path,
        "base_dit_shards": base_dit_shards,
    }


def init_pipeline(model_dir: str) -> WanVideoPipeline:
    """Initializes the SCOPE video generation pipeline.

    Loads the text encoder, VAE, and SCOPE DiT into a unified pipeline
    object ready for inference.

    Args:
        model_dir: Path to the model directory containing all weights.

    Returns:
        A fully initialized WanVideoPipeline with SCOPE DiT loaded.
    """
    logger.info("=" * 50)
    logger.info("Loading models from: %s", model_dir)
    logger.info("=" * 50)

    files = find_model_files(model_dir)

    # Build model configs for pipeline (text encoder + optional base DiT + VAE).
    model_configs = [
        ModelConfig(files["t5_path"]),
        ModelConfig(files["vae_path"]),
    ]
    if files["base_dit_shards"]:
        model_configs.insert(1, ModelConfig(files["base_dit_shards"]))

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=ModelConfig(files["tokenizer_path"]),
        redirect_common_files=False,
    )

    # Load SCOPE DiT (replaces base DiT in the pipeline).
    logger.info(
        "Loading SCOPE DiT (%d shard(s))...", len(files["scope_shards"])
    )
    dit = SCOPEDiT(**DIT_CONFIG).to(torch.bfloat16)

    state_dict = {}
    for shard in files["scope_shards"]:
        logger.info("  %s", os.path.basename(shard))
        state_dict.update(load_file(shard))

    missing, unexpected = dit.load_state_dict(state_dict, strict=False)
    logger.info(
        "  Loaded %d keys | Missing: %d | Unexpected: %d",
        len(state_dict),
        len(missing),
        len(unexpected),
    )

    pipe.dit = dit.to(pipe.device)
    del dit, state_dict
    torch.cuda.empty_cache()

    logger.info("Pipeline ready.\n")
    return pipe


# ============================================================
# Inference
# ============================================================


def draw_input_overlay(
    frame: Image.Image,
    row: pd.Series,
    frame_idx: int,
    total_frames: int,
) -> Image.Image:
    """Draws PC-style input overlay (WASD + mouse) on a single frame.

    Mapping from the SCOPE action schema (which is gamepad-flavored) to PC:
      j_left  (left stick X/Y)  -> WASD movement keys (deadzone 0.3)
      j_right (right stick X/Y) -> mouse look delta arrow
      BUTTON_COLS               -> action chips top-left (Fire/Aim/Jump/...)
    Layout:
      top-left      action button chips
      bottom-left   WASD key cluster + frame counter
      bottom-right  mouse box with crosshair + delta arrow
    """
    img = frame.copy() if frame.mode == "RGBA" else frame.convert("RGBA")
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    pad = 8
    BTN_LABELS = {
        "right_trigger": "Fire",
        "left_trigger":  "Aim",
        "south":         "Jump",
        "right_thumb":   "Melee",
        "west":          "Reload",
        "north":         "Switch",
    }

    y = pad
    for name in BUTTON_COLS:
        pressed = float(row.get(name, 0.0)) > 0.5
        label = BTN_LABELS.get(name, name)
        fg = (60, 220, 60, 240) if pressed else (200, 200, 200, 140)
        bg = (0, 0, 0, 170) if pressed else (0, 0, 0, 90)
        d.rectangle([pad, y, pad + 80, y + 16], fill=bg)
        d.text((pad + 4, y + 1), label, fill=fg)
        y += 18

    try:
        lx, ly = float(row["j_left"][0]), float(row["j_left"][1])
    except (TypeError, IndexError, KeyError):
        lx, ly = 0.0, 0.0
    try:
        rx, ry = float(row["j_right"][0]), float(row["j_right"][1])
    except (TypeError, IndexError, KeyError):
        rx, ry = 0.0, 0.0
    DZ = 0.3
    wasd = {
        "W": ly < -DZ,
        "A": lx < -DZ,
        "S": ly >  DZ,
        "D": lx >  DZ,
    }

    key_sz = 26
    key_gap = 4
    row_w = 4 * key_sz + 3 * key_gap
    row_x = (w - row_w) // 2
    row_y = h - pad - key_sz - 16

    def draw_key(kx: int, ky: int, label: str, pressed: bool):
        fill = (60, 180, 60, 230) if pressed else (0, 0, 0, 150)
        outline = (180, 255, 180, 255) if pressed else (220, 220, 220, 200)
        d.rectangle([kx, ky, kx + key_sz, ky + key_sz],
                    fill=fill, outline=outline, width=2)
        d.text((kx + 8, ky + 6), label,
               fill=(255, 255, 255, 245) if pressed else (220, 220, 220, 190))

    for i, letter in enumerate("WASD"):
        kx = row_x + i * (key_sz + key_gap)
        draw_key(kx, row_y, letter, wasd[letter])

    counter_y = row_y + key_sz + 2
    d.rectangle([row_x, counter_y, row_x + row_w, counter_y + 12],
                fill=(0, 0, 0, 150))
    d.text((row_x + 4, counter_y - 1),
           f"frame {frame_idx + 1}/{total_frames}",
           fill=(255, 255, 255, 230))

    box_w = 70
    box_h = 50
    box_x = w - pad - box_w
    box_y = h - pad - box_h - 16
    d.rectangle([box_x, box_y, box_x + box_w, box_y + box_h],
                outline=(255, 255, 255, 180), width=2,
                fill=(0, 0, 0, 110))
    d.text((box_x + box_w - 38, box_y + box_h + 1), "mouse",
           fill=(255, 255, 255, 220))
    cx = box_x + box_w // 2
    cy = box_y + box_h // 2
    d.line([cx - 6, cy, cx + 6, cy], fill=(180, 180, 180, 200), width=1)
    d.line([cx, cy - 6, cx, cy + 6], fill=(180, 180, 180, 200), width=1)
    mag = (rx * rx + ry * ry) ** 0.5
    if mag > 0.05:
        scale_x = (box_w // 2) - 6
        scale_y = (box_h // 2) - 6
        tip_x = int(cx + max(-1.0, min(1.0, rx)) * scale_x)
        tip_y = int(cy + max(-1.0, min(1.0, ry)) * scale_y)
        d.line([cx, cy, tip_x, tip_y], fill=(255, 80, 80, 240), width=2)
        d.ellipse([tip_x - 3, tip_y - 3, tip_x + 3, tip_y + 3],
                  fill=(255, 80, 80, 240))

    return img.convert("RGB")


def generate_video(
    pipe: WanVideoPipeline,
    image_path: str,
    action_path: str,
    args: argparse.Namespace,
) -> str:
    """Generates a single video from an input image and action sequence.

    Args:
        pipe: Initialized SCOPE video generation pipeline.
        image_path: Path to the input image (first frame).
        action_path: Path to the action parquet file.
        args: Parsed command-line arguments (height, width, etc.).

    Returns:
        Path to the saved output video file.
    """
    image = crop_and_resize(
        Image.open(image_path).convert("RGB"),
        args.height,
        args.width,
    )
    num_frames = compute_num_frames(args.max_frames)
    keyboard, mouse = load_actions(action_path, num_frames)

    t0 = time.perf_counter()
    video = pipe(
        prompt=args.prompt,
        negative_prompt=NEGATIVE_PROMPT,
        input_image=image,
        num_frames=num_frames,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        seed=args.seed,
        tiled=True,
        keyboard_action=keyboard,
        mouse_action=mouse,
    )
    gen_s = time.perf_counter() - t0
    out_fps = 20  # save_video target fps
    gen_fps = num_frames / gen_s if gen_s > 0 else 0.0
    rtf = gen_fps / out_fps if out_fps else 0.0
    logger.info(
        "[scope] generated %d frames in %.2fs  -  gen %.2f fps  -  %.2fx real-time @ %d fps playback",
        num_frames, gen_s, gen_fps, rtf, out_fps,
    )

    if not getattr(args, "no_overlay", False):
        action_df = pd.read_parquet(action_path)
        if len(action_df) < num_frames:
            action_df = pd.concat(
                [action_df, pd.concat([action_df.iloc[[-1]]] *
                                       (num_frames - len(action_df)),
                                       ignore_index=True)],
                ignore_index=True,
            )
        video = [
            draw_input_overlay(f, action_df.iloc[i], i, num_frames)
            for i, f in enumerate(video)
        ]

    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(args.output_dir, f"{stem}.mp4")
    save_video(video, out_path, fps=out_fps, quality=5)
    return out_path


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for SCOPE inference.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="SCOPE: Action-Conditioned Game World Video Generation"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to SCOPE model directory (contains all weights).",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default=None,
        help="Single input image path (first frame).",
    )
    parser.add_argument(
        "--input_image_dir",
        type=str,
        default=None,
        help="Directory of images for batch inference.",
    )
    parser.add_argument(
        "--action_path",
        type=str,
        required=True,
        help="Action signal parquet file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt describing the scene.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for generated videos.",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--max_frames", type=int, default=81)
    parser.add_argument("--no_overlay", action="store_true",
                        help="Disable per-frame keyboard/joystick input overlay drawn on each output frame.")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Entry point for SCOPE inference."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    if not args.input_image and not args.input_image_dir:
        raise ValueError("Must specify --input_image or --input_image_dir.")
    os.makedirs(args.output_dir, exist_ok=True)

    pipe = init_pipeline(args.model_dir)

    # Collect input images.
    if args.input_image_dir:
        images = sorted([
            os.path.join(args.input_image_dir, f)
            for f in os.listdir(args.input_image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
    else:
        images = [args.input_image]

    logger.info("Generating %d video(s)...\n", len(images))

    for i, img_path in enumerate(images):
        logger.info("[%d/%d] %s", i + 1, len(images), os.path.basename(img_path))
        try:
            out = generate_video(pipe, img_path, args.action_path, args)
            logger.info("  -> %s", out)
        except RuntimeError as e:
            logger.error("  FAILED: %s", e)
        finally:
            torch.cuda.empty_cache()

    logger.info("Done.")


if __name__ == "__main__":
    main()
