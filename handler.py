#!/usr/bin/env python3
"""
RunPod serverless handler for Qwen-Image-Edit-2511 with real quantization support.

This is a fixed fork of the conalmullan/video-toolkit-qwen-edit worker. What was
broken there and is fixed here:

- `use_fp8` was ignored: the old handler always loaded the BF16 model (57.7GB),
  which OOMs on any GPU below 80GB. This handler really quantizes.
- No Lightning LoRA was loaded, so 4-step requests ran the base model (which
  needs ~50 steps), producing poor output. This handler loads the official
  Lightning LoRA with the required shift=3 scheduler and true_cfg_scale=1.0.
- No CPU offload: the 16.6GB BF16 text encoder stayed in VRAM. This handler
  enables model CPU offload so the encoder only touches the GPU while encoding.

Backends (input `backend`, default "torchao_fp8"):
- "torchao_fp8": torchao Float8WeightOnlyConfig via diffusers TorchAoConfig.
  True FP8 weights (~20.5GB). Native FP8 compute needs compute capability >= 8.9
  (RTX 4090 / Hopper). NOTE: unverified on 2511 - probe before trusting it.
- "gguf": unsloth GGUF transformer (default Q6_K, 17.95GB) via
  GGUFQuantizationConfig. Verified pattern (tomosud HF Space app).
  `gguf_quant` may be Q6_K (default), Q8_0 or Q4_K_M.

Legacy `use_fp8` bool is still accepted: true -> torchao_fp8, false -> gguf.
`backend` takes precedence when both are given.

Input format:
{
    "input": {
        "image_base64": str,           # Required - primary input image
        "images_base64": [str],        # Optional - up to 2 reference images
        "prompt": str,                 # Required - edit instruction
        "negative_prompt": str,        # Optional (default: "")
        "backend": str,                # Optional: "torchao_fp8" | "gguf"
        "use_fp8": bool,               # Optional legacy alias (default: true)
        "gguf_quant": str,             # Optional: Q6_K | Q8_0 | Q4_K_M
        "lightning_steps": int,        # Optional: 4 (default) or 8
        "seed": int,                   # Optional (random if not set)
        "num_inference_steps": int,    # Accepted but ignored: steps are fixed
                                       # by the Lightning LoRA (4 or 8).
        "guidance_scale": float,       # Accepted but ignored (Qwen ignores it).
        "auto_resize": bool,           # Optional (default: true)
        "r2": dict,                    # Optional - Cloudflare R2 upload config
    }
}

Output format:
{
    "success": true,
    "edited_image_base64": str,
    "seed": int,
    "inference_time_ms": int,
    "image_size": [width, height],
    "num_inference_steps": int,        # Actual steps used (4 or 8)
    "backend": str,                    # Backend actually used
    "use_fp8": bool,                   # True when backend is torchao_fp8
}
"""

import base64
import gc
import io
import math
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import runpod
import torch
from PIL import Image

# HuggingFace model IDs
MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
GGUF_REPO = "unsloth/Qwen-Image-Edit-2511-GGUF"

LORA_FILES = {
    4: "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    8: "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
}

GGUF_FILES = {
    "Q6_K": "qwen-image-edit-2511-Q6_K.gguf",  # 17.95GB (default)
    "Q8_0": "qwen-image-edit-2511-Q8_0.gguf",  # 22.35GB (tighter on 24GB)
    "Q4_K_M": "qwen-image-edit-2511-Q4_K_M.gguf",  # 12.84GB (lower quality)
}

# Lightning LoRA was distilled with shift=3 (ModelTC/Qwen-Image-Lightning).
LIGHTNING_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

# Longest image side allowed into the model. Larger inputs are downscaled to
# keep activation memory inside 24GB (and because edits rarely need 4K input).
MAX_IMAGE_DIM = 1024

# Lazy-loaded pipeline
_pipeline = None
_pipeline_config = {}


def log(message: str) -> None:
    """Log message to stderr (visible in RunPod logs)."""
    print(message, file=sys.stderr, flush=True)


def setup_hf_cache() -> None:
    """Set up HuggingFace cache - use baked-in cache from image."""
    baked_cache = Path("/root/.cache/huggingface")

    if baked_cache.exists():
        os.environ["HF_HOME"] = str(baked_cache)
        log(f"Using baked-in model cache: {baked_cache}")
    elif Path("/runpod-volume").exists() and os.access("/runpod-volume", os.W_OK):
        cache_path = Path("/runpod-volume/.cache/huggingface")
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_path)
        log(f"Using RunPod network volume cache: {cache_path}")
    else:
        log("WARNING: No cache found, will download models")


def get_gpu_vram_gb() -> int:
    """Detect GPU VRAM using PyTorch."""
    try:
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_id)
            vram_gb = props.total_memory // (1024 ** 3)
            log(f"Detected GPU: {props.name}, VRAM: {vram_gb}GB")
            return vram_gb
    except Exception as e:
        log(f"Warning: Could not detect GPU VRAM: {e}")
    return 24  # Default assumption


def log_free_disk() -> None:
    """Log free disk space (both backends download tens of GB)."""
    try:
        usage = shutil.disk_usage(os.environ.get("HF_HOME", "/root"))
        free_gb = usage.free / (1024 ** 3)
        log(f"Free disk: {free_gb:.1f}GB")
        if free_gb < 25:
            log("WARNING: low disk space - a backend download may fail. "
                "Using both backends on one worker needs ~95GB total.")
    except Exception as e:
        log(f"Warning: Could not check disk space: {e}")


def decode_base64_image(image_base64: str) -> Optional[Image.Image]:
    """Decode base64 string to PIL Image."""
    try:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        log(f"Error decoding base64 image: {e}")
        return None


def maybe_resize(image: Image.Image) -> Image.Image:
    """Downscale images whose longest side exceeds MAX_IMAGE_DIM.

    Keeps activation memory inside 24GB. Dimensions are rounded down to a
    multiple of 16 for the VAE.
    """
    w, h = image.size
    if max(w, h) <= MAX_IMAGE_DIM:
        return image
    scale = MAX_IMAGE_DIM / max(w, h)
    new_w = max(16, (int(w * scale) // 16) * 16)
    new_h = max(16, (int(h * scale) // 16) * 16)
    log(f"Resizing input {w}x{h} -> {new_w}x{new_h} (max dim {MAX_IMAGE_DIM})")
    return image.resize((new_w, new_h), Image.LANCZOS)


def encode_image_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_transformer_torchao_fp8():
    """Load the transformer with torchao FP8 weight-only quantization.

    No device_map: weights are quantized on CPU, then moved to GPU by the
    pipeline's CPU offload. Passing device_map="cuda" would hold both the
    original and quantized weights in VRAM during load and OOM.
    """
    from diffusers import QwenImageTransformer2DModel, TorchAoConfig
    from torchao.quantization import Float8WeightOnlyConfig

    log("Loading transformer with torchao FP8 weight-only quantization...")
    return QwenImageTransformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        quantization_config=TorchAoConfig(Float8WeightOnlyConfig()),
        torch_dtype=torch.bfloat16,
    )


def load_transformer_gguf(gguf_quant: str):
    """Load the transformer from an unsloth GGUF file."""
    from diffusers import GGUFQuantizationConfig, QwenImageTransformer2DModel
    from huggingface_hub import hf_hub_download

    filename = GGUF_FILES[gguf_quant]
    log(f"Downloading GGUF transformer {GGUF_REPO}/{filename}...")
    gguf_file = hf_hub_download(repo_id=GGUF_REPO, filename=filename)
    log(f"Loading GGUF transformer ({gguf_quant})...")
    return QwenImageTransformer2DModel.from_single_file(
        gguf_file,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
        config=MODEL_ID,
        subfolder="transformer",
    )


def get_pipeline(backend: str, gguf_quant: str = "Q6_K", lightning_steps: int = 4):
    """Get or initialize the diffusers pipeline (lazy loading)."""
    global _pipeline, _pipeline_config

    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline

    current_config = {"backend": backend, "lightning_steps": lightning_steps}
    if backend == "gguf":
        current_config["gguf_quant"] = gguf_quant

    if _pipeline is not None and _pipeline_config == current_config:
        return _pipeline

    # Need to reinitialize
    if _pipeline is not None:
        log("Pipeline config changed, reinitializing...")
        del _pipeline
        _pipeline = None
        gc.collect()
        torch.cuda.empty_cache()

    log_free_disk()
    log(f"Loading pipeline: backend={backend}, lightning_steps={lightning_steps}"
        + (f", gguf_quant={gguf_quant}" if backend == "gguf" else ""))
    start = time.time()

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        LIGHTNING_SCHEDULER_CONFIG
    )

    try:
        if backend == "torchao_fp8":
            transformer = load_transformer_torchao_fp8()
        elif backend == "gguf":
            transformer = load_transformer_gguf(gguf_quant)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Passing the transformer in skips re-downloading the transformer
        # subfolder; only text encoder + VAE + configs come from the repo.
        _pipeline = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
        )
    except Exception as e:
        import traceback

        log(f"Pipeline loading error: {e}")
        log(f"Full traceback:\n{traceback.format_exc()}")
        raise

    # Text encoder (BF16 16.6GB) must not stay resident: offload it to CPU so
    # only the transformer + VAE + LoRA occupy VRAM during diffusion.
    # Order matters: offload first, then load LoRA (tomosud app pattern).
    _pipeline.enable_model_cpu_offload()

    lora_file = LORA_FILES[lightning_steps]
    log(f"Loading Lightning LoRA ({lightning_steps} steps): {lora_file}")
    _pipeline.load_lora_weights(
        LORA_REPO, weight_name=lora_file, adapter_name="lightning"
    )
    _pipeline.set_adapters(["lightning"], adapter_weights=[1.0])

    _pipeline_config = current_config
    log(f"Pipeline loaded in {time.time() - start:.1f}s")
    return _pipeline


def upload_to_r2(image_base64: str, job_id: str, r2_config: dict):
    """Upload image to Cloudflare R2 and return (presigned_url, object_key)."""
    try:
        import uuid

        import boto3
        from botocore.config import Config

        log("Uploading to R2...")
        client = boto3.client(
            "s3",
            endpoint_url=r2_config["endpoint_url"],
            aws_access_key_id=r2_config["access_key_id"],
            aws_secret_access_key=r2_config["secret_access_key"],
            config=Config(signature_version="s3v4"),
        )
        object_key = f"qwen-edit/results/{job_id}_{uuid.uuid4().hex[:8]}.png"
        image_bytes = base64.b64decode(image_base64)
        client.put_object(
            Bucket=r2_config["bucket_name"],
            Key=object_key,
            Body=image_bytes,
            ContentType="image/png",
        )
        presigned_url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": r2_config["bucket_name"], "Key": object_key},
            ExpiresIn=7200,
        )
        log(f"  R2 upload complete: {object_key}")
        return presigned_url, object_key
    except Exception as e:
        log(f"Error uploading to R2: {e}")
        return None, None


def resolve_backend(job_input: dict):
    """Pick the backend. Returns (backend, error_message_or_None)."""
    backend = job_input.get("backend")
    if backend is not None:
        if backend not in ("torchao_fp8", "gguf"):
            return None, (
                f"Unknown backend '{backend}'. "
                "Supported: torchao_fp8, gguf"
            )
        return backend, None
    # Legacy flag: use_fp8=true -> torchao_fp8, false -> gguf (BF16 would OOM).
    return ("torchao_fp8" if job_input.get("use_fp8", True) else "gguf"), None


def handle_edit(job_input: dict, job_id: str, work_dir: Path) -> dict:
    """Handle image edit operation using Qwen-Image-Edit-2511."""
    start_time = time.time()

    image_base64 = job_input.get("image_base64")
    images_base64 = job_input.get("images_base64", [])
    prompt = job_input.get("prompt")
    negative_prompt = job_input.get("negative_prompt", "")

    backend, err = resolve_backend(job_input)
    if err:
        return {"error": err}

    gguf_quant = job_input.get("gguf_quant", "Q6_K")
    if backend == "gguf" and gguf_quant not in GGUF_FILES:
        return {"error": f"Unknown gguf_quant '{gguf_quant}'. "
                         f"Supported: {', '.join(GGUF_FILES)}"}

    lightning_steps = job_input.get("lightning_steps", 4)
    if lightning_steps not in LORA_FILES:
        return {"error": f"lightning_steps must be 4 or 8, got {lightning_steps}"}

    seed = job_input.get("seed")
    auto_resize = job_input.get("auto_resize", True)
    r2_config = job_input.get("r2")

    # Steps are fixed by the Lightning LoRA; a requested step count is ignored.
    requested_steps = job_input.get("num_inference_steps")
    if requested_steps is not None and requested_steps != lightning_steps:
        log(f"Ignoring num_inference_steps={requested_steps}: Lightning LoRA "
            f"fixes steps at {lightning_steps}")

    if not image_base64:
        return {"error": "Missing required 'image_base64' in input"}
    if not prompt:
        return {"error": "Missing required 'prompt' in input"}

    input_image = decode_base64_image(image_base64)
    if input_image is None:
        return {"error": "Failed to decode input image from base64"}
    log(f"Primary image size: {input_image.size}")
    if auto_resize:
        input_image = maybe_resize(input_image)

    all_images = [input_image]
    for i, ref_b64 in enumerate(images_base64[:2]):  # Max 2 additional images
        ref_image = decode_base64_image(ref_b64)
        if ref_image is None:
            return {"error": f"Failed to decode reference image {i + 2} from base64"}
        if auto_resize:
            ref_image = maybe_resize(ref_image)
        all_images.append(ref_image)
        log(f"Reference image {i + 2} size: {ref_image.size}")
    log(f"Total images for edit: {len(all_images)}")

    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    log(f"Using seed: {seed}")

    pipe = get_pipeline(backend, gguf_quant, lightning_steps)

    log(f"Running edit: backend={backend}, steps={lightning_steps}, "
        f"true_cfg_scale=1.0")
    gen_start = time.time()
    try:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        output = pipe(
            image=all_images,
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else " ",
            num_inference_steps=lightning_steps,
            true_cfg_scale=1.0,  # Lightning LoRA requires low CFG
            generator=generator,
            num_images_per_prompt=1,
        )
        output_image = output.images[0]
    except Exception as e:
        import traceback

        log(f"Generation error: {e}")
        log(f"Traceback:\n{traceback.format_exc()}")
        raise

    gen_time = time.time() - gen_start
    log(f"Generation completed in {gen_time:.1f}s")

    output_base64 = encode_image_base64(output_image)
    elapsed_ms = int((time.time() - start_time) * 1000)
    result = {
        "success": True,
        "edited_image_base64": output_base64,
        "seed": seed,
        "inference_time_ms": elapsed_ms,
        "image_size": list(output_image.size),
        "num_inference_steps": lightning_steps,
        "backend": backend,
        "use_fp8": backend == "torchao_fp8",
    }

    if r2_config:
        url, r2_key = upload_to_r2(output_base64, job_id, r2_config)
        if url:
            result["output_url"] = url
            result["r2_key"] = r2_key

    return result


def handler(job: dict) -> dict:
    """Main RunPod handler - routes to edit operation."""
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})
    operation = job_input.get("operation", "edit")
    log(f"Job {job_id}: operation={operation}")

    work_dir = Path(tempfile.mkdtemp(prefix=f"runpod_{job_id}_"))
    log(f"Working directory: {work_dir}")

    try:
        if operation == "edit":
            return handle_edit(job_input, job_id, work_dir)
        else:
            return {"error": f"Unknown operation: {operation}. Supported: edit"}
    except torch.cuda.OutOfMemoryError as e:
        log(f"CUDA OOM: {e}")
        torch.cuda.empty_cache()
        return {"error": "GPU out of memory. Try backend='gguf' "
                         "(smaller, 17.95GB) or a smaller image."}
    except Exception as e:
        import traceback

        log(f"Handler exception: {e}")
        log(traceback.format_exc())
        return {"error": f"Internal error: {str(e)}"}
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
            log("Cleaned up working directory")
        except Exception:
            pass


# RunPod serverless entry point
if __name__ == "__main__":
    log("Starting RunPod Qwen-Edit handler (FP8/GGUF fixed fork)...")

    if torch.cuda.is_available():
        log(f"CUDA available: {torch.cuda.get_device_name(0)}")
        vram_gb = get_gpu_vram_gb()
        log(f"VRAM: {vram_gb}GB")
    else:
        log("WARNING: CUDA not available!")

    setup_hf_cache()

    # Pre-load the default backend at startup so the first job is fast.
    # WARMUP_BACKEND=none skips this (first job loads lazily, needs a long
    # job timeout). Warm-up failure is non-fatal: the first job retries.
    warmup_backend = os.environ.get("WARMUP_BACKEND", "torchao_fp8")
    if warmup_backend.lower() != "none":
        log(f"Pre-loading pipeline at startup (backend={warmup_backend})...")
        try:
            get_pipeline(warmup_backend)
            log("Pipeline pre-loaded successfully")
        except Exception as e:
            log(f"Warning: Pipeline pre-load failed: {e}")
            log("Will retry on first request")

    runpod.serverless.start({"handler": handler})
