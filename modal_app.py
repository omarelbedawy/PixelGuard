"""
PixelGuard — Modal Serverless API (v2: GPU immunize + CORS enabled)
=====================================================================
Design:
- Model weights are downloaded ONCE at image build time (.run_function()),
  baked into the image. No network download happens at request time,
  even after a cold start.
- Both endpoints run on T4 GPU and scale to zero when idle — free when
  not in use, wakes up in seconds when called.
- CORS is enabled explicitly via FastAPI's CORSMiddleware. Without this,
  Postman/curl work fine but browser fetch() calls get silently blocked.
- The immunize loop uses the SAME fp16/fp32 precision split as the
  original Colab script: the VAE forward pass runs in fp16 (fast), but
  gradients are converted back to fp32 before the update step to avoid
  underflow. This preserves the exact original attack quality while
  running ~1.5-2x faster on GPU.

Deploy:
  pip install modal
  modal setup
  modal deploy modal_app.py
"""

import base64
import io

import modal

app = modal.App("pixelguard")

MODEL_REPO = "runwayml/stable-diffusion-inpainting"


# ============================================================
# BUILD-TIME WEIGHT DOWNLOAD (runs once, at `modal deploy`)
# ============================================================
def _download_vae():
    from diffusers import AutoencoderKL

    AutoencoderKL.from_pretrained(MODEL_REPO, subfolder="vae")


def _download_full_pipeline():
    from diffusers import StableDiffusionInpaintPipeline

    StableDiffusionInpaintPipeline.from_pretrained(MODEL_REPO)


vae_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "diffusers>=0.27.0", "transformers",
        "accelerate", "pillow", "fastapi[standard]",
    )
    .run_function(_download_vae)
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "diffusers>=0.27.0", "transformers",
        "accelerate", "pillow", "fastapi[standard]",
    )
    .run_function(_download_full_pipeline)
)


# ============================================================
# SHARED MODEL LOADERS (loaded once per warm container)
# ============================================================
_vae = None
_vae_dtype = None


def _get_vae():
    global _vae, _vae_dtype
    if _vae is None:
        import torch
        from diffusers import AutoencoderKL

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # fp16 on GPU for speed, fp32 fallback on CPU
        _vae_dtype = torch.float16 if device == "cuda" else torch.float32
        _vae = AutoencoderKL.from_pretrained(
            MODEL_REPO, subfolder="vae", torch_dtype=_vae_dtype
        ).to(device)
        _vae.eval()
        for p in _vae.parameters():
            p.requires_grad = False
    return _vae, _vae_dtype


_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        _pipe = StableDiffusionInpaintPipeline.from_pretrained(
            MODEL_REPO, torch_dtype=torch.float16
        ).to("cuda")
        _pipe.enable_attention_slicing()
    return _pipe


# ============================================================
# CORE PGD LOGIC — same precision-split trick as the Colab script
# ============================================================
def _run_immunize(image_bytes: bytes, eps: float, iters: int):
    import gc

    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image

    vae, latent_dtype = _get_vae()
    device = next(vae.parameters()).device

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((512, 512))
    eps_scaled = float(eps) * 2.0
    alpha = 0.01
    iters = min(int(iters), 1000)

    transform = T.Compose([T.ToTensor()])
    X_f32 = transform(image).unsqueeze(0).to(device, dtype=torch.float32) * 2.0 - 1.0
    X_adv_f32 = X_f32.clone().detach()
    target_latent = torch.zeros((1, 4, 64, 64), device=device, dtype=latent_dtype)

    for _ in range(iters):
        # fp16 copy ONLY for the VAE forward pass (fast on GPU tensor cores)
        X_adv_f16 = X_adv_f32.to(latent_dtype).detach()
        X_adv_f16.requires_grad = True

        latent = vae.encode(X_adv_f16).latent_dist.mean
        loss = nn.MSELoss()(latent, target_latent)
        grad = torch.autograd.grad(loss, [X_adv_f16])[0]

        # back to fp32 for the actual update math (avoids underflow)
        grad_f32 = grad.to(torch.float32)

        with torch.no_grad():
            X_adv_f32 = X_adv_f32 - alpha * grad_f32.sign()
            X_adv_f32 = torch.min(torch.max(X_adv_f32, X_f32 - eps_scaled), X_f32 + eps_scaled)
            X_adv_f32 = torch.clamp(X_adv_f32, -1.0, 1.0).detach()

    final = (X_adv_f32 / 2.0 + 0.5).clamp(0, 1).squeeze(0).cpu()
    out_img = T.ToPILImage()(final)

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    result_b64 = base64.b64encode(buf.getvalue()).decode()

    del X_adv_f32, X_f32, target_latent
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result_b64, iters


def _make_cors_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI()
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return web_app


# ============================================================
# ENDPOINT 1: IMMUNIZE — now on GPU, CORS-enabled
# ============================================================
@app.function(image=vae_image, gpu="T4", timeout=300)
@modal.asgi_app()
def immunize():
    from fastapi import Request

    web_app = _make_cors_app()

    @web_app.post("/")
    async def _immunize(request: Request):
        payload = await request.json()
        image_bytes = base64.b64decode(payload["image_base64"])
        eps = payload.get("eps", 0.02)
        iters = payload.get("iters", 500)
        result_b64, used = _run_immunize(image_bytes, eps, iters)
        return {"image_base64": result_b64, "iters_used": used}

    return web_app


# ============================================================
# ENDPOINT 2: TEST INPAINTING — GPU T4, CORS-enabled
# ============================================================
@app.function(image=gpu_image, gpu="T4", timeout=300)
@modal.asgi_app()
def test_inpaint():
    from fastapi import Request

    web_app = _make_cors_app()

    @web_app.post("/")
    async def _test_inpaint(request: Request):
        import torch
        from PIL import Image

        payload = await request.json()
        pipe = _get_pipe()
        base_image = Image.open(
            io.BytesIO(base64.b64decode(payload["image_base64"]))
        ).convert("RGB").resize((512, 512))
        mask_image = Image.open(
            io.BytesIO(base64.b64decode(payload["mask_base64"]))
        ).convert("RGB").resize((512, 512))
        prompt = payload.get("prompt", "")

        generator = torch.Generator(device="cuda").manual_seed(42)
        result = pipe(
            prompt=prompt,
            image=base_image,
            mask_image=mask_image,
            num_inference_steps=20,
            generator=generator,
        ).images[0]

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        result_b64 = base64.b64encode(buf.getvalue()).decode()
        return {"image_base64": result_b64}

    return web_app
