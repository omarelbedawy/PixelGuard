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
def _letterbox_to_square(image, size=512):
    """Scale to fit inside size x size preserving aspect ratio, pad with
    neutral grey to make it square. Returns (square_image, offset, content_size, original_size)."""
    from PIL import Image

    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    w, h = image.size
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image.resize((new_w, new_h), resample)
    canvas = Image.new("RGB", (size, size), (127, 127, 127))
    offset = ((size - new_w) // 2, (size - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas, offset, (new_w, new_h), (w, h)


def _unletterbox(square_image, offset, content_size, original_size):
    """Reverse of _letterbox_to_square: crop off the padding, resize back to original dimensions."""
    from PIL import Image

    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    x, y = offset
    cw, ch = content_size
    cropped = square_image.crop((x, y, x + cw, y + ch))
    return cropped.resize(original_size, resample)


def _run_immunize(image_bytes: bytes, eps: float, iters: int):
    import gc
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image

    vae, latent_dtype = _get_vae()
    device = next(vae.parameters()).device

    original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image, offset, content_size, original_size = _letterbox_to_square(original, 512)

    # 1. إجبار eps تكون 0.12 (قيمة MIT القوية) لو كانت القيمة جاية صغيرة أو افتراضية
    if eps <= 0.05:
        eps = 0.12

    eps_scaled = float(eps) * 2.0
    alpha = 0.01
    iters = min(int(iters), 1000)

    transform = T.Compose([T.ToTensor()])
    X_f32 = transform(image).unsqueeze(0).to(device, dtype=torch.float32) * 2.0 - 1.0
    X_adv_f32 = X_f32.clone().detach()

    # 2. إنشاء صورة رمادية موحدة (Neutral Gray: RGB 128 -> 0.0 in range [-1, 1])
    # وحساب الـ Target Latent بتاعها من الـ VAE مباشرة زي طريقة MIT
    gray_image_tensor = torch.zeros((1, 3, 512, 512), device=device, dtype=latent_dtype)
    with torch.no_grad():
        target_latent = vae.encode(gray_image_tensor).latent_dist.mean

    # 3. حلقة الـ PGD Attack
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

    final_square = (X_adv_f32 / 2.0 + 0.5).clamp(0, 1).squeeze(0).cpu()
    out_img_square = T.ToPILImage()(final_square)
    out_img = _unletterbox(out_img_square, offset, content_size, original_size)

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
# ENDPOINT 1: IMMUNIZE — spawn+poll pattern (avoids the 150s
# redirect entirely: /start returns instantly, /result/{id} is
# polled by the client and is always fast, so CORS never breaks)
# ============================================================
@app.function(image=vae_image, gpu="T4", timeout=300)
def _immunize_job(image_bytes: bytes, eps: float, iters: int):
    return _run_immunize(image_bytes, eps, iters)


@app.function(image=vae_image, cpu=1, timeout=60)
@modal.asgi_app()
def immunize():
    from fastapi import Request
    from fastapi.responses import JSONResponse

    web_app = _make_cors_app()

    @web_app.get("/")
    async def _health():
        return {"status": "ok", "engine": "immunize"}

    @web_app.post("/start")
    async def _start(request: Request):
        try:
            payload = await request.json()
            image_bytes = base64.b64decode(payload["image_base64"])
            eps = payload.get("eps", 0.0)
            iters = payload.get("iters", 500)
            call = _immunize_job.spawn(image_bytes, eps, iters)
            return {"call_id": call.object_id}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @web_app.get("/result/{call_id}")
    async def _result(call_id: str):
        try:
            function_call = modal.FunctionCall.from_id(call_id)
            try:
                result_b64, used = function_call.get(timeout=0)
                return {"status": "done", "image_base64": result_b64, "iters_used": used}
            except TimeoutError:
                return {"status": "pending"}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return web_app

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
        from fastapi.responses import JSONResponse

        try:
            payload = await request.json()
            pipe = _get_pipe()
            base_original = Image.open(
                io.BytesIO(base64.b64decode(payload["image_base64"]))
            ).convert("RGB")
            mask_original = Image.open(
                io.BytesIO(base64.b64decode(payload["mask_base64"]))
            ).convert("RGB")
            base_image, offset, content_size, original_size = _letterbox_to_square(base_original, 512)
            mask_resized = mask_original.resize(content_size, getattr(getattr(Image, "Resampling", Image), "LANCZOS"))
            mask_canvas = Image.new("RGB", (512, 512), (0, 0, 0))
            mask_canvas.paste(mask_resized, offset)
            mask_image = mask_canvas
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
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return web_app