from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers.models import ControlNetModel
from diffusers.utils import load_image
from insightface.app import FaceAnalysis

from InstantID.pipeline_stable_diffusion_xl_instantid import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)

workspace = Path(__file__).resolve().parent
repo = workspace / "InstantID"
source_path = workspace / "images" / "ghulam-mujitaba-480x480.jpg"
output_path = workspace / "instantid_result.png"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# MPS can abort on mixed float16/float32 SDXL text operations; use float32 for stability.
dtype = torch.float32

if not source_path.exists():
    raise FileNotFoundError(source_path)
if not (repo / "checkpoints" / "ip-adapter.bin").exists():
    raise FileNotFoundError("InstantID/checkpoints/ip-adapter.bin")
if not (repo / "models" / "antelopev2").exists():
    raise FileNotFoundError("InstantID/models/antelopev2")

face_app = FaceAnalysis(
    name="antelopev2",
    root=str(repo),
    providers=["CPUExecutionProvider"],
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

face_image = load_image(str(source_path))
face_info = face_app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
if not face_info:
    raise RuntimeError("No face was detected in the source image.")
face_info = sorted(
    face_info,
    key=lambda item: (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]),
)[-1]
face_emb = face_info["embedding"]
face_kps = draw_kps(face_image, face_info["kps"])

controlnet = ControlNetModel.from_pretrained(
    str(repo / "checkpoints" / "ControlNetModel"),
    torch_dtype=dtype,
)
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=dtype,
)
pipe.to(device)
pipe.load_ip_adapter_instantid(str(repo / "checkpoints" / "ip-adapter.bin"))

image = pipe(
    prompt="a cinematic studio portrait, soft natural light, editorial photography, detailed",
    negative_prompt="low quality, blurry, distorted, deformed, watermark, text",
    image_embeds=face_emb,
    image=face_kps,
    controlnet_conditioning_scale=0.8,
    ip_adapter_scale=0.8,
    num_inference_steps=5,
    guidance_scale=5,
).images[0]
image.save(output_path)
print(f"InstantID device: {device}")
print(f"Saved result: {output_path}")