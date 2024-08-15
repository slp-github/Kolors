import os
import torch
import gradio as gr
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse

# from PIL import Image
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import (
    StableDiffusionXLPipeline,
)
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import EulerDiscreteScheduler


root_dir = os.path.dirname(os.path.abspath(__file__))
server_host = os.environ.get('KOLORS_HOST') or "http://127.0.0.1:8000"

# 创建FastAPI应用
fastapi_app = FastAPI()

# Initialize global variables for models and pipeline
text_encoder = None
tokenizer = None
vae = None
scheduler = None
unet = None
pipe = None


class InferItem(BaseModel):
    prompt: str
    use_random_seed: bool = True
    seed: Optional[str] = None
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    guidance_scale: int = 5
    num_images_per_prompt: int = 1


def load_models():
    global text_encoder, tokenizer, vae, scheduler, unet, pipe

    if text_encoder is None:
        ckpt_dir = f"{root_dir}/weights/Kolors"

        # Load the text encoder on CPU (this speeds stuff up 2x)
        text_encoder = (
            ChatGLMModel.from_pretrained(
                f"{ckpt_dir}/text_encoder", torch_dtype=torch.float16
            )
            .to("cpu")
            .half()
        )
        tokenizer = ChatGLMTokenizer.from_pretrained(f"{ckpt_dir}/text_encoder")

        # Load the VAE and UNet on GPU
        vae = (
            AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None)
            .half()
            .to("cuda")
        )
        scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
        unet = (
            UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None)
            .half()
            .to("cuda")
        )

        # Prepare the pipeline
        pipe = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            force_zeros_for_empty_prompt=False,
        )
        pipe = pipe.to("cuda")
        pipe.enable_model_cpu_offload()  # Enable offloading to balance CPU/GPU usage


def infer(
    prompt,
    use_random_seed,
    seed,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    num_images_per_prompt,
):
    load_models()

    if use_random_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    generator = torch.Generator(pipe.device).manual_seed(seed)
    images = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
    ).images

    saved_images = []
    output_dir = f"{root_dir}/scripts/outputs"
    os.makedirs(output_dir, exist_ok=True)

    for i, image in enumerate(images):
        file_path = os.path.join(output_dir, "sample_test.jpg")
        base_name, ext = os.path.splitext(file_path)
        counter = 1
        while os.path.exists(file_path):
            file_path = f"{base_name}_{counter}{ext}"
            counter += 1
        image.save(file_path)
        saved_images.append(file_path)

    return saved_images


@fastapi_app.get("/image/{file_path:path}")
async def get_image(file_path: str):
    # 构造完整的文件路径
    image_path = os.path.join(root_dir, file_path)

    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    # 返回图片文件
    return FileResponse(image_path)


@fastapi_app.post("/generate")
async def generate(item: InferItem):
    images = infer(
        prompt=item.prompt,
        use_random_seed=item.use_random_seed,
        seed=item.seed,
        height=item.height,
        width=item.width,
        num_inference_steps=item.num_inference_steps,
        guidance_scale=item.guidance_scale,
        num_images_per_prompt=item.num_images_per_prompt,
    )
    data = [image.replace(root_dir, server_host + "/image") for image in images]
    return {"code": 0, "msg": "success", "data": data}


def gradio_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Kolors: Diffusion Model Gradio Interface")
                prompt = gr.Textbox(label="Prompt")
                use_random_seed = gr.Checkbox(label="Use Random Seed", value=True)
                seed = gr.Slider(
                    minimum=0,
                    maximum=2**32 - 1,
                    step=1,
                    label="Seed",
                    randomize=True,
                    visible=False,
                )
                use_random_seed.change(
                    lambda x: gr.update(visible=not x), use_random_seed, seed
                )
                height = gr.Slider(
                    minimum=128, maximum=2048, step=64, label="Height", value=1024
                )
                width = gr.Slider(
                    minimum=128, maximum=2048, step=64, label="Width", value=1024
                )
                num_inference_steps = gr.Slider(
                    minimum=1, maximum=100, step=1, label="Inference Steps", value=50
                )
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    step=0.1,
                    label="Guidance Scale",
                    value=5.0,
                )
                num_images_per_prompt = gr.Slider(
                    minimum=1, maximum=10, step=1, label="Images per Prompt", value=1
                )
                btn = gr.Button("Generate Image")

            with gr.Column():
                output_images = gr.Gallery(
                    label="Output Images", elem_id="output_gallery"
                )

        btn.click(
            fn=infer,
            inputs=[
                prompt,
                use_random_seed,
                seed,
                height,
                width,
                num_inference_steps,
                guidance_scale,
                num_images_per_prompt,
            ],
            outputs=output_images,
        )

    return demo


if __name__ == "__main__":
    app = gr.mount_gradio_app(fastapi_app, gradio_interface(), path="/gr")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
