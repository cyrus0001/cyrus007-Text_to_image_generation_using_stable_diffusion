

import torch
import gradio as gr
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    EulerDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DPMSolverMultistepScheduler
)
def load_pipeline(scheduler_name):
    model_id = "runwayml/stable-diffusion-v1-5"

    # Scheduler selection
    if scheduler_name == "Euler":
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    elif scheduler_name == "DDIM":
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    elif scheduler_name == "PNDM":
        scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
    elif scheduler_name == "LMS":
        scheduler = LMSDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    elif scheduler_name == "DPM":
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float32 
    )
    pipe = pipe.to("cuda") 
    return pipe

def generate_image(prompt, negative_prompt, height, width, steps, guidance, seed, scheduler_name):
    pipe = load_pipeline(scheduler_name)
    generator = torch.manual_seed(int(seed)) if seed else None

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=int(height),
        width=int(width),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator
    ).images[0]

    return image

with gr.Blocks() as interface:
    gr.Markdown("## 🖼️ Gen AI Text-to-Image Generator using Stable Diffusion with scheduler")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="a futuristic city under the sea")
            negative_prompt = gr.Textbox(label="Negative Prompt", value="blurry, low quality")
            scheduler_dropdown = gr.Dropdown(
                choices=["Euler", "DDIM", "PNDM", "LMS", "DPM"],
                value="Euler",
                label="Scheduler"
            )
            height = gr.Slider(256, 768, value=512, step=64, label="Height")
            width = gr.Slider(256, 768, value=512, step=64, label="Width")
            steps = gr.Slider(10, 100, value=40, step=5, label="Inference Steps")
            guidance = gr.Slider(1.0, 15.0, value=8.5, step=0.5, label="Guidance Scale")
            seed = gr.Textbox(label="Seed", value="1234")
            generate_btn = gr.Button("Generate Image")

        with gr.Column():
            output_image = gr.Image(label="Generated Image")

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, height, width, steps, guidance, seed, scheduler_dropdown],
        outputs=output_image
    )

interface.launch(share=True,debug = True)

