from diffusers import AutoPipelineForText2Image, StableDiffusionImg2ImgPipeline
from PIL import Image
import gradio as gr
import random
import torch
import math

css = """
.btn-green {
  background-image: linear-gradient(to bottom right, #6dd178, #00a613) !important;
  border-color: #22c55e !important;
  color: #166534 !important;
}
.btn-green:hover {
  background-image: linear-gradient(to bottom right, #6dd178, #6dd178) !important;
}
"""

def generate(prompt, turbo_steps, samp_steps, seed, progress=gr.Progress(track_tqdm=True)):
    print("prompt = ", prompt)
    if seed < 0:
        seed = random.randint(1,999999)
    image = txt2img(
        prompt,
        num_inference_steps=turbo_steps,
        guidance_scale=0.0,
        generator=torch.manual_seed(seed),
    ).images[0]
    upscaled_image = image.resize((1024,1024), 1)
    final_image = img2img(
        prompt,
        upscaled_image,
        num_inference_steps=samp_steps,
        guidance_scale=5,
        strength=1,
        generator=torch.manual_seed(seed),
    ).images[0]
    return [final_image], seed
        
def set_base_models():
    txt2img = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype = torch.float16,
        variant = "fp16"
    )
    txt2img.to("cuda")
    img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        "Lykon/dreamshaper-8",
        torch_dtype = torch.float16,
        variant = "fp16",
        safety_checker=None
    )
    img2img.to("cuda")
    return txt2img, img2img

with gr.Blocks(css=css) as demo:
    with gr.Column():
        prompt = gr.Textbox(label="Prompt")
        submit_btn = gr.Button("Generate", elem_classes="btn-green")
        
        with gr.Row():
            turbo_steps = gr.Slider(1, 4, value=1, step=1, label="Turbo steps")
            sampling_steps = gr.Slider(1, 6, value=3, step=1, label="Refiner steps")
            seed = gr.Number(label="Seed", value=-1, minimum=-1, precision=0)
            lastSeed = gr.Number(label="Last Seed", value=-1, interactive=False)
            
        gallery = gr.Gallery(show_label=False, preview=True, container=False, height=1100)
        
    submit_btn.click(generate, [prompt, turbo_steps, sampling_steps, seed], [gallery, lastSeed], queue=True)
    
txt2img, img2img = set_base_models()
demo.launch(debug=True)