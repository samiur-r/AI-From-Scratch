# Diffusion Models Quick Reference

Diffusion Models is a generative modeling framework that creates high-quality images by learning to reverse a gradual noise-adding process, enabling state-of-the-art image generation, text-to-image synthesis, and creative applications.

### Installation
```bash
# Install diffusers (Hugging Face)
pip install diffusers transformers accelerate

# Install Stable Diffusion specific
pip install diffusers[torch] transformers

# Additional dependencies
pip install torch torchvision xformers

# For training and advanced features
pip install datasets wandb tensorboard
```

### Importing Diffusion Models
```python
# Hugging Face Diffusers
from diffusers import StableDiffusionPipeline, DDPMPipeline, DDIMScheduler
import torch

# Core components
from diffusers import UNet2DModel, DDPMScheduler
from PIL import Image

# For custom implementations
import numpy as np
import matplotlib.pyplot as plt
```

* * * * *

## 1. Text-to-Image Generation
```python
# Load Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Generate image from text prompt
prompt = "a beautiful landscape with mountains and a lake at sunset"
image = pipe(prompt).images[0]

# Save or display
image.save("generated_landscape.png")
image.show()

# Advanced generation with parameters
image = pipe(
    prompt=prompt,
    negative_prompt="blurry, low quality, distorted",
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512,
    generator=torch.Generator().manual_seed(42)
).images[0]
```

## 2. Image-to-Image Generation
```python
from diffusers import StableDiffusionImg2ImgPipeline

# Load image-to-image pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Load initial image
init_image = Image.open("input_image.jpg").resize((512, 512))

# Transform the image
prompt = "a beautiful oil painting of a landscape"
image = pipe(
    prompt=prompt,
    image=init_image,
    strength=0.75,  # How much to transform (0.0 to 1.0)
    guidance_scale=7.5,
    num_inference_steps=50
).images[0]

image.save("transformed_image.png")
```

## 3. Image Inpainting
```python
from diffusers import StableDiffusionInpaintPipeline

# Load inpainting pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Load image and mask
image = Image.open("image_with_object_to_remove.jpg")
mask = Image.open("mask_of_area_to_inpaint.jpg")  # White = inpaint, Black = keep

# Inpaint the masked area
prompt = "a beautiful garden"
result = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

result.save("inpainted_image.png")
```

## 4. Unconditional Image Generation
```python
# Load unconditional generation pipeline
pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

# Generate random samples
images = pipe(batch_size=4).images

# Display results
for i, img in enumerate(images):
    img.save(f"unconditional_sample_{i}.png")
```

## 5. Custom Scheduler Usage
```python
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler

# Replace default scheduler with faster one
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# Use DDIM for faster generation
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Or use DPM-Solver for even faster generation
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe = pipe.to("cuda")

# Generate with fewer steps
image = pipe(
    "a futuristic cityscape",
    num_inference_steps=25  # Reduced from typical 50
).images[0]
```

## 6. Batch Generation and Memory Optimization
```python
# Enable memory efficient attention
pipe.enable_attention_slicing()

# Enable CPU offloading for large models
pipe.enable_sequential_cpu_offload()

# Batch generation
prompts = [
    "a red car in the mountains",
    "a blue ocean with waves",
    "a forest with tall trees",
    "a modern city skyline"
]

# Generate multiple images
images = pipe(
    prompts,
    num_inference_steps=30,
    guidance_scale=7.5
).images

for i, img in enumerate(images):
    img.save(f"batch_image_{i}.png")
```

## 7. Fine-tuning Diffusion Models
```python
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from datasets import load_dataset
import torch.nn.functional as F

def train_diffusion_model(dataset_name, output_dir, num_epochs=100):
    # Load dataset
    dataset = load_dataset(dataset_name, split="train")

    # Initialize model and scheduler
    model = UNet2DModel(
        sample_size=64,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
            "UpBlock2D", "UpBlock2D", "UpBlock2D"
        ),
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(num_epochs):
        for batch in dataset:
            clean_images = batch["images"]

            # Sample noise and timesteps
            noise = torch.randn(clean_images.shape)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (clean_images.shape[0],)
            )

            # Add noise to images
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Predict noise
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # Save model
    model.save_pretrained(output_dir)

# Example usage (commented out)
# train_diffusion_model("huggan/flowers-102-categories", "./my-diffusion-model")
```

## 8. Advanced Prompting Techniques
```python
def advanced_prompting(pipe, base_prompt):
    """Advanced techniques for better prompt engineering"""

    # Prompt weighting (increase/decrease importance)
    weighted_prompt = f"({base_prompt}:1.2), highly detailed, masterpiece"

    # Negative prompting (what to avoid)
    negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy"

    # Style modifiers
    style_prompt = f"{base_prompt}, in the style of Van Gogh, oil painting"

    # Quality boosters
    quality_prompt = f"{base_prompt}, 4k, ultra high resolution, photorealistic"

    results = {}

    # Generate with different prompt strategies
    for name, prompt in [
        ("base", base_prompt),
        ("weighted", weighted_prompt),
        ("styled", style_prompt),
        ("quality", quality_prompt)
    ]:
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=torch.Generator().manual_seed(42)
        ).images[0]

        results[name] = image
        image.save(f"prompt_{name}.png")

    return results

# Example usage
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

results = advanced_prompting(pipe, "a majestic dragon flying over a castle")
```

## 9. ControlNet for Controlled Generation
```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import cv2

# Load ControlNet model (Canny edge detection)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Process control image (edge detection)
image = cv2.imread("input_image.jpg")
canny = cv2.Canny(image, 100, 200)
canny_image = Image.fromarray(canny)

# Generate controlled image
prompt = "a beautiful landscape painting"
result = pipe(
    prompt=prompt,
    image=canny_image,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

result.save("controlled_generation.png")
```

## 10. Model Optimization and Deployment
```python
def optimize_diffusion_model(pipe):
    """Optimize diffusion model for production deployment"""

    # Enable various optimizations
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    # Use half precision
    pipe = pipe.to(dtype=torch.float16)

    # Compile model for faster inference (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")

    # Enable CPU offloading for memory efficiency
    pipe.enable_sequential_cpu_offload()

    return pipe

def benchmark_performance(pipe, prompt, num_runs=5):
    """Benchmark model performance"""
    import time

    times = []
    for _ in range(num_runs):
        start_time = time.time()
        image = pipe(prompt, num_inference_steps=30).images[0]
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    print(f"Average generation time: {avg_time:.2f} seconds")
    print(f"Images per minute: {60/avg_time:.1f}")

    return avg_time

# Optimize and benchmark
pipe = optimize_diffusion_model(pipe)
benchmark_performance(pipe, "a beautiful sunset over mountains")
```

## 11. Custom Pipeline Creation
```python
from diffusers import DiffusionPipeline
import torch

class CustomDiffusionPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler, vae, text_encoder, tokenizer):
        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer
        )

    @torch.no_grad()
    def __call__(self, prompt, num_inference_steps=50, guidance_scale=7.5):
        # Encode text
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Create unconditional embeddings
        uncond_input = self.tokenizer(
            "",
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Combine embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Initialize latents
        latents = torch.randn((1, 4, 64, 64), device=self.device)

        # Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Denoising loop
        for t in self.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Update latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents
        with torch.no_grad():
            image = self.vae.decode(latents / 0.18215).sample

        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)[0]

        return Image.fromarray(image)

# Example: Load and use custom pipeline
# custom_pipe = CustomDiffusionPipeline.from_pretrained("path/to/custom/model")
# image = custom_pipe("a custom generated image")
```

## 12. Integration with Other Tools
```python
# Integration with Gradio for web interface
import gradio as gr

def generate_image_gradio(prompt, negative_prompt="", steps=50, guidance=7.5):
    """Gradio interface for image generation"""
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=guidance
    ).images[0]
    return image

# Create Gradio interface
iface = gr.Interface(
    fn=generate_image_gradio,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
        gr.Textbox(label="Negative Prompt", placeholder="What to avoid..."),
        gr.Slider(10, 100, value=50, label="Steps"),
        gr.Slider(1, 20, value=7.5, label="Guidance Scale")
    ],
    outputs=gr.Image(type="pil"),
    title="Diffusion Model Image Generator"
)

# Launch interface
# iface.launch()

# Integration with API
from flask import Flask, request, jsonify
import base64
from io import BytesIO

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_api():
    data = request.json
    prompt = data.get('prompt', '')

    image = pipe(prompt).images[0]

    # Convert to base64
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({'image': img_str})

# Run API server
# app.run(host='0.0.0.0', port=5000)
```

* * * * *

Summary
=======

- **Text-to-image generation** create images from natural language descriptions
- **Multiple modalities** support image-to-image, inpainting, and unconditional generation
- **Flexible schedulers** DDPM, DDIM, DPM-Solver for different speed-quality trade-offs
- **Pre-trained models** Stable Diffusion and other models available via Hugging Face
- **Production optimizations** memory efficiency, batch processing, and model compilation
- **Extensible framework** custom pipelines, ControlNet integration, and fine-tuning support
- **Community ecosystem** large collection of models, schedulers, and applications