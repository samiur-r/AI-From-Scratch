# Diffusion Models Quick Reference

Diffusion Models is a generative modeling framework that creates high-quality images by learning to reverse a gradual noise-adding process, enabling state-of-the-art image generation, text-to-image synthesis, and creative applications.

### Installation
```bash
# Install TensorFlow and related packages
pip install tensorflow tensorflow-probability

# Install Keras CV (contains diffusion models)
pip install keras-cv

# Additional dependencies
pip install pillow numpy matplotlib requests

# For advanced features
pip install datasets wandb tensorboard
```

### Importing Diffusion Models
```python
# TensorFlow and Keras
import tensorflow as tf
import keras_cv

# Core components
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# For custom implementations
import tensorflow_probability as tfp
```

* * * * *

## 1. Text-to-Image Generation
```python
# Load Stable Diffusion model using Keras CV
model = keras_cv.models.StableDiffusion(
    img_width=512,
    img_height=512,
    jit_compile=True  # Enable XLA compilation for faster inference
)

# Generate image from text prompt
prompt = "a beautiful landscape with mountains and a lake at sunset"
image = model.text_to_image(
    prompt=prompt,
    batch_size=1,
    num_steps=50
)

# Convert to PIL and save
image_pil = keras_cv.utils.tensor_to_image(image[0])
image_pil.save("generated_landscape.png")
image_pil.show()

# Advanced generation with parameters
image = model.text_to_image(
    prompt=prompt,
    negative_prompt="blurry, low quality, distorted",
    num_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512,
    seed=42
)

# Display generated image
plt.figure(figsize=(8, 8))
plt.imshow(keras_cv.utils.tensor_to_image(image[0]))
plt.axis('off')
plt.title("Generated Image")
plt.show()
```

## 2. Image-to-Image Generation
```python
# Load initial image and preprocess
init_image = Image.open("input_image.jpg").resize((512, 512))
init_image_array = np.array(init_image)
init_image_tensor = tf.cast(init_image_array, tf.float32) / 255.0
init_image_tensor = tf.expand_dims(init_image_tensor, axis=0)

# Transform the image using Stable Diffusion
prompt = "a beautiful oil painting of a landscape"

# Image-to-image generation
image = model.text_to_image(
    prompt=prompt,
    num_steps=50,
    guidance_scale=7.5,
    seed=42
)

# For more control over strength, you can implement custom img2img
def image_to_image(model, init_image, prompt, strength=0.75, num_steps=50):
    # Add noise to initial image based on strength
    noise_level = int(strength * num_steps)

    # Generate latent representation
    encoded_image = model.image_encoder(init_image)

    # Add noise
    noise = tf.random.normal(tf.shape(encoded_image))
    noisy_image = encoded_image + noise * strength

    # Generate new image
    result = model.text_to_image(
        prompt=prompt,
        num_steps=num_steps - noise_level,
        guidance_scale=7.5,
        seed=42
    )

    return result

# Apply transformation
transformed_image = image_to_image(model, init_image_tensor, prompt, strength=0.75)

# Save result
result_pil = keras_cv.utils.tensor_to_image(transformed_image[0])
result_pil.save("transformed_image.png")
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
# Create a simple DDPM model for unconditional generation
class SimpleDDPM(tf.keras.Model):
    def __init__(self, image_size=64, timesteps=1000):
        super().__init__()
        self.image_size = image_size
        self.timesteps = timesteps

        # Simple U-Net architecture
        self.unet = self._build_unet()

    def _build_unet(self):
        inputs = tf.keras.layers.Input(shape=(self.image_size, self.image_size, 3))
        time_input = tf.keras.layers.Input(shape=())

        # Simple encoder-decoder for demonstration
        x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)

        # Time embedding
        t = tf.keras.layers.Dense(128, activation='relu')(time_input)
        t = tf.keras.layers.Reshape((1, 1, 128))(t)
        t = tf.tile(t, [1, self.image_size, self.image_size, 1])

        # Combine features and time
        x = tf.keras.layers.Concatenate()([x, t])

        # Decoder
        x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        outputs = tf.keras.layers.Conv2D(3, 3, padding='same')(x)

        return tf.keras.Model([inputs, time_input], outputs)

    def call(self, x, t):
        return self.unet([x, t])

# Create and use unconditional model
ddpm_model = SimpleDDPM(image_size=64, timesteps=1000)

# Generate random samples
def generate_samples(model, num_samples=4, img_size=64, timesteps=1000):
    # Start with random noise
    x = tf.random.normal((num_samples, img_size, img_size, 3))

    # Reverse diffusion process (simplified)
    for t in reversed(range(0, timesteps, timesteps // 50)):  # 50 steps
        t_batch = tf.fill((num_samples,), t)

        # Predict noise
        predicted_noise = model(x, t_batch)

        # Denoise (simplified step)
        alpha = 0.98  # Simplified noise schedule
        x = (x - predicted_noise * (1 - alpha)) / tf.sqrt(alpha)

        if t > 0:
            noise = tf.random.normal(tf.shape(x))
            x = x + noise * 0.1

    # Clip to valid range
    x = tf.clip_by_value((x + 1) / 2, 0, 1)

    return x

# Generate samples
generated_images = generate_samples(ddpm_model, num_samples=4)

# Display results
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for i in range(4):
    row, col = i // 2, i % 2
    axes[row, col].imshow(generated_images[i])
    axes[row, col].axis('off')
    axes[row, col].set_title(f'Sample {i+1}')

    # Save individual images
    img_pil = Image.fromarray((generated_images[i].numpy() * 255).astype(np.uint8))
    img_pil.save(f'unconditional_sample_{i}.png')

plt.tight_layout()
plt.show()
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
# Enable mixed precision for memory efficiency
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Batch generation
prompts = [
    "a red car in the mountains",
    "a blue ocean with waves",
    "a forest with tall trees",
    "a modern city skyline"
]

# Generate multiple images in batch
images = model.text_to_image(
    prompt=prompts,
    batch_size=len(prompts),
    num_steps=30,
    guidance_scale=7.5,
    seed=42
)

# Save generated images
for i, img_tensor in enumerate(images):
    img_pil = keras_cv.utils.tensor_to_image(img_tensor)
    img_pil.save(f"batch_image_{i}.png")

# Memory optimization with gradient checkpointing
@tf.function
def memory_efficient_generation(prompts, batch_size=2):
    """Generate images with memory optimization"""
    all_images = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        # Clear memory
        tf.keras.backend.clear_session()

        # Generate batch
        batch_images = model.text_to_image(
            prompt=batch_prompts,
            batch_size=len(batch_prompts),
            num_steps=30
        )

        all_images.extend(batch_images)

    return all_images

# Use memory-efficient generation for large batches
large_prompts = [f"image prompt {i}" for i in range(20)]
generated_images = memory_efficient_generation(large_prompts, batch_size=4)
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
def optimize_diffusion_model(model):
    """Optimize diffusion model for production deployment"""

    # Enable mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    # Compile model with XLA
    model.compile(jit_compile=True)

    # Create optimized inference function
    @tf.function(experimental_relax_shapes=True)
    def optimized_text_to_image(prompt, num_steps=30):
        return model.text_to_image(
            prompt=prompt,
            num_steps=num_steps,
            guidance_scale=7.5
        )

    return optimized_text_to_image

def benchmark_performance(model_fn, prompt, num_runs=5):
    """Benchmark model performance"""
    import time

    times = []
    for _ in range(num_runs):
        start_time = time.time()
        image = model_fn(prompt, num_steps=30)
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    print(f"Average generation time: {avg_time:.2f} seconds")
    print(f"Images per minute: {60/avg_time:.1f}")

    return avg_time

# Optimize and benchmark
optimized_model = optimize_diffusion_model(model)
avg_time = benchmark_performance(optimized_model, "a beautiful sunset over mountains")

# Convert to TensorFlow Lite for mobile deployment
def convert_to_tflite(model):
    """Convert model to TensorFlow Lite"""
    # This is a simplified example - actual conversion may require more steps
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    # Save the model
    with open('diffusion_model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("Model converted to TensorFlow Lite")

# Note: Full diffusion models are quite large for mobile deployment
# You might want to use a distilled or smaller version
# convert_to_tflite(model)

# Export to SavedModel format
tf.saved_model.save(model, 'diffusion_savedmodel')
print("Model saved in SavedModel format")
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
    image = model.text_to_image(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        num_steps=int(steps),
        guidance_scale=guidance,
        seed=42
    )

    # Convert tensor to PIL Image
    image_pil = keras_cv.utils.tensor_to_image(image[0])
    return image_pil

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
    title="TensorFlow Diffusion Model Image Generator"
)

# Launch interface
# iface.launch()

# Integration with Flask API
from flask import Flask, request, jsonify
import base64
from io import BytesIO

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_api():
    data = request.json
    prompt = data.get('prompt', '')
    steps = data.get('steps', 30)
    guidance_scale = data.get('guidance_scale', 7.5)

    # Generate image
    image_tensor = model.text_to_image(
        prompt=prompt,
        num_steps=steps,
        guidance_scale=guidance_scale,
        seed=42
    )

    # Convert to PIL and then to base64
    image_pil = keras_cv.utils.tensor_to_image(image_tensor[0])
    buffer = BytesIO()
    image_pil.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({
        'image': img_str,
        'prompt': prompt,
        'steps': steps,
        'guidance_scale': guidance_scale
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'framework': 'tensorflow'})

# Run API server
# app.run(host='0.0.0.0', port=5000)

# Integration with TensorFlow Serving
def prepare_for_serving(model, export_path):
    """Prepare model for TensorFlow Serving deployment"""

    @tf.function
    def serving_fn(prompt):
        return model.text_to_image(
            prompt=prompt,
            num_steps=30,
            guidance_scale=7.5
        )

    # Create concrete function
    concrete_fn = serving_fn.get_concrete_function(
        tf.TensorSpec(shape=[], dtype=tf.string)
    )

    # Save for serving
    tf.saved_model.save(
        model,
        export_path,
        signatures={'serving_default': concrete_fn}
    )

    print(f"Model saved for TensorFlow Serving at {export_path}")

# Example: prepare for serving
# prepare_for_serving(model, 'diffusion_serving_model')\n```

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