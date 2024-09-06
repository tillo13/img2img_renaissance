import os
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# Configuration for single image processing
IMAGE_PATH = 'dog.png'
OUTPUT_DIR = './output'
PROMPT = "put a cat sitting next to the dog"

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the model
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Load the image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

image = load_image(IMAGE_PATH)

# Generate the modified image
num_inference_steps = 50
image_guidance_scale = 1.0
text_guidance_scale = 7.5

generated_images = pipe(prompt=PROMPT, image=image, num_inference_steps=num_inference_steps,
                        image_guidance_scale=image_guidance_scale, text_guidance_scale=text_guidance_scale).images

# Save the generated image
output_filepath = os.path.join(OUTPUT_DIR, "transformed_dog.png")
generated_images[0].save(output_filepath)

print(f"Processed image saved to: {output_filepath}")