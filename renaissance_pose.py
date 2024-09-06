import os
import requests
import torch
from PIL import Image, ImageOps, ImageFilter
from io import BytesIO
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import time
import gc
import numpy as np
from datetime import datetime
from enhance_image import enhance_image  # Only import what is needed

### GLOBAL VARIABLES ###

# Image processing parameters
BASE_RESOLUTION = 512  # Base size for resizing the shortest dimension while maintaining aspect ratio
UPSCALING_ENABLED = True  # Set to True if upscaling to higher resolution before and after processing
TARGET_RESOLUTION = 1024  # Target resolution for resizing after processing

NUM_INFERENCE_STEPS = 50  # Number of inference steps for quality

# Model configuration parameters
IMAGE_GUIDANCE_SCALE = 1.2  # Controls adherence to the input image
TEXT_GUIDANCE_SCALE = 7.5  # Controls adherence to the text prompt

# Prompts for gender-specific images
PROMPTS = {
    "male": "a highly detailed, expressive, and vibrant portrait of a man in the style of a Renaissance painting.",
    "female": "a highly detailed, expressive, and vibrant portrait of a woman in the style of a Renaissance painting."
}

DEBUG_MODE = False  # Set this to True to save debug images

### END GLOBAL VARIABLES ###

### GENERAL SETTINGS ###

# Directories
INPUT_DIRS = {
    "male": "./incoming_images/male",
    "female": "./incoming_images/female"
}
OUTPUT_DIR = "./model_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model ID for the pipeline
MODEL_ID = "timbrooks/instruct-pix2pix"

# Device settings
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

# Timer variables
start_time_script = time.time()
model_load_time = 0
total_images_processed = 0
image_processing_times = []

# Function to download and load an image
def download_image(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

# Function to load an image from a local path or a fallback URL
def load_image(image_path, fallback_url=None):
    try:
        print(f"Loading the image from the local path: {image_path}")
        image = Image.open(image_path)
    except FileNotFoundError:
        if fallback_url:
            print(f"Local image not found. Loading from fallback URL: {fallback_url}")
            image = download_image(fallback_url)
        else:
            raise
    return image

print(f"Using device: {DEVICE}")

print("Loading the Stable Diffusion model...")
start_time_model_loading = time.time()
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if USE_CUDA else torch.float32,
    safety_checker=None
).to(DEVICE)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
end_time_model_loading = time.time()
model_load_time = end_time_model_loading - start_time_model_loading
print(f"Model loaded and ready. Time taken to load model: {model_load_time:.2f} seconds")

# Function to maintain aspect ratio during resizing
def resize_with_aspect_ratio(image, size, resample=Image.LANCZOS):
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    if original_width > original_height:
        new_width = size
        new_height = int(size / aspect_ratio)
    else:
        new_height = size
        new_width = int(size * aspect_ratio)
    return image.resize((new_width, new_height), resample)

# Function to blend images softly
def blend_images(base_img, enhanced_img, alpha=0.5):
    blended = Image.blend(base_img, enhanced_img, alpha)
    return blended

# Function to process an image with the specified prompt
def process_image(image_path, prompt, output_path):
    global total_images_processed
    print(f"Processing image: {image_path} with prompt: '{prompt}'")
    start_time_image = time.time()

    # Set the path for the enhanced input image (non-debug files not saved)
    enhanced_input_image_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_enhanced.png"
    enhanced_input_image_path = os.path.join(output_path, enhanced_input_image_filename)

    # Parameters for enhancing the image
    enhance_params = {
        "TARGET_SIZE": 1024,
        "DENOISE_STRENGTH": (2, 2, 7, 21),
        "SHARPEN_AMOUNT": 0.8,
        "SHARPNESS_ENHANCE": 15.0,
        "CONTRAST_ENHANCE": 1.1,
        "COLOR_ENHANCE": 1.2
    }

    # Enhance the initial input image (save only if DEBUG_MODE is True)
    enhanced_input_image_temp_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(image_path))[0]}_temp_enhanced.png")
    enhance_image(image_path, output_path, **enhance_params)
    
    if DEBUG_MODE:
        os.rename(enhanced_input_image_temp_path, enhanced_input_image_path)
        print(f"Debug: Enhanced input image saved to: {enhanced_input_image_path}")

    # Check if enhanced image exists before proceeding
    if not os.path.exists(enhanced_input_image_path):
        print(f"Error: Enhanced image not found at {enhanced_input_image_path}")
        return

    # Load the enhanced input image
    image = load_image(enhanced_input_image_path)

    # Resize for the model input while preserving aspect ratio
    image = resize_with_aspect_ratio(image, BASE_RESOLUTION)
    
    # When adjusting the cropped size, use both original dimensions carefully
    width, height = image.size
    if width > height:
        new_height = BASE_RESOLUTION
        new_width = int(width * BASE_RESOLUTION / height)
    else:
        new_width = BASE_RESOLUTION
        new_height = int(height * BASE_RESOLUTION / width)
        
    image = image.resize((new_width, new_height), Image.LANCZOS)

    try:
        start_time_processing = time.time()
        edited_image = pipe(
            prompt,
            image=image,
            num_inference_steps=NUM_INFERENCE_STEPS,
            image_guidance_scale=IMAGE_GUIDANCE_SCALE,
            text_guidance_scale=TEXT_GUIDANCE_SCALE
        ).images[0]
        end_time_processing = time.time()
        processing_time = end_time_processing - start_time_processing
        image_processing_times.append(processing_time)
        total_images_processed += 1
        print(f"Time taken for processing this image: {processing_time:.2f} seconds")

        # Resize back to original dimensions after processing
        edited_image = resize_with_aspect_ratio(edited_image, TARGET_RESOLUTION)
        print(f"Resized edited image to: {edited_image.size}")

        # Generate timestamp and create new filename with HHMMSS
        timestamp = datetime.now().strftime("%H%M%S")
        output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{timestamp}.png"
        output_file_path = os.path.join(output_path, output_filename)

        edited_image.save(output_file_path)
        print(f"Saved edited image to: {output_file_path}")

        if DEBUG_MODE:
            # Set the path for the final enhanced output image
            final_enhanced_image_filename = f"{os.path.splitext(output_filename)[0]}_final_enhanced.png"
            final_enhanced_path = os.path.join(output_path, final_enhanced_image_filename)

            # Enhance the final output image and save
            enhance_image(output_file_path, output_path, **enhance_params)

            final_enhanced_image_temp_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(output_file_path))[0]}_enhanced.png")
            os.rename(final_enhanced_image_temp_path, final_enhanced_path)
            print(f"Final enhanced image saved to: {final_enhanced_path}")

            # Load the original and final enhanced images
            original_img = Image.open(output_file_path)
            final_enhanced_img = Image.open(final_enhanced_path)

            # Blur the final enhanced image slightly to smooth the transition
            final_enhanced_img = final_enhanced_img.filter(ImageFilter.GaussianBlur(radius=2))

            # Blend the final enhanced and original images
            blended_img = blend_images(original_img, final_enhanced_img, alpha=0.6)

            # Save the final blended image
            blended_img.save(final_enhanced_path)
            print(f"Blended and saved final image to: {final_enhanced_path}")

    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA out of memory error: {e}. Try reducing image size or inference steps further.")
        return

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return

    finally:
        if not DEBUG_MODE:
            if os.path.exists(enhanced_input_image_path):
                os.remove(enhanced_input_image_path)
            if os.path.exists(enhanced_input_image_temp_path):
                os.remove(enhanced_input_image_temp_path)
        # Clear unused variables and memory
        del image
        del edited_image
        torch.cuda.empty_cache()
        gc.collect()

# Process all images in the directories
for gender, input_dir in INPUT_DIRS.items():
    print(f"Processing images from the {gender} directory: {input_dir}")
    for img_filename in os.listdir(input_dir):
        if img_filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            input_image_path = os.path.join(input_dir, img_filename)
            output_image_path = OUTPUT_DIR
            process_image(input_image_path, PROMPTS[gender], output_image_path)

# Summary calculation
end_time_script = time.time()
total_time_script = end_time_script - start_time_script
average_image_time = sum(image_processing_times) / total_images_processed if total_images_processed > 0 else 0

# Print summary
print("\n=== SUMMARY ===")
print(f"Model load time: {model_load_time:.2f} secs")
print(f"Total images processed: {total_images_processed}")
print(f"Average time per image to complete: {average_image_time:.3f} secs")
print(f"Total time to complete script: {total_time_script:.2f} secs")