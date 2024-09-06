import os
import cv2
import requests
from PIL import Image, ImageEnhance, ExifTags
import numpy as np
import shutil
import psutil
from datetime import datetime
from tqdm import tqdm
from mtcnn import MTCNN
import dlib
from gfpgan import GFPGANer

#### DEFAULT GLOBAL IMAGE VALUES ####
DEFAULT_VALUES = {
    "ENHANCED_IMAGES_DIRECTORY": "generated_images_enhanced",
    "TARGET_SIZE": 2048,
    "DENOISE_STRENGTH": (1, 1, 5, 10),
    "SHARPEN_AMOUNT": 0.4,
    "SHARPEN_SIGMA": 0.6,
    "SHARPEN_KERNEL_SIZE": (3, 3),
    "SHARPNESS_ENHANCE": 10.1,
    "CONTRAST_ENHANCE": 1.03,
    "COLOR_ENHANCE": 1.05,
    "ADD_COMPARISONS": True,
    "COMPARISONS_FOLDER": "comparisons",
    "RESOURCES_FOLDER": "resources",
    "MODEL_PATH": os.path.join("resources", "GFPGANv1.4.pth"),
    "GFPGAN_HAS_ALIGNED": False,
    "GFPGAN_ONLY_CENTER_FACE": True,
    "GFPGAN_PASTE_BACK": True,
}

def get_values(**kwargs):
    values = DEFAULT_VALUES.copy()
    values.update(kwargs)
    return values

def memory_available():
    return psutil.virtual_memory().available

def download_file(url, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_path, 'wb') as file:
            shutil.copyfileobj(response.raw, file)
        print(f'Downloaded {local_path}')
    else:
        print(f'Failed to download {url}')

def denoise_image(img_cv, denoise_strength):
    h, hForColorComponents, templateWindowSize, searchWindowSize = denoise_strength
    return cv2.fastNlMeansDenoisingColored(img_cv, None, h, hForColorComponents, templateWindowSize, searchWindowSize)

def unsharp_mask(img_cv, kernel_size, sigma, amount, threshold=0):
    blurred = cv2.GaussianBlur(img_cv, kernel_size, sigma)
    sharpened = float(amount + 1) * img_cv - float(amount) * blurred
    sharpened = np.maximum(sharpened, 0)
    sharpened = np.minimum(sharpened, 255)
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(img_cv - blurred) < threshold
        np.copyto(sharpened, img_cv, where=low_contrast_mask)
    return sharpened

def safe_save(filepath, img):
    base, ext = os.path.splitext(filepath)
    if os.path.exists(filepath):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_base = f"{timestamp}_{os.path.basename(base)}"
        filepath = os.path.join(os.path.dirname(base), new_base + ext)
    img.save(filepath)
    print(f'Saved: {filepath}')

def process_image(filepath, face_detector, landmark_predictor, gfpgan, values):
    try:
        print(f'Opening image: {filepath}')
        with Image.open(filepath) as img:
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = img._getexif()
                if exif is not None:
                    exif = dict(exif.items())
                    orientation_value = exif.get(orientation)
                    if orientation_value == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation_value == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation_value == 8:
                        img = img.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                pass

            print(f'Original size: {img.size} ({os.path.getsize(filepath) / (1024 * 1024):.2f} MB)')
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            faces = face_detector.detect_faces(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            print(f'Number of faces detected: {len(faces)}')

            for face in faces:
                x, y, w, h = face['box']
                face_img = img_cv[y:y+h, x:x+w]
                if gfpgan is not None:
                    _, _, face_img = gfpgan.enhance(face_img, has_aligned=values["GFPGAN_HAS_ALIGNED"], only_center_face=values["GFPGAN_ONLY_CENTER_FACE"], paste_back=values["GFPGAN_PASTE_BACK"])
                else:
                    face_img = denoise_image(face_img, values["DENOISE_STRENGTH"])
                    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    rect = dlib.rectangle(0, 0, face_img.shape[1], face_img.shape[0])
                    landmarks = landmark_predictor(gray, rect)
                    face_img = unsharp_mask(face_img, kernel_size=values["SHARPEN_KERNEL_SIZE"], sigma=values["SHARPEN_SIGMA"], amount=values["SHARPEN_AMOUNT"])

                img_cv[y:y+h, x:x+w] = face_img

            img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            img = ImageEnhance.Sharpness(img).enhance(values["SHARPNESS_ENHANCE"])
            img = ImageEnhance.Contrast(img).enhance(values["CONTRAST_ENHANCE"])
            img = ImageEnhance.Color(img).enhance(values["COLOR_ENHANCE"])

            if max(img.size) > values["TARGET_SIZE"]:
                scale_factor = values["TARGET_SIZE"] / max(img.size)
                new_dimensions = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
                img = img.resize(new_dimensions, Image.LANCZOS)
                print(f'Resized final image to ensure longest side is at most {values["TARGET_SIZE"]}: {img.size}')

            return img

    except IOError as e:
        print(f"Error processing file {filepath}: {e}, skipping...")
        return None

def enhance_image(filepath, output_folder, **kwargs):
    values = get_values(**kwargs)  # Get default and overridden values
    
    face_detector = MTCNN()    
    face_predictor_path = os.path.join(values["RESOURCES_FOLDER"], 'shape_predictor_68_face_landmarks.dat')

    if not os.path.exists(face_predictor_path):
        download_url = "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
        print(f"{face_predictor_path} not found. Downloading from {download_url}...")
        download_file(download_url, face_predictor_path)

    landmark_predictor = dlib.shape_predictor(face_predictor_path)

    if not os.path.exists(values["MODEL_PATH"]):
        model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
        print(f"GFPGAN model not found. Downloading from {model_url}...")
        download_file(model_url, values["MODEL_PATH"])

    print("Initializing GFPGAN for face enhancement...")
    gfpgan = GFPGANer(
        model_path=values["MODEL_PATH"],
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None
    )

    processed_img = process_image(filepath, face_detector, landmark_predictor, gfpgan, values)
    if processed_img:
        base_filename = os.path.basename(filepath).split('.')[0]
        new_filepath = os.path.join(output_folder, f"{base_filename}_enhanced.png")
        safe_save(new_filepath, processed_img)

        if values["ADD_COMPARISONS"]:
            comparison_filepath = os.path.join(values["COMPARISONS_FOLDER"], f"{base_filename}_comparison.png")
            os.makedirs(values["COMPARISONS_FOLDER"], exist_ok=True)
            with Image.open(filepath) as orig_img:
                common_height = min(orig_img.height, processed_img.height)
                orig_img_resized = orig_img.resize((int(orig_img.width * common_height / orig_img.height), common_height), Image.LANCZOS)
                processed_img_resized = processed_img.resize((int(processed_img.width * common_height / processed_img.height), common_height))

                comparison_img = Image.new('RGB', (orig_img_resized.width + processed_img_resized.width, common_height))
                comparison_img.paste(orig_img_resized, (0, 0))
                comparison_img.paste(processed_img_resized, (orig_img_resized.width, 0))

                safe_save(comparison_filepath, comparison_img)

    else:
        print(f"Failed to process {filepath}")

def main(image_files, **kwargs):
    values= get_values(**kwargs)  # Get default and overridden values
    os.makedirs(values["COMPARISONS_FOLDER"], exist_ok=True)
    os.makedirs(values["ENHANCED_IMAGES_DIRECTORY"], exist_ok=True)

    for filepath in tqdm(image_files):
        enhance_image(filepath, values["ENHANCED_IMAGES_DIRECTORY"], **kwargs)

if __name__ == '__main__':
    main()