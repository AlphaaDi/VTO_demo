import io
import os
import json
import base64
from openai import OpenAI
from attributes_predictor_module.openai.schemas import GarmentDescription
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

def encode_image_pil(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def resize_image_to_fit_larger_side(img, target_size):
    original_width, original_height = img.size
    if original_width > original_height:
        new_width = target_size
        new_height = int((target_size / original_width) * original_height)
    else:
        new_height = target_size
        new_width = int((target_size / original_height) * original_width)

    resized_image = img.resize((new_width, new_height), Image.LANCZOS)
    return resized_image

def form_image_request_body(prompts, image):
    base64_image = encode_image_pil(image)
    request_body = [{
            "role": "system",
            "content": prompts["system"]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompts["user"]},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]
    return request_body

class GarmentDescriptionGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        relative_path = os.path.join(current_file_directory, 'config.json')
        with open(relative_path, 'r') as f:
            self.config = json.load(f)

    def get_description(self, image):
        resized_image = resize_image_to_fit_larger_side(image, self.config['image_size'])
        messages = form_image_request_body(self.config['prompts'], resized_image)
        response = self.client.beta.chat.completions.parse(
            model=self.config['model'],
            messages=messages,
            response_format=GarmentDescription,
        )
        garment_description = response.choices[0].message.parsed
        return garment_description.text_description