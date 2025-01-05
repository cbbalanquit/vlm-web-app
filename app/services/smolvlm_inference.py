# app/services/smolvlm_inference.py

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from io import BytesIO
import base64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor only once
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.float16
).to(DEVICE)

def run_inference(image_data: bytes, user_prompt: str) -> str:
    """
    image_data (bytes): raw bytes of the image (e.g., from base64 decode).
    user_prompt (str): the text prompt/question.
    Return: the model's generated text.
    """
    # Load image from memory
    try:
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")

    # Construct the messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt}
            ]
        },
    ]

    # Prepare input
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[pil_image], return_tensors="pt").to(DEVICE)

    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=100)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text
