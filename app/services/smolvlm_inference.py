# app/services/smolvlm_inference.py
import os
import torch
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from transformers.image_utils import load_image
from io import BytesIO
import base64
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

gpu_index = os.environ.get("GPU_INDEX", "0")
DEVICE = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load model and processor only once
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    device_map={"": DEVICE}
)#.to(DEVICE)

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
    start_time = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast('cuda'):
        generated_ids = model.generate(**inputs, max_new_tokens=200)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    elapsed_time = time.perf_counter() - start_time

    num_tokens = len(generated_ids[0])
    tokens_per_sec = num_tokens / elapsed_time

    gpu_memory = torch.cuda.memory_allocated(DEVICE) / (1024 ** 2)

    generated_text += (
        f"\n\nTotal of {elapsed_time:.2f} seconds."
        f"{tokens_per_sec:.2f} tokens/sec."
        f"\nGPU Memory Allocated: {gpu_memory:.2f} MiB."
    )

    torch.cuda.empty_cache()

    return generated_text
