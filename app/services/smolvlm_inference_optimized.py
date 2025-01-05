# app/services/smolvlm_inference.py
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from transformers.image_utils import load_image
from io import BytesIO
import base64

# Memory optimization settings
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Load model and processor only once
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")

# Using CPU-only mode for stability
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    device_map={'': 'cpu'},  # Map everything to CPU
    torch_dtype=torch.float32,  # Use float32 for CPU
    low_cpu_mem_usage=True
)

def run_inference(image_data: bytes, user_prompt: str) -> str:
    """
    image_data (bytes): raw bytes of the image (e.g., from base64 decode).
    user_prompt (str): the text prompt/question.
    Return: the model's generated text.
    """
    try:
        # Load and resize image
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")
        pil_image = pil_image.resize((224, 224))

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
        inputs = processor(text=prompt, images=[pil_image], return_tensors="pt")

        # Generate with memory-efficient settings
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                num_beams=1
            )
            
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    except Exception as e:
        print(f"Inference error: {str(e)}")
        raise RuntimeError(f"Inference failed: {str(e)}")