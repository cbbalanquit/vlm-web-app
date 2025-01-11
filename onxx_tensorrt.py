import os
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# Set device
gpu_index = os.environ.get("GPU_INDEX", "0")
DEVICE = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16  # You may consider float16 for broader compatibility
).to(DEVICE)

# Set to evaluation mode
model.eval()