import os
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import io

# Set device
gpu_index = os.environ.get("GPU_INDEX", "0")
DEVICE = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.float16  # Use float16 for better compatibility
).to(DEVICE)

# Set model to evaluation mode
model.eval()

# Create a dummy input
dummy_image = Image.new('RGB', (224, 224), color='white')  # Adjust size as needed
buffered = io.BytesIO()
dummy_image.save(buffered, format="JPEG")
image_bytes = buffered.getvalue()

# Define a dummy prompt
dummy_prompt = "This is a dummy prompt for testing."

# Preprocess inputs
inputs = processor(images=dummy_image, text=dummy_prompt, return_tensors="pt")

# Move tensors to the appropriate device
for key in inputs:
    inputs[key] = inputs[key].to(DEVICE)

# Export the model to ONNX
torch.onnx.export(
    model,
    args=(inputs['pixel_values'], inputs['input_ids'], inputs['attention_mask']),
    f="smolvlm_instruct.onnx",
    export_params=True,
    opset_version=11,  # Ensure compatibility with TensorRT
    do_constant_folding=True,
    input_names=['pixel_values', 'input_ids', 'attention_mask'],
    output_names=['output'],
    dynamic_axes={
        'pixel_values': {0: 'batch_size'},
        'input_ids': {0: 'batch_size'},
        'attention_mask': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("Model has been successfully exported to smolvlm_instruct.onnx")
