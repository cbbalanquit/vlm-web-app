from flask import Blueprint, request, jsonify
import base64
from app.services.llava_inference import run_inference

inference_bp = Blueprint("inference_bp", __name__)

@inference_bp.route("/infer_smolvlm", methods=["POST"])
def infer_smolvlm():
    """
    Expects JSON or form-data containing:
      - 'imageBase64': base64 string of the image
      - 'prompt': text prompt/question
    Returns JSON: {"answer": "..."}
    """
    data = request.get_json() or request.form

    if "image" not in data:
        return jsonify({"error": "Missing 'imageBase64' field"}), 400
    if "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' field"}), 400

    image_b64 = data["image"]
    prompt = data["prompt"]

    if image_b64.startswith("data:image"):
        image_b64 = image_b64.split(",")[1]

    try:
        image_data = base64.b64decode(image_b64)
    except Exception as e:
        return jsonify({"error": f"Invalid base64 data: {str(e)}"}), 400

    try:
        answer = run_inference(image_data, prompt)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

    return jsonify({"answer": answer})
