#!/usr/bin/env python3

import cv2
import base64
import requests

def main():
    # Open the default camera (device index 0).
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Capture a single frame.
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        return

    # Release the camera resource.
    cap.release()
    cv2.destroyAllWindows()

    # Convert the frame to bytes (JPEG in-memory).
    _, encoded_img = cv2.imencode('.jpg', frame)
    image_b64 = base64.b64encode(encoded_img).decode("utf-8")

    # Define the prompt.
    prompt = "What do you see in this picture?"

    # Construct the JSON payload.
    payload = {
        "image": image_b64,
        "prompt": prompt
    }

    # POST to your Flask endpoint (adjust host/port as needed).
    resp = requests.post(
        "http://127.0.0.1:5000/api/infer_smolvlm",
        json=payload
    )

    # Print the result.
    if resp.status_code == 200:
        print("Answer:", resp.json().get("answer"))
    else:
        print("Error:", resp.text)

if __name__ == "__main__":
    main()
