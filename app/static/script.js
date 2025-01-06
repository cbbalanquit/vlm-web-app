document.addEventListener("DOMContentLoaded", function() {
  const cameraSelect = document.getElementById("cameraSelect");
  const video = document.getElementById("video");
  const captureButton = document.getElementById("captureButton");
  const promptInput = document.getElementById("promptInput");
  const answerBox = document.getElementById("answerBox");

  let currentStream = null;

  // 1. Prompt for camera access first
  requestCameraAccess()
    .then(() => {
      // 2. Now enumerate devices
      return navigator.mediaDevices.enumerateDevices();
    })
    .then((devices) => {
      const videoDevices = devices.filter((device) => device.kind === "videoinput");
      
      // Populate the dropdown
      videoDevices.forEach((device, index) => {
        const option = document.createElement("option");
        option.value = device.deviceId;
        option.text = device.label || `Camera ${index + 1}`;
        cameraSelect.appendChild(option);
      });

      // If at least one camera is available, start the first one by default
      if (videoDevices.length > 0) {
        console.log("Video Devices:", videoDevices);
        console.log("Using deviceId:", videoDevices[0].deviceId);
        startCamera(videoDevices[0].deviceId);
      }
    })
    .catch((err) => {
      console.error("Permission or enumeration error:", err);
      alert("Could not access camera or enumerate devices. Check console for details.");
    });

  // Function: Prompt for camera access with minimal constraints
  function requestCameraAccess() {
    return new Promise((resolve, reject) => {
      // Minimal constraints to force prompt
      const testConstraints = { video: true, audio: false };
      
      // Request camera access
      navigator.mediaDevices
        .getUserMedia(testConstraints)
        .then((stream) => {
          // Immediately stop the test stream (we only needed it for permission prompt)
          stream.getTracks().forEach((track) => track.stop());
          resolve(); // permission granted
        })
        .catch((err) => {
          reject(err); // permission denied or error
        });
    });
  }

  // 2. Start the selected camera
  function startCamera(deviceId) {
    if (currentStream) {
      // Stop any previously running video streams
      currentStream.getTracks().forEach(track => track.stop());
    }

    const constraints = {
      audio: false,
      video: { deviceId: { exact: deviceId } }
    };

    navigator.mediaDevices.getUserMedia(constraints)
      .then(stream => {
        currentStream = stream;
        video.srcObject = stream;
      })
      .catch(err => {
        console.error("Error accessing camera:", err);
        alert("Could not start camera. Check console for details.");
      });
  }

  // 3. Handle camera dropdown change
  cameraSelect.addEventListener("change", () => {
    startCamera(cameraSelect.value);
  });

  // 4. Capture frame + send to backend
  captureButton.addEventListener("click", () => {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  
    // Convert to base64 data URL (JPEG or PNG)
    const base64Image = canvas.toDataURL("image/jpeg"); 
    const prompt = promptInput.value;

    console.log(base64Image);
  
    // POST to your new route
    fetch("/api/infer_smolvlm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: base64Image,
        prompt: prompt
      })
    })
      .then(response => response.json())
      .then(data => {
        if (data.error) {
          console.error("Server Error:", data.error);
          answerBox.innerText = "Error: " + data.error;
        } else {
          let finalText = data.answer;

          const assistantIndex = finalText.indexOf("Assistant:");
          if (assistantIndex !== -1) {
            finalText = finalText.substring(assistantIndex);
          }

          finalText = "Prompt: " + prompt + "\n\n" + finalText;
          finalText = finalText.replace("Assistant:", "Response:");

          answerBox.innerText = finalText;
        }
      })
      .catch(err => {
        console.error("Fetch Error:", err);
        answerBox.innerText = "Error: " + err;
      });
  });
});
