let video = document.getElementById("video");
let prediction = document.getElementById("prediction");
let stream;
let intervalId;

function startCamera() {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then((s) => {
      stream = s;
      video.srcObject = stream;
      intervalId = setInterval(captureAndSend, 1000);
    })
    .catch((err) => {
      console.error("Error accessing webcam:", err);
    });
}

function stopCamera() {
  clearInterval(intervalId);
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }
}

function captureAndSend() {
  let canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  let ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);
  let dataURL = canvas.toDataURL("image/jpeg");

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataURL })
  })
    .then(res => res.json())
    .then(data => {
      prediction.textContent = data.prediction;
    })
    .catch(err => {
      console.error("Prediction error:", err);
    });
}
