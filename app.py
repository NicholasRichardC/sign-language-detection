# app.py
from flask import Flask, render_template, request, jsonify
from model import load_model, base64_to_image, preprocess_image
import torch
import string

app = Flask(__name__)

# Load model once saat server start
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("best_asl_model.pth", device=DEVICE)

# Label ASL (A-Z + space + delete + nothing)
labels = list(string.ascii_uppercase) + ['space', 'delete', 'nothing']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_base64 = data['image']
    img_np = base64_to_image(img_base64)
    input_tensor = preprocess_image(img_np).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        pred_label = labels[pred_idx]

    return jsonify({"prediction": pred_label})

if __name__ == '__main__':
    app.run(debug=True)
