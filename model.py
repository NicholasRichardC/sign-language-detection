# model.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

class ASLModel(nn.Module):
    def __init__(self, num_classes=29):
        super(ASLModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 25 * 25, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(model_path="best_asl_model.pth", device="cpu"):
    model = ASLModel(num_classes=29)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def base64_to_image(base64_str):
    base64_str = base64_str.split(',')[-1]
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data)).convert('RGB')
    return np.array(img)

def preprocess_image(img_np):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor = transform(img_np)
    tensor = tensor.unsqueeze(0)
    return tensor
