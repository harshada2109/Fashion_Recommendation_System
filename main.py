import os
import pickle
from tqdm import tqdm

import numpy as np
from numpy.linalg import norm
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


if os.path.exists("embeddings.pkl") and os.path.exists("filenames.pkl"):
    print("embeddings.pkl and filenames.pkl already exist. Skipping feature extraction.")
    exit(0)


print("PKL files not found. Running feature extraction...")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

base_model = models.resnet50(pretrained=True)
base_model.eval()

for param in base_model.parameters():
    param.requires_grad = False

feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
feature_extractor.add_module("flatten", nn.Flatten())
feature_extractor.to(device)
feature_extractor.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_features(img_path, model, device):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(img_tensor)

    features = features.cpu().numpy().flatten()
    normalized_features = features / norm(features)
    return normalized_features

filenames = []
for file in os.listdir("images"):
    file_path = os.path.join("images", file)
    if os.path.isfile(file_path):
        filenames.append(file_path)


feature_list = []
for file in tqdm(filenames, desc="Extracting features"):
    feature_list.append(extract_features(file, feature_extractor, device))

pickle.dump(feature_list, open("embeddings.pkl", "wb"))
pickle.dump(filenames, open("filenames.pkl", "wb"))

print("Feature extraction complete. Saved embeddings.pkl and filenames.pkl.")