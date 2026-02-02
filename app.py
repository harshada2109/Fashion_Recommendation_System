import os
import pickle

import numpy as np
from numpy.linalg import norm
from PIL import Image
import streamlit as st
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
from torchvision import models, transforms

feature_list = np.array(pickle.load(open("embeddings.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        std=[0.229, 0.224, 0.225],
    )
])


st.title("Fashion Recommendation System")


def save_uploaded_file(uploaded_file):
    try:
        os.makedirs("uploads", exist_ok=True)
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        print("Error saving file:", e)
        return 0


def feature_extraction(img_path, model, device):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(img_tensor)

    features = features.cpu().numpy().flatten()
    normalized_features = features / norm(features)
    return normalized_features


def recommend(features, feature_list):
    neighbors = NearestNeighbors(
        n_neighbors=6,
        algorithm="brute",
        metric="euclidean"
    )
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


uploaded_file = st.file_uploader("Choose an image")

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image")

        img_path = os.path.join("uploads", uploaded_file.name)
        features = feature_extraction(img_path, feature_extractor, device)
        indices = recommend(features, feature_list)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occurred in file upload")