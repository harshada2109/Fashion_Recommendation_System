# Fashion Image Recommendation System

This project implements a content-based image retrieval system that recommends visually similar fashion products using ResNet-50 features and K-Nearest Neighbors.
A user uploads an image through a Streamlit interface, and the system returns the closest-matching images from the dataset.

The project uses PyTorch for feature extraction and scikit-learn KNN for similarity search.

## 1. Features

- Extracts 2048-dimensional embeddings using pretrained ResNet-50 (PyTorch).

- Normalizes embeddings for consistent similarity scoring.

- Uses Euclidean distance with KNN to find nearest matches.

- Interactive Streamlit web UI for uploading and previewing results.

- Embeddings are precomputed and stored in .pkl files for fast retrieval.

## 2. Dataset Instructions

You can use either the full or small dataset from Kaggle:

Dataset (High Resolution Images; 23gb - 25gb):

https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

Alternative dataset (Low Resolution Images; 500mb - 600mb):
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

## Steps:

1. Download the dataset from Kaggle.

2. Extract the downloaded archive.

3. Keep only the images folder in the root directory of this project.
4. Delete everything else (CSV files, metadata, etc.).

Your final structure should look like:
```
Recommendation-System/
│
├── images/
│     ├── img1.jpg
│     ├── img2.jpg
│     └── ...
├── main.py
├── app.py
├── requirements.txt
└── README.md
```
## 3. Installation & Setup

Follow these commands exactly:
```
git clone https://github.com/Hrishi1084/Recommendation-System.git
cd Recommendation-System
pip install -r requirements.txt
python main.py
streamlit run app.py
```
### What each step does

`python main.py` : Extracts embeddings for all images and saves them as:

- embeddings.pkl

- filenames.pkl

This step automatically skips if these files already exist.

`streamlit run app.py` : Starts the UI where you can upload an image and receive recommendations.

## 4. Project Structure
```
Recommendation-System/
│
├── app.py              # Streamlit application for similarity search
├── main.py             # Embedding generation script
├── embeddings.pkl      # Saved image feature vectors (auto-generated)
├── filenames.pkl       # List of image paths (auto-generated)
├── images/             # Dataset images
├── uploads/            # Uploaded images (auto-created)
├── requirements.txt
└── README.md
```

## 5. How It Works (Technical Overview)

**1. Feature Extraction**

Uses a pretrained ResNet-50 model without the final classification layer.    
Each image is converted into a 2048-dim vector.

**2. Normalization**

Feature vectors are L2-normalized for stable similarity comparisons.

**3. KNN Search**

scikit-learn’s NearestNeighbors (brute force, Euclidean distance) is used to retrieve the closest images.

**4. Streamlit App**

- User uploads an image

- Feature vector is extracted

- Top 5–6 most similar items are displayed

## Author
**Hrishikesh Suryawanshi**