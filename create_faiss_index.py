import torch
import clip
from PIL import Image
import pandas as pd
import faiss
import numpy as np


# Load your CLIP model
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


model, preprocess, device = load_model()

# Load the CSV with image paths and titles
csv_path = "./data/train_cat_breeds.csv"
data_df = pd.read_csv(csv_path)


def encode_images(model, preprocess, device, image_paths):
    all_features = []
    for image_path in image_paths:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        all_features.append(image_features.cpu().numpy())
    return np.vstack(all_features)


# Encode all images
image_paths = data_df["image_path"].tolist()
image_features = encode_images(model, preprocess, device, image_paths)


# Build and save the FAISS index
def build_faiss_index(features):
    dimension = features.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean distance) index
    index.add(features)
    return index


index = build_faiss_index(image_features)
faiss.write_index(index, "./data/faiss.index")

print("FAISS index created and saved successfully.")
