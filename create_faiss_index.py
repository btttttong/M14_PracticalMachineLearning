import torch
import clip
from PIL import Image
import pandas as pd
import faiss
import numpy as np
from tqdm import tqdm


def load_model():
    device = "cpu"  # Explicitly set to CPU
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


# Load your CLIP model
print("Loading CLIP model...")
model, preprocess, device = load_model()
print("CLIP model loaded successfully.")

# Load the CSV with image paths and titles
csv_path = "./data/train_cat_breeds.csv"
print(f"Loading CSV from {csv_path}...")
data_df = pd.read_csv(csv_path)
print("CSV loaded successfully.")
print(data_df.head())  # Print the first few rows to ensure it loaded correctly


def encode_images(model, preprocess, device, image_paths):
    all_features = []
    for image_path in tqdm(image_paths, desc="Encoding images"):
        try:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
            all_features.append(image_features.cpu().numpy())
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    return np.vstack(all_features)


# Encode all images
image_paths = data_df["image_path"].tolist()
print(f"Encoding {len(image_paths)} images...")
image_features = encode_images(model, preprocess, device, image_paths)
print("Images encoded successfully.")


# Build and save the FAISS index
def build_faiss_index(features):
    dimension = features.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean distance) index
    index.add(features)
    return index


print("Building FAISS index...")
index = build_faiss_index(image_features)
faiss_index_path = "./data/faiss.index"
faiss.write_index(index, faiss_index_path)
print(f"FAISS index created and saved successfully at {faiss_index_path}.")
