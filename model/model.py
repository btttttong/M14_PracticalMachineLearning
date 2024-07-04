import torch
import clip
import faiss
import pandas as pd
from PIL import Image


# Load your CLIP model
def load_model():
    device = "cpu"  # Explicitly set to CPU
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


# Load FAISS index
def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    return index


# Encode text and search FAISS index
def encode_text(model, preprocess, device, text):
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features.cpu().numpy()


def search_faiss_index(text_features, index, k=3):
    distances, indices = index.search(text_features, k)
    return indices[0]


def query(title, model, preprocess, device, index, data_df):
    text_features = encode_text(model, preprocess, device, title)
    top_indices = search_faiss_index(text_features, index)
    images = data_df.iloc[top_indices]
    return images
