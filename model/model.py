import torch
import clip
import faiss
import pandas as pd
from PIL import Image


# Load your CLIP model
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


model, preprocess, device = load_model()


# Load FAISS index
def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    return index


faiss_index_path = "./data/faiss.index"
index = load_faiss_index(faiss_index_path)

# Load the CSV with image paths and titles
csv_path = "./data/train_cat_breeds.csv"
data_df = pd.read_csv(csv_path)


def encode_text(model, preprocess, device, text):
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features.cpu().numpy()


def search_faiss_index(text_features, index, k=3):
    distances, indices = index.search(text_features, k)
    return indices[0]


def query(title):
    text_features = encode_text(model, preprocess, device, title)
    top_indices = search_faiss_index(text_features, index)
    images = [data_df.iloc[idx]["image_path"] for idx in top_indices]
    return images
