import streamlit as st
import pandas as pd
from model.model import load_model, load_faiss_index, query
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Load your CLIP model and preprocess function
@st.cache_resource
def load_resources():
    model, preprocess, device = load_model()
    index = load_faiss_index("./data/faiss.index")
    data_df = pd.read_csv("./data/train_cat_breeds.csv")
    return model, preprocess, device, index, data_df


model, preprocess, device, index, data_df = load_resources()

# Streamlit UI
st.title("Cat Breed Image Finder")

title = st.text_input("Cat Breed", "Siamese")

if st.button("Find Image"):
    st.write(f"Finding image for {title}...")
    result_df = query(title, model, preprocess, device, index, data_df)
    for _, row in result_df.iterrows():
        st.image(row["image_path"], caption=row["breed"])
