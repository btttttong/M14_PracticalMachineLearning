import streamlit as st
import pandas as pd
import numpy as np
from model.model import query

st.title("What Cat Breed Are You Looking For?")

title = st.text_input("Cat Breed", "Siamese")

if st.button("Find Image"):
    st.write(f"Finding image for {title}...")
    lst_image = query(title)
    for image in lst_image:
        st.image(image, caption=title)
