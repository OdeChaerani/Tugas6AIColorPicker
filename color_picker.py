"""
Nama    : Wa Ode Zachra Chaerani
NPM     : 140810230062
Kelas   : B
"""

import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

st.set_page_config(page_title="Image Color Picker", layout="wide")
st.title("🎨 SEVENTEEN Image Color Picker")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #f7cac9, #92a8d1);
        background-attachment: fixed;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Settings")
k = st.sidebar.slider("Number of Dominant Colors", 1, 17, 5)

uploaded_file = st.file_uploader("Choose Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    col1, col2 = st.columns([3, 2])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        small_img = image.resize((200, 200))
        img_np = np.array(small_img)
        pixels = img_np.reshape((-1, 3))

        sample_size = min(10000, len(pixels))
        idx = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[idx]

        with st.spinner("Analyzing colors..."):
            model = KMeans(n_clusters=k, init="k-means++", random_state=42)
            model.fit(sample_pixels)

            cluster_labels = model.predict(sample_pixels)
            unique, counts = np.unique(cluster_labels, return_counts=True)
            sorted_idx = np.argsort(-counts)
            sorted_colors = model.cluster_centers_[sorted_idx].astype(int)
            sorted_counts = counts[sorted_idx]
            total = np.sum(sorted_counts)

        st.subheader("Dominant Color Palette")
        fig, ax = plt.subplots(figsize=(k, 2))
        for i, color in enumerate(sorted_colors):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=np.array(color)/255))
        ax.set_xlim(0, k)
        ax.set_ylim(0, 1)
        ax.axis("off")
        st.pyplot(fig)

        st.subheader("Color Codes")
        for i, color in enumerate(sorted_colors):
            hex_color = mcolors.to_hex(color / 255)
            rgb_tuple = tuple(int(x) for x in color)
            percent = (sorted_counts[i] / total) * 100
            st.markdown(
                f"<div style='display:flex; align-items:center; margin-bottom:10px;'>"
                f"<div style='width:40px; height:40px; background-color:{hex_color}; border-radius:5px;'></div>"
                f"<div style='margin-left:10px;'>"
                f"<b>{percent:.1f}%</b> — RGB: {rgb_tuple}, Hex: {hex_color}"
                f"</div></div>",
                unsafe_allow_html=True
            )
