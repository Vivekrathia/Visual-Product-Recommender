import os
import sys
import tempfile
import time

import streamlit as st
from PIL import Image

# Allow importing from src/
sys.path.append("src")

from recommender import ProductRecommender


# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Visual Product Recommender",
    page_icon="🛍️",
    layout="wide"
)


# ----------------------------
# Load CSS
# ----------------------------
def load_css():
    css_path = os.path.join("styles", "style.css")

    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )


load_css()


# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_recommender():
    return ProductRecommender()


recommender = load_recommender()


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:

    st.title("⚙ Settings")

    top_k = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=10,
        value=5
    )

    st.divider()

    st.markdown("### 🤖 Model")

    st.info(
        """
        **Backbone:** ResNet50

        **Similarity:** Cosine Similarity

        **Embeddings:** 2048-D
        """
    )


# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
    <div class="main-title">
        🛍️ Visual Product Recommender
    </div>

    <div class="subtitle">
        Upload a product image and discover visually similar products using AI.
    </div>
    """,
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)


# ----------------------------
# Recommendation Pipeline
# ----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    left, right = st.columns([1, 3])

    with left:

        st.markdown(
            '<div class="section-title">📤 Uploaded Image</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as tmp:

        image.save(tmp.name)

        temp_path = tmp.name

    start = time.time()

    with st.spinner("Searching similar products..."):

        recommendations = recommender.recommend(
            temp_path,
            top_k=top_k
        )

    elapsed = time.time() - start

    with right:

        st.markdown(
            '<div class="section-title">⭐ Similar Products</div>',
            unsafe_allow_html=True
        )

        cols = st.columns(top_k)

        for col, product in zip(cols, recommendations):

            with col:

                st.image(
                    product["image_path"],
                    use_container_width=True
                )

                st.markdown(
                    f"""
                    <div class="product-name">
                        {product['product_name']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="badge">
                    👕 {product['article_type']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="badge">
                    🛍 {product['category']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(
                    float(product["similarity"])
                )

                st.caption(
                    f"⭐ {product['similarity']:.2%} Match"
                )

    st.success(
        f"Found {len(recommendations)} similar products in {elapsed:.2f} seconds."
    )

    os.remove(temp_path)


st.markdown("---")

st.markdown(
    """
    <div class="footer">
        Built with ❤️ using Streamlit, TensorFlow & ResNet50
    </div>
    """,
    unsafe_allow_html=True
)