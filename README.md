# 🛍️ Visual Product Recommender

An AI-powered image recommendation system that finds visually similar fashion products using **ResNet50** feature extraction and **Cosine Similarity**. Upload an image of a product, and the application recommends visually similar items from the dataset.

---

## 📸 Demo

> Add screenshots here after deploying.

| Home Page | Recommendations |
|-----------|-----------------|
| ![Home](screenshots/home.png) | ![Results](screenshots/results.png) |

---

## ✨ Features

- 📤 Upload any fashion product image
- 🧠 Deep feature extraction using **ResNet50**
- ⚡ GPU-accelerated embedding generation (TensorFlow)
- 🔍 Similarity search using **Cosine Similarity**
- 🎯 Top-K visually similar recommendations
- 🎨 Clean Streamlit interface with custom CSS
- 📦 Batch embedding generation for faster preprocessing

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- ResNet50
- NumPy
- Pandas
- Pillow
- Scikit-learn
- Streamlit

---

## 📂 Project Structure

```text
VisualProductRecommender/

├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── embeddings.py
│   ├── feature_extractor.py
│   ├── recommender.py
│   └── similarity.py
│
├── styles/
│   └── style.css
│
├── data/
│   └── styles_clean.csv
│
├── embeddings/
│   ├── image_embeddings.npy
│   └── image_paths.pkl
│
├── models/
│
└── outputs/
```

---

## ⚙️ How It Works

1. Upload a product image.
2. The image is preprocessed and passed through a pretrained **ResNet50** model.
3. A **2048-dimensional feature embedding** is generated.
4. Cosine Similarity compares the embedding against the precomputed dataset embeddings.
5. The most visually similar products are displayed.

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/VisualProductRecommender.git

cd VisualProductRecommender
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📊 Model Details

| Component | Description |
|----------|-------------|
| Backbone | ResNet50 |
| Weights | ImageNet |
| Feature Size | 2048-D |
| Similarity Metric | Cosine Similarity |
| Framework | TensorFlow 2.x |

---

## 📦 Dataset

This project uses the **Fashion Product Images Dataset**.

Due to GitHub file size limitations, the image dataset and generated embeddings are **not included** in this repository.

Place the dataset in:

```text
data/images/
```

Generate embeddings using:

```bash
python src/embeddings.py
```

---

## 🎯 Future Improvements

- FAISS-based similarity search
- CLIP embeddings
- Category and gender filters
- Search history
- Image caching
- Deployment on Streamlit Community Cloud
- Multi-model support

---

## 👨‍💻 Author

**Vivek Rathia**

M.Tech Artificial Intelligence

Amity University

---

## ⭐ If you like this project

If you found this project useful, consider giving it a ⭐ on GitHub!
