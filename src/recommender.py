import os
import pandas as pd

from feature_extractor import FeatureExtractor
from similarity import SimilarityCalculator

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ProductRecommender:

    def __init__(self):

        print("Loading recommendation engine...\n")

        self.extractor = FeatureExtractor()

        self.similarity = SimilarityCalculator(

            embedding_file=os.path.join(
                BASE_DIR,
                "embeddings",
                "image_embeddings.npy"
            ),

            image_paths_file=os.path.join(
                BASE_DIR,
                "embeddings",
                "image_paths.pkl"
            )

        )

        # Load product metadata
        metadata_path = os.path.join(
            BASE_DIR,
            "data",
            "styles_clean.csv"
        )

        self.metadata = pd.read_csv(metadata_path)

        # Fast lookup dictionary
        self.product_lookup = self.metadata.set_index(
            "id"
        ).to_dict("index")

        print("\n✅ Recommendation Engine Ready!")

    def recommend(self, query_image_path: str, top_k: int = 5):

        query_embedding = self.extractor.extract(query_image_path)

        similar_images = self.similarity.find_similar_images(
            query_embedding,
            top_k
        )

        recommendations = []

        for image_path, similarity_score in similar_images:

            image_id = int(
                os.path.splitext(
                    os.path.basename(image_path)
                )[0]
            )

            product = self.product_lookup.get(image_id)

            if product is None:
                continue

            recommendations.append({

                "image_path": image_path,

                "similarity": float(similarity_score),

                "product_name": product["productDisplayName"],

                "category": product["masterCategory"],

                "sub_category": product["subCategory"],

                "article_type": product["articleType"],

                "gender": product["gender"],

                "colour": product["baseColour"],

                "season": product["season"],

                "usage": product["usage"]

            })

        return recommendations


if __name__ == "__main__":

    recommender = ProductRecommender()

    query_image = os.path.join(
        BASE_DIR,
        "data",
        "images",
        "15970.jpg"
    )

    results = recommender.recommend(
        query_image,
        top_k=5
    )

    print("\n========== Recommended Products ==========\n")

    for index, product in enumerate(results, start=1):

        print(f"Recommendation #{index}")
        print(f"Image Path   : {product['image_path']}")
        print(f"Similarity   : {product['similarity']:.4f}")
        print(f"Product Name : {product['product_name']}")
        print(f"Category     : {product['category']}")
        print(f"Sub Category : {product['sub_category']}")
        print(f"Article Type : {product['article_type']}")
        print(f"Gender       : {product['gender']}")
        print(f"Colour       : {product['colour']}")
        print(f"Season       : {product['season']}")
        print(f"Usage        : {product['usage']}")
        print("-" * 60)