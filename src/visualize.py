import matplotlib.pyplot as plt
from PIL import Image

# from recommender import ProductRecommender

from recommender import ProductRecommender


class RecommendationVisualizer:

    def __init__(self):

        self.recommender = ProductRecommender()

    def show_recommendations(self, query_image, top_k=5):

        recommendations = self.recommender.recommend(
            query_image,
            top_k
        )

        plt.figure(figsize=(15, 4))

        # Query Image
        plt.subplot(1, top_k + 1, 1)
        plt.imshow(Image.open(query_image))
        plt.title("Query")
        plt.axis("off")

        # Recommended Images
        for i, (image_path, score) in enumerate(recommendations, start=2):

            plt.subplot(1, top_k + 1, i)
            plt.imshow(Image.open(image_path))
            plt.title(f"{score:.2f}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    visualizer = RecommendationVisualizer()

    query_image = "../data/images/15970.jpg"

    visualizer.show_recommendations(
        query_image,
        top_k=5
    )