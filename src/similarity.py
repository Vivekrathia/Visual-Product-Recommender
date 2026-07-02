import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityCalculator:

    def __init__(self, embedding_file, image_paths_file):

        self.embeddings = np.load(embedding_file)

        self.image_paths = np.load(
            image_paths_file,
            allow_pickle=True
        )

        print("✅ Embeddings loaded successfully.")
        print(f"Total embeddings: {len(self.embeddings)}")

    def find_similar_images(self, query_embedding, top_k=5):

        similarities = cosine_similarity(
            [query_embedding],
            self.embeddings
        )[0]

        indices = np.argsort(similarities)[::-1]

        results = []

        for idx in indices[:top_k]:

            results.append(
                (
                    self.image_paths[idx],
                    similarities[idx]
                )
            )

        return results


if __name__ == "__main__":

    calculator = SimilarityCalculator(
        embedding_file="../embeddings/image_embeddings.npy",
        image_paths_file="../embeddings/image_paths.pkl"
    )

    query = calculator.embeddings[0]

    results = calculator.find_similar_images(query)

    print("\nTop Similar Images\n")

    for image_path, score in results:
        print(f"{score:.4f}   {image_path}")