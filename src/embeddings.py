import json
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_extractor import FeatureExtractor


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class EmbeddingGenerator:

    BATCH_SIZE = 32
    CHECKPOINT_INTERVAL = 16  # Save every 16 batches (~512 images)

    def __init__(self, csv_path=None, image_folder=None):

        self.csv_path = csv_path or os.path.join(
            BASE_DIR,
            "data",
            "styles_clean.csv"
        )

        self.image_folder = image_folder or os.path.join(
            BASE_DIR,
            "data",
            "images"
        )

        self.embeddings_dir = os.path.join(
            BASE_DIR,
            "embeddings"
        )

        os.makedirs(
            self.embeddings_dir,
            exist_ok=True
        )

        self.embedding_file = os.path.join(
            self.embeddings_dir,
            "image_embeddings.npy"
        )

        self.image_paths_file = os.path.join(
            self.embeddings_dir,
            "image_paths.pkl"
        )

        self.progress_file = os.path.join(
            self.embeddings_dir,
            "progress.json"
        )

        print("Loading Feature Extractor...")
        self.extractor = FeatureExtractor()

    def save_checkpoint(
        self,
        embeddings,
        image_paths,
        processed
    ):

        np.save(
            self.embedding_file,
            np.array(embeddings)
        )

        with open(
            self.image_paths_file,
            "wb"
        ) as f:
            pickle.dump(
                image_paths,
                f
            )

        with open(
            self.progress_file,
            "w"
        ) as f:

            json.dump(
                {
                    "processed": processed
                },
                f,
                indent=4
            )

        print(f"\n💾 Checkpoint saved ({processed} images)\n")

    def load_checkpoint(self):

        if not os.path.exists(self.progress_file):

            return [], [], 0

        print("\n🔄 Checkpoint found.")

        embeddings = np.load(
            self.embedding_file
        ).tolist()

        with open(
            self.image_paths_file,
            "rb"
        ) as f:

            image_paths = pickle.load(f)

        with open(
            self.progress_file,
            "r"
        ) as f:

            processed = json.load(f)["processed"]

        print(
            f"Loaded {processed} images."
        )

        return (
            embeddings,
            image_paths,
            processed
        )

    def generate_embeddings(self):

        df = pd.read_csv(
            self.csv_path
        )

        (
            embeddings,
            image_paths,
            processed
        ) = self.load_checkpoint()

        total_images = len(df)

        print(
            f"\nFound {total_images} images.\n"
        )

        batch_counter = 0

        for start in tqdm(

            range(
                processed,
                total_images,
                self.BATCH_SIZE
            ),

            desc="Generating Embeddings"

        ):

            batch = df.iloc[
                start:
                start + self.BATCH_SIZE
            ]

            batch_images = []

            batch_paths = []

            for row in batch.itertuples(index=False):

                image_name = f"{row.id}.jpg"

                image_path = os.path.join(
                    self.image_folder,
                    image_name
                )

                if not os.path.exists(
                    image_path
                ):
                    continue

                try:

                    img = self.extractor.preprocess(
                        image_path
                    )

                    batch_images.append(img)

                    batch_paths.append(
                        os.path.abspath(
                            image_path
                        )
                    )

                except Exception as e:

                    print(
                        f"Skipping {image_name}: {e}"
                    )

            if not batch_images:
                continue

            batch_images = np.vstack(
                batch_images
            )

            batch_embeddings = self.extractor.extract_batch(
                batch_images
            )

            for emb, path in zip(
                batch_embeddings,
                batch_paths
            ):

                embeddings.append(emb)

                image_paths.append(path)

            batch_counter += 1

            if (
                batch_counter %
                self.CHECKPOINT_INTERVAL
                == 0
            ):

                self.save_checkpoint(

                    embeddings,

                    image_paths,

                    len(embeddings)

                )

        np.save(
            self.embedding_file,
            np.array(embeddings)
        )

        with open(
            self.image_paths_file,
            "wb"
        ) as f:

            pickle.dump(
                image_paths,
                f
            )

        if os.path.exists(
            self.progress_file
        ):
            os.remove(
                self.progress_file
            )

        print("\n===================================")
        print("Embedding generation completed!")
        print("===================================")
        print(
            f"Total embeddings : {len(embeddings)}"
        )
        print(
            f"Embedding shape  : {np.array(embeddings).shape}"
        )
        print(
            f"Saved to: {self.embeddings_dir}"
        )


if __name__ == "__main__":

    generator = EmbeddingGenerator()

    generator.generate_embeddings()