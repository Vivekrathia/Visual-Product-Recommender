import numpy as np

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model


class FeatureExtractor:

    def __init__(self):

        # Load pretrained ResNet50
        base_model = ResNet50(
            weights="imagenet",
            include_top=True
        )

        # Remove classification layer
        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.layers[-2].output
        )

        print("✅ ResNet50 loaded successfully.")

    def preprocess(self, image_path):

        """
        Load and preprocess one image.
        Returns shape:
        (1, 224, 224, 3)
        """

        img = image.load_img(
            image_path,
            target_size=(224, 224)
        )

        img_array = image.img_to_array(img)

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        img_array = preprocess_input(
            img_array
        )

        return img_array

    def extract(self, image_path):

        """
        Extract embedding from a single image.
        Used by recommender.py
        """

        img_array = self.preprocess(
            image_path
        )

        embedding = self.model.predict(
            img_array,
            verbose=0
        )

        return embedding.flatten()

    def extract_batch(self, batch_images):

        """
        Extract embeddings from a batch.

        Input shape:
        (batch_size,224,224,3)

        Output shape:
        (batch_size,2048)
        """

        return self.model.predict(
            batch_images,
            verbose=0
        )


if __name__ == "__main__":

    extractor = FeatureExtractor()

    sample_image = "../data/images/1163.jpg"

    embedding = extractor.extract(
        sample_image
    )

    print("\nEmbedding Shape:")
    print(embedding.shape)

    print("\nFirst 10 values:")
    print(embedding[:10])