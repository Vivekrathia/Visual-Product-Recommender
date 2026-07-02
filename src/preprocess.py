import os
import pandas as pd


class DataPreprocessor:

    def __init__(self, csv_path: str, image_folder: str):
        self.csv_path = csv_path
        self.image_folder = image_folder
        self.df = None

    def load_dataset(self):
        """Load the dataset while skipping malformed rows."""
        self.df = pd.read_csv(
            self.csv_path,
            engine="python",
            on_bad_lines="skip"
        )

        print(f"✅ Loaded {len(self.df)} records")

    def remove_missing_values(self):
        """Remove rows with missing important values."""

        before = len(self.df)

        self.df = self.df.dropna(
            subset=[
                "masterCategory",
                "subCategory",
                "articleType",
                "productDisplayName"
            ]
        )

        after = len(self.df)

        print(f"✅ Removed {before-after} rows with missing values")

    def remove_missing_images(self):
        """Remove rows whose image does not exist."""

        valid_rows = []

        for _, row in self.df.iterrows():

            image_name = f"{row['id']}.jpg"

            image_path = os.path.join(
                self.image_folder,
                image_name
            )

            if os.path.exists(image_path):
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows)

        print(f"✅ Remaining records: {len(self.df)}")

    def save_clean_csv(self, output_path):
        self.df.to_csv(output_path, index=False)
        print(f"✅ Saved cleaned dataset to {output_path}")


if __name__ == "__main__":

    processor = DataPreprocessor(
        csv_path="../data/styles.csv",
        image_folder="../data/images"
    )

    processor.load_dataset()
    processor.remove_missing_values()
    processor.remove_missing_images()

    processor.save_clean_csv(
        "../data/styles_clean.csv"
    )