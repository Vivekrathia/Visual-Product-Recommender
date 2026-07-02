import os
import pandas as pd


class DatasetExplorer:
    def __init__(self, csv_path: str, image_folder: str):
        self.csv_path = csv_path
        self.image_folder = image_folder
        self.df = None

    def load_dataset(self):
        """Load the CSV file."""
        self.df = pd.read_csv(
            self.csv_path,
            engine="python",
            on_bad_lines="skip"
        )
        print("✅ Dataset loaded successfully.")

    def dataset_shape(self):
        """Display number of rows and columns."""
        print("\nDataset Shape")
        print("-------------------------")
        print(self.df.shape)

    def show_columns(self):
        """Display all column names."""
        print("\nColumns")
        print("-------------------------")
        print(self.df.columns.tolist())

    def missing_values(self):
        """Display missing values."""
        print("\nMissing Values")
        print("-------------------------")
        print(self.df.isnull().sum())

    def unique_categories(self):
        """Display category information."""
        print("\nMaster Categories")
        print("-------------------------")
        print(self.df["masterCategory"].value_counts())

    def article_types(self):
        """Display article types."""
        print("\nTop Article Types")
        print("-------------------------")
        print(self.df["articleType"].value_counts().head(20))

    # def brands(self):
    #     """Display top brands."""
    #     print("\nTop Brands")
    #     print("-------------------------")
    #     print(self.df["brandName"].value_counts().head(20))

    def image_count(self):
        """Count images inside image folder."""
        total = len([
            file
            for file in os.listdir(self.image_folder)
            if file.endswith(".jpg")
        ])

        print("\nImage Count")
        print("-------------------------")
        print(total)


if __name__ == "__main__":

    explorer = DatasetExplorer(
        csv_path="../data/styles.csv",
        image_folder="../data/images"
    )

    explorer.load_dataset()
    explorer.dataset_shape()
    explorer.show_columns()
    explorer.missing_values()
    explorer.unique_categories()
    explorer.article_types()
    # explorer.brands()
    explorer.image_count()