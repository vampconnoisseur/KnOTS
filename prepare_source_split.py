import pandas as pd
from sklearn.model_selection import train_test_split
import os

SOURCE_CSV_PATH = 'arxiv90k.csv'
TRAIN_OUTPUT_PATH = 'arxiv90k_train.csv'
TEST_OUTPUT_PATH = 'arxiv90k_test.csv'
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42

def split_source_data():
    if not os.path.exists(SOURCE_CSV_PATH):
        print(f"Error: The source file '{SOURCE_CSV_PATH}' was not found.")
        return

    print(f"Loading the full raw dataset from '{SOURCE_CSV_PATH}'...")
    df = pd.read_csv(SOURCE_CSV_PATH).dropna(subset=['Abstracts', 'categories'])

    print(f"Splitting data into {1-TEST_SPLIT_SIZE:.0%} train and {TEST_SPLIT_SIZE:.0%} test...")
    train_df, test_df = train_test_split(
        df, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE
    )

    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_df.to_csv(TEST_OUTPUT_PATH, index=False)

    print("\nSuccessfully created master data splits from the raw 90k dataset:")
    print(f"  - Training source: '{TRAIN_OUTPUT_PATH}' ({len(train_df)} rows)")
    print(f"  - Testing source:  '{TEST_OUTPUT_PATH}' ({len(test_df)} rows)")

if __name__ == "__main__":
    split_source_data()