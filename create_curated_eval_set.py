import pandas as pd
import os
from tqdm import tqdm
import re

from dataset.arxiv_multilabel import clean_and_extract_labels

from configs.arxiv_classification_shared import TASK_A_SUBSET, TASK_B_SUBSET, ALL_LABELS


MIN_TASK_A_LABELS = 1       
MIN_TASK_B_LABELS = 1       
MIN_TOTAL_LABELS = 2        
MAX_TOTAL_LABELS = 7        

SOURCE_CSV_PATH = 'arxiv90k_final_test.csv'
OUTPUT_CSV_PATH = 'arxiv_curated_eval.csv'

def create_curated_set():
    if not os.path.exists(SOURCE_CSV_PATH):
        print(f"Error: Source file not found at '{SOURCE_CSV_PATH}'")
        return

    print(f"Loading source dataset from '{SOURCE_CSV_PATH}'...")
    df = pd.read_csv(SOURCE_CSV_PATH)
    df.dropna(subset=['Abstracts', 'categories'], inplace=True)
    print(f"Source dataset loaded with {len(df)} rows.")

    curated_samples = []

    task_a_set = set(TASK_A_SUBSET)
    task_b_set = set(TASK_B_SUBSET)
    ALL_LABELS_SET = set(ALL_LABELS) 

    print("Filtering for mixed-domain labels based on new tasks...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        labels = set(clean_and_extract_labels(str(row['categories'])))

        if not labels.issubset(ALL_LABELS_SET):
            continue

        if not (MIN_TOTAL_LABELS <= len(labels) <= MAX_TOTAL_LABELS):
            continue

        task_a_count = len(labels.intersection(task_a_set))
        task_b_count = len(labels.intersection(task_b_set))

        if task_a_count >= MIN_TASK_A_LABELS and task_b_count >= MIN_TASK_B_LABELS:
            curated_samples.append(row)

    if not curated_samples:
        print("\nWarning: No samples found matching the criteria. Try making the criteria more lenient (e.g., increase MAX_TOTAL_LABELS).")
        return

    curated_df = pd.DataFrame(curated_samples)
    curated_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"\nSuccessfully created the curated evaluation set!")
    print(f"Found {len(curated_df)} samples matching the criteria.")
    print(f"Saved to '{OUTPUT_CSV_PATH}'")

if __name__ == "__main__":
    create_curated_set()