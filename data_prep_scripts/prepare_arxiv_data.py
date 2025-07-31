import pandas as pd
from sklearn.model_selection import train_test_split
from configs.arxiv_classification_shared import TASK_A_SUBSET, TASK_B_SUBSET
from tqdm.auto import tqdm
import os
import re

CWD = os.getcwd() 
DATA_DIR = os.path.join(CWD, "data")

RAW_DATA_SOURCE = 'arxiv90k.csv'

TRAIN_OUTPUT_PATH = os.path.join(DATA_DIR, 'arxiv90k_train.csv')
TEST_OUTPUT_PATH = os.path.join(DATA_DIR, 'arxiv90k_test.csv')
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42

TASK_A_OUTPUT_CSV = os.path.join(DATA_DIR, 'train_task_A_cs.csv')
TASK_B_OUTPUT_CSV = os.path.join(DATA_DIR, 'train_task_B_math.csv')
TASK_C_COMBINED_OUTPUT_CSV = os.path.join(DATA_DIR, 'train_task_C_combined.csv')
TASK_D_UNION_OUTPUT_CSV = os.path.join(DATA_DIR, 'train_task_D_union.csv')

EVAL_OUTPUT_PATH = os.path.join(DATA_DIR, 'arxiv_curated_eval.csv')
EVAL_SOURCE_PATH = TEST_OUTPUT_PATH

MIN_LABELS_FROM_TASK_A = 1
MIN_LABELS_FROM_TASK_B = 1

def clean_and_extract_labels(label_string: str) -> list:
    """Cleans a string of category labels from the ArXiv dataset."""
    if not isinstance(label_string, str):
        return []
    s = re.sub(r'[;,\'\"\[\]\(\)]', ' ', label_string)
    potential_labels = s.split()
    cleaned_labels = [label for label in potential_labels if '.' in label or '-' in label]
    return list(set(cleaned_labels))

def step_1_split_source_data():
    """
    Performs the initial 80/20 train/test split of the raw dataset.
    """
    print("\n" + "="*50)
    print("--- STEP 1: Splitting Raw Data into Train/Test Sets ---")
    print("="*50)

    if not os.path.exists(RAW_DATA_SOURCE):
        print(f"Error: The source file '{RAW_DATA_SOURCE}' was not found.")
        return False

    print(f"Loading the full raw dataset from '{RAW_DATA_SOURCE}'...")
    df = pd.read_csv(RAW_DATA_SOURCE).dropna(subset=['Abstracts', 'categories'])

    print(f"Splitting data into {1-TEST_SPLIT_SIZE:.0%} train and {TEST_SPLIT_SIZE:.0%} test...")
    train_df, test_df = train_test_split(
        df, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE
    )

    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_df.to_csv(TEST_OUTPUT_PATH, index=False)

    print("\nSuccessfully created master data splits:")
    print(f"  - Training source: '{TRAIN_OUTPUT_PATH}' ({len(train_df)} rows)")
    print(f"  - Testing source:  '{TEST_OUTPUT_PATH}' ({len(test_df)} rows)")
    return True

def step_2_create_training_datasets():
    """
    Uses the master training set to create the four specialized training sets.
    """
    print("\n" + "="*50)
    print("--- STEP 2: Creating Balanced and Union Training Datasets ---")
    print("="*50)

    if not os.path.exists(TRAIN_OUTPUT_PATH):
        print(f"Error: Source file '{TRAIN_OUTPUT_PATH}' not found. Please run Step 1 first.")
        return False

    print(f"Loading source data from {TRAIN_OUTPUT_PATH}...")
    df = pd.read_csv(TRAIN_OUTPUT_PATH).dropna(subset=['categories', 'Abstracts'])

    task_a_set = set(TASK_A_SUBSET)
    task_b_set = set(TASK_B_SUBSET)
    domain_umbrella_set = task_a_set.union(task_b_set)

    task_A_disjoint, task_B_disjoint, task_D_union = [], [], []

    print("Filtering samples...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        paper_labels = set(clean_and_extract_labels(str(row['categories'])))

        if not paper_labels or not paper_labels.issubset(domain_umbrella_set):
            continue
        
        task_D_union.append(row)

        is_in_A = not paper_labels.isdisjoint(task_a_set)
        is_in_B = not paper_labels.isdisjoint(task_b_set)

        if is_in_A and not is_in_B:
            task_A_disjoint.append(row)
        elif is_in_B and not is_in_A:
            task_B_disjoint.append(row)

    df_A = pd.DataFrame(task_A_disjoint)
    df_B = pd.DataFrame(task_B_disjoint)
    print(f"\nFound {len(df_A)} pure, disjoint samples for Task A (CS).")
    print(f"Found {len(df_B)} pure, disjoint samples for Task B (MATH).")

    if len(df_A) > 0 and len(df_B) > 0:
        target_size = min(len(df_A), len(df_B))
        print(f"\nBalancing specialist datasets to {target_size} samples each.")
        
        df_A_balanced = df_A.sample(n=target_size, random_state=RANDOM_STATE)
        df_B_balanced = df_B.sample(n=target_size, random_state=RANDOM_STATE)
        
        df_A_balanced.to_csv(TASK_A_OUTPUT_CSV, index=False)
        df_B_balanced.to_csv(TASK_B_OUTPUT_CSV, index=False)
        print(f"  - Saved '{TASK_A_OUTPUT_CSV}' ({len(df_A_balanced)} rows)")
        print(f"  - Saved '{TASK_B_OUTPUT_CSV}' ({len(df_B_balanced)} rows)")

        print("\nCreating combined dataset from balanced specialist sets (Task C)...")
        df_C = pd.concat([df_A_balanced, df_B_balanced]).sample(frac=1, random_state=RANDOM_STATE)
        df_C.to_csv(TASK_C_COMBINED_OUTPUT_CSV, index=False)
        print(f"  - Saved '{TASK_C_COMBINED_OUTPUT_CSV}' ({len(df_C)} rows)")
    else:
        print("\nWarning: Could not create specialist/combined datasets.")

    print("\nCreating union dataset from all pure source samples (Task D)...")
    df_D = pd.DataFrame(task_D_union).sample(frac=1, random_state=RANDOM_STATE)
    df_D.to_csv(TASK_D_UNION_OUTPUT_CSV, index=False)
    print(f"  - Saved '{TASK_D_UNION_OUTPUT_CSV}' ({len(df_D)} rows)")
    return True

def step_3_create_evaluation_set():
    """
    Filters the master test set to create a curated evaluation set
    of pure, multi-domain samples.
    """
    print("\n" + "="*50)
    print("--- STEP 3: Creating Curated Evaluation Set from Test Data ---")
    print("="*50)
    
    if not os.path.exists(EVAL_SOURCE_PATH):
        print(f"Error: Source file not found at '{EVAL_SOURCE_PATH}'")
        return False

    print(f"Loading source dataset from '{EVAL_SOURCE_PATH}'...")
    df = pd.read_csv(EVAL_SOURCE_PATH).dropna(subset=['Abstracts', 'categories'])
    print(f"Source dataset loaded with {len(df)} rows.")

    curated_samples = []
    task_a_set = set(TASK_A_SUBSET)
    task_b_set = set(TASK_B_SUBSET)
    domain_umbrella_set = task_a_set.union(task_b_set)

    print("Filtering for pure, mixed-domain samples...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        labels = set(clean_and_extract_labels(str(row['categories'])))

        if not labels or not labels.issubset(domain_umbrella_set):
            continue

        task_a_count = len(labels.intersection(task_a_set))
        task_b_count = len(labels.intersection(task_b_set))

        if task_a_count >= MIN_LABELS_FROM_TASK_A and task_b_count >= MIN_LABELS_FROM_TASK_B:
            curated_samples.append(row)

    if not curated_samples:
        print("\nWarning: No samples found matching the curation criteria.")
        return True

    curated_df = pd.DataFrame(curated_samples)
    curated_df.to_csv(EVAL_OUTPUT_PATH, index=False)

    print(f"\nSuccessfully created the curated evaluation set!")
    print(f"Found {len(curated_df)} samples matching the criteria.")
    print(f"Saved to '{EVAL_OUTPUT_PATH}'")
    return True


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    if step_1_split_source_data():
        if step_2_create_training_datasets():
            if step_3_create_evaluation_set():
                print("\n\n--- All data preparation steps completed successfully! ---")