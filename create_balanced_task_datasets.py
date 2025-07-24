import pandas as pd
from tqdm.auto import tqdm
import os

try:
    from dataset.arxiv_multilabel import clean_and_extract_labels
    from configs.arxiv_classification_shared import TASK_A_SUBSET, TASK_B_SUBSET, ALL_LABELS
except ImportError:
    print("Error: Could not import project modules.")
    print("Please run this script from the project's root directory.")
    exit()


SOURCE_CSV_PATH = 'arxiv90k_train.csv'

TASK_A_OUTPUT_CSV = 'train_task_A_cs.csv'
TASK_B_OUTPUT_CSV = 'train_task_B_math.csv'
TASK_C_COMBINED_OUTPUT_CSV = 'train_task_C_combined.csv'
TASK_D_UNION_OUTPUT_CSV = 'train_task_D_union.csv'     

def create_all_training_datasets():
    if not os.path.exists(SOURCE_CSV_PATH):
        print(f"Error: Source file '{SOURCE_CSV_PATH}' not found.")
        print("Please run 'prepare_source_split.py' first.")
        return

    print(f"Loading source data from {SOURCE_CSV_PATH}...")
    df = pd.read_csv(SOURCE_CSV_PATH).dropna(subset=['categories', 'Abstracts'])

    task_a_set = set(TASK_A_SUBSET)
    task_b_set = set(TASK_B_SUBSET)
    domain_umbrella_set = task_a_set.union(task_b_set)

    task_A_disjoint_samples = []
    task_B_disjoint_samples = []
    task_D_union_samples = [] 

    print("Filtering samples for all tasks in a single pass...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        paper_labels = set(clean_and_extract_labels(str(row['categories'])))

        if not paper_labels:
            continue

        if not paper_labels.issubset(domain_umbrella_set):
            continue
        
        task_D_union_samples.append(row)

        is_in_A = not paper_labels.isdisjoint(task_a_set)
        is_in_B = not paper_labels.isdisjoint(task_b_set)

        if is_in_A and not is_in_B:
            task_A_disjoint_samples.append(row)
        elif is_in_B and not is_in_A:
            task_B_disjoint_samples.append(row)

    df_A = pd.DataFrame(task_A_disjoint_samples)
    df_B = pd.DataFrame(task_B_disjoint_samples)
    print(f"\nFound {len(df_A)} pure, disjoint samples for Task A (CS).")
    print(f"Found {len(df_B)} pure, disjoint samples for Task B (MATH).")

    if len(df_A) > 0 and len(df_B) > 0:
        target_size = min(len(df_A), len(df_B))
        print(f"\nBalancing specialist datasets to {target_size} samples each.")
        
        df_A_balanced = df_A.sample(n=target_size, random_state=42)
        df_B_balanced = df_B.sample(n=target_size, random_state=42)
        
        df_A_balanced.to_csv(TASK_A_OUTPUT_CSV, index=False)
        df_B_balanced.to_csv(TASK_B_OUTPUT_CSV, index=False)
        print(f"  - Saved '{TASK_A_OUTPUT_CSV}' ({len(df_A_balanced)} rows)")
        print(f"  - Saved '{TASK_B_OUTPUT_CSV}' ({len(df_B_balanced)} rows)")

        print("\nCreating combined dataset from balanced specialist sets (Task C)...")
        df_C_combined = pd.concat([df_A_balanced, df_B_balanced])
        shuffled_df_C = df_C_combined.sample(frac=1, random_state=42).reset_index(drop=True)
        shuffled_df_C.to_csv(TASK_C_COMBINED_OUTPUT_CSV, index=False)
        print(f"  - Saved '{TASK_C_COMBINED_OUTPUT_CSV}' ({len(shuffled_df_C)} rows)")
    else:
        print("\nWarning: Could not create specialist/combined datasets. Not enough disjoint samples found.")

    print("\nCreating union dataset from all pure source samples (Task D)...")
    df_D_union = pd.DataFrame(task_D_union_samples)
    shuffled_df_D = df_D_union.sample(frac=1, random_state=42).reset_index(drop=True)
    shuffled_df_D.to_csv(TASK_D_UNION_OUTPUT_CSV, index=False)
    print(f"  - Saved '{TASK_D_UNION_OUTPUT_CSV}' ({len(shuffled_df_D)} rows)")

if __name__ == "__main__":
    create_all_training_datasets()