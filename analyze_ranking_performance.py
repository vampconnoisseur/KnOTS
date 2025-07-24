import json
import os
import argparse

def analyze_recall(file_path, k):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    total_recall = 0.0
    num_samples = len(data)

    if num_samples == 0:
        print("Prediction file is empty.")
        return

    for entry in data:
        true_labels_set = set(entry['true_labels'])
        predicted_top_k_set = set(entry['predicted_top_k'][:k])
        
        num_correct_in_top_k = len(true_labels_set.intersection(predicted_top_k_set))
        
        num_true_labels = len(true_labels_set)
        if num_true_labels > 0:
            recall_at_k = num_correct_in_top_k / num_true_labels
        else:
            recall_at_k = 0.0 

        total_recall += recall_at_k

    mean_recall = (total_recall / num_samples) * 100
    
    print(f"\n--- Recall Analysis for: {os.path.basename(file_path)} ---")
    print(f"Mean Recall@{k}:    {mean_recall:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ranking recall from prediction files.")
    parser.add_argument('prediction_files', nargs='+', help="Path(s) to the prediction JSON file(s).")
    parser.add_argument('--k', type=int, default=4, help="The 'k' value for Recall@k. Default is 5.")
    args = parser.parse_args()

    for file in args.prediction_files:
        analyze_recall(file, args.k)