import torch
import numpy as np
import matplotlib.pyplot as plt
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

# Import our shared class mapping to know which class is which
from configs.reddit_classification_shared import ID_TO_SUBREDDIT, LIFESTYLE_SUBREDDITS, SCIENCE_TECH_SUBREDDITS

# --- Configuration ---
SCIENCE_ADAPTER_PATH = "./lora_rank16/reddit_science_culture_lora"
LIFESTYLE_ADAPTER_PATH = "./lora_rank16/reddit_lifestyle_culture_lora"
BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B"
NUM_LABELS = len(ID_TO_SUBREDDIT)

# --- Helper Function to Get the Head Weights ---
def get_classification_head(adapter_path, base_model_name):
    """
    Loads a PEFT model and returns the weights of its classification head.
    """
    # Load the base model and the adapter
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=NUM_LABELS)
    peft_model = PeftModel.from_pretrained(model, adapter_path)
    
    # The head is usually named 'score' for Llama-based sequence classifiers
    # We return a detached copy on the CPU
    return peft_model.score.weight.detach().cpu()

# --- Main Analysis Script ---
if __name__ == "__main__":
    print("Extracting head from Science adapter...")
    science_head_weights = get_classification_head(SCIENCE_ADAPTER_PATH, BASE_MODEL_NAME)

    print("Extracting head from Lifestyle adapter...")
    lifestyle_head_weights = get_classification_head(LIFESTYLE_ADAPTER_PATH, BASE_MODEL_NAME)

    # In this case, the delta_W is just the final weight, since the head started at zero.
    # We will analyze the magnitude of each row. Each row corresponds to one output class.
    
    # Calculate the L2 norm of each row (i.e., for each output class)
    science_magnitudes_per_class = torch.linalg.norm(science_head_weights, dim=1).numpy()
    lifestyle_magnitudes_per_class = torch.linalg.norm(lifestyle_head_weights, dim=1).numpy()

    class_names = [ID_TO_SUBREDDIT[i] for i in range(NUM_LABELS)]

    # --- Plotting the Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    x = np.arange(len(class_names))

    # --- Science Plot ---
    # Color bars based on domain
    science_colors = ['#1f77b4' if c in SCIENCE_TECH_SUBREDDITS else '#d3d3d3' for c in class_names]
    ax1.bar(x, science_magnitudes_per_class, color=science_colors)
    ax1.set_ylabel("Magnitude of Weight Change (L2 Norm)")
    ax1.set_title("Per-Class Weight Changes in Head from SCIENCE Training", fontsize=14)
    ax1.grid(axis='y')

    # --- Lifestyle Plot ---
    # Color bars based on domain
    lifestyle_colors = ['#ff7f0e' if c in LIFESTYLE_SUBREDDITS else '#d3d3d3' for c in class_names]
    ax2.bar(x, lifestyle_magnitudes_per_class, color=lifestyle_colors)
    ax2.set_ylabel("Magnitude of Weight Change (L2 Norm)")
    ax2.set_title("Per-Class Weight Changes in Head from LIFESTYLE Training", fontsize=14)
    ax2.grid(axis='y')

    plt.xticks(x, class_names, rotation=60, ha="right")
    plt.xlabel("Subreddit Class")
    plt.tight_layout()

    plot_filename = "head_change_analysis.png"
    plt.savefig(plot_filename)
    print(f"\nAnalysis plot saved to {plot_filename}")