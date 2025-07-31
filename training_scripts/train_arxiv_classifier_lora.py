"""
Trains a LoRA adapter for multi-label classification on a specified ArXiv dataset.

This script is designed to be run multiple times to train different specialist
and baseline models. It is controlled by command-line arguments that specify
which configuration to use and which dataset within that configuration to train on.

Key functionalities include:
- Loading a pre-trained language model (e.g., Llama 3).
- Applying a LoRA configuration for parameter-efficient fine-tuning.
- Calculating and applying class weights to handle dataset imbalance.
- Splitting the training data on-the-fly into training and validation sets.
- Evaluating the model periodically on the validation set using Macro F1-score.
- Saving the best performing adapter checkpoint based on validation performance.
"""
import os
import torch
import argparse
import itertools
import numpy as np
import pandas as pd
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

from utils import get_config_from_name
from dataset.arxiv_multilabel import prepare_dataloaders, clean_and_extract_labels
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def calculate_class_weights(csv_path, label_to_id, max_weight=500.0, min_weight=0.1):
    """
    Calculates positive class weights for BCEWithLogitsLoss based on label frequency.

    This function reads a dataset CSV, counts the occurrences of each label,
    and calculates weights to counteract class imbalance. Weights are capped
    to ensure numerical stability during training.

    Args:
        csv_path (str): Path to the training data CSV file.
        label_to_id (dict): Mapping from label name to its integer ID.
        max_weight (float): The maximum (ceiling) value for any class weight.
        min_weight (float): The minimum (floor) value for any class weight.

    Returns:
        torch.Tensor: A tensor of weights corresponding to each class ID.
    """
    print(f"Calculating class weights from: {csv_path}")
    df = pd.read_csv(csv_path).dropna(subset=['categories'])
    num_labels = len(label_to_id)
    total_samples = len(df)
    
    label_counts = Counter()
    for categories_str in df['categories']:
        labels = clean_and_extract_labels(categories_str)
        for label in labels:
            if label in label_to_id:
                label_counts[label] += 1

    pos_weights = torch.ones(num_labels)
    
    print(f"\n--- Class Frequencies & Weights (Capped at {max_weight}) ---")
    for label, i in label_to_id.items():
        count = label_counts.get(label, 0)
        
        if count == 0:
            raw_weight = max_weight
        elif count == total_samples:
            raw_weight = min_weight
        else:
            raw_weight = (total_samples - count) / count
        
        clipped_weight = max(min_weight, min(raw_weight, max_weight))
        
        pos_weights[i] = clipped_weight
        print(f"{label:<15} | Count: {count:<6} | Raw Weight: {raw_weight:<10.2f} | Clipped Weight: {clipped_weight:.2f}")
    print("-" * 50)

    return pos_weights


def evaluate_f1_score(model, dataloader, device, threshold=0.5):
    """
    Calculates Micro and Macro F1 scores for a multi-label classification model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): The DataLoader for the evaluation set.
        device (torch.device): The device to run evaluation on (e.g., 'cuda').
        threshold (float): The probability threshold to classify a label as positive.

    Returns:
        tuple[float, float]: A tuple containing the micro_f1 and macro_f1 scores.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating F1 Score", leave=False):
            labels = batch.pop('labels').to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            
            logits = outputs.logits
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if not all_labels or len(all_labels) == 0: return 0.0, 0.0
    
    micro_f1 = f1_score(np.array(all_labels), np.array(all_preds), average='micro', zero_division=0)
    macro_f1 = f1_score(np.array(all_labels), np.array(all_preds), average='macro', zero_division=0)

    return micro_f1, macro_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Multi-Label Classifier with LoRA on ArXiv.")
    parser.add_argument('--config_name', type=str, required=True, help="Name of the ArXiv config file (e.g., 'arxiv_merge_config').")
    parser.add_argument('--task_index', type=int, required=True, help="Index of the dataset config to use for this training run.")
    parser.add_argument('--model_save_dir', type=str, default="./lora_adapters_arxiv", help="Dir to save trained adapters.")
    parser.add_argument('--max_steps', type=int, default=10000, help="Total training steps.")
    parser.add_argument('--eval_every', type=int, default=1000, help="Evaluate every N steps.")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate.")
    args = parser.parse_args()

    print(f"Loading config: {args.config_name}")
    config = get_config_from_name(args.config_name)
    dataset_config = config['dataset'][args.task_index]
    model_config = config['model']
    TASK_NAME = dataset_config['name']
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_config['name'], cache_dir=model_config.get('cachedir'))
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.sep_token is None: tokenizer.add_special_tokens({'sep_token': '[SEP]'})

    dataset_config['tokenizer'] = tokenizer
    dataset_config['label_to_id'] = model_config['label_to_id']

    class_weights = calculate_class_weights(dataset_config['path'], model_config['label_to_id']).to(DEVICE)
    
    print("Preparing data loaders...")
    loaders = prepare_dataloaders(dataset_config)
    train_dataloader = loaders['train']
    val_dataloader = loaders['test']

    print(f"Loading base model: {model_config['name']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config['name'],
        num_labels=model_config['num_labels'],
        problem_type="multi_label_classification",
        cache_dir=model_config.get('cachedir'),
        torch_dtype=torch.bfloat16
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.eos_token_id

    peft_config = LoraConfig(**model_config['peft_config'])
    model = get_peft_model(model, peft_config).to(DEVICE)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=args.max_steps)

    print(f"Starting Multi-Label Fine-Tuning for task: {TASK_NAME}")
    infinite_loader = itertools.cycle(train_dataloader)
    progress_bar = tqdm(range(args.max_steps))
    best_val_macro_f1 = -1.0

    for step in progress_bar:
        model.train()
        batch = next(infinite_loader)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        loss = loss_fct(outputs.logits, batch['labels'])
        
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        progress_bar.set_description(f"Step {step+1}, Loss: {loss.item():.4f}")

        if (step + 1) % args.eval_every == 0:
            micro_f1, macro_f1 = evaluate_f1_score(model, val_dataloader, DEVICE)
            print(f"\nStep {step+1} | Validation Micro-F1: {micro_f1:.4f} | Validation Macro-F1: {macro_f1:.4f}")

            if macro_f1 > best_val_macro_f1:
                print(f"New best model found! (Macro F1: {macro_f1:.4f} > {best_val_macro_f1:.4f}). Saving adapter...")
                best_val_macro_f1 = macro_f1
                save_path = os.path.join(args.model_save_dir, f"{TASK_NAME}_lora")
                model.save_pretrained(save_path)

    print("\nTraining finished.")