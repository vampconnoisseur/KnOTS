"""
Fine-tunes a Transformers sequence classification model using Low-Rank Adaptation (LoRA).

This script is designed to take a specific dataset configuration from a shared config file,
load a pre-trained language model, apply a LoRA configuration, and fine-tune it on the
specified Reddit dataset. The best performing adapter, based on validation accuracy, is saved.
"""
import os
import torch
import argparse
import itertools
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import numpy as np
from huggingface_hub import login

from utils import get_config_from_name
from dataset.pushshift_reddit import prepare_train_loaders, prepare_test_loaders

os.environ["TOKENIZERS_PARALLELISM"] = "true"
try:
    token = os.getenv('HUGGINGFACE_TOKEN')
    if token: login(token=token)
except Exception: pass
import transformers
transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

def evaluate_accuracy(model, dataloader, device):
    """
    Calculates the classification accuracy of a model on a given dataset.

    Args:
        model (torch.nn.Module): The sequence classification model to evaluate.
        dataloader (torch.utils.data.DataLoader): The DataLoader for the evaluation data.
        device (torch.device): The device (CPU or CUDA) to run the evaluation on.

    Returns:
        float: The accuracy of the model, as a value between 0.0 and 1.0.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Accuracy", leave=False):
            labels = batch.pop('labels').to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if not all_labels: return 0.0
    return np.mean(np.array(all_preds) == np.array(all_labels))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Sequence Classification LM with LoRA on Reddit.")
    parser.add_argument('--config_name', type=str, default='reddit_merge_classification_config', help="Name of the classification config file.")
    parser.add_argument('--task_index', type=int, required=True, help="Index of the dataset config to use.")
    parser.add_argument('--max_steps', type=int, default=15000, help="Total training steps.")
    parser.add_argument('--eval_every', type=int, default=2000, help="Evaluate every N steps.")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate.")
    args = parser.parse_args()

    config = get_config_from_name(args.config_name)
    dataset_config = config['dataset'][args.task_index]
    model_config = config['model']
    MODEL_SAVE_DIR = config['model_dir']
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    TASK_NAME = dataset_config['name']
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_config['name'], cache_dir=model_config.get('cachedir'))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})


    dataset_config['tokenizer'] = tokenizer
    dataset_config['subreddit_to_id'] = model_config['subreddit_to_id']

    print("Preparing data loaders...")
    train_dataloader = prepare_train_loaders(dataset_config)['full']
    val_dataloader = prepare_test_loaders(dataset_config)['val']

    print(f"Loading base model for Sequence Classification: {model_config['name']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config['name'],
        return_dict=True,
        num_labels=model_config['num_labels'],
        cache_dir=model_config.get('cachedir'),
        torch_dtype=torch.bfloat16
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(**model_config['peft_config'])
    model = get_peft_model(model, peft_config).to(DEVICE)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=100, num_training_steps=args.max_steps
    )

    print(f"Starting Classification Fine-Tuning for: {TASK_NAME}")
    infinite_train_loader = itertools.cycle(train_dataloader)
    progress_bar = tqdm(range(args.max_steps))
    best_val_accuracy = -1.0

    for step in progress_bar:
        model.train()
        batch = next(infinite_train_loader)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.set_description(f"Step {step+1}, Loss: {loss.item():.4f}")

        if (step + 1) % args.eval_every == 0:
            val_accuracy = evaluate_accuracy(model, val_dataloader, DEVICE)
            print(f"\nStep {step+1} | Val Accuracy: {val_accuracy:.4f} (Best: {best_val_accuracy:.4f})")

            if val_accuracy > best_val_accuracy:
                print("New best model found! Saving adapter...")
                best_val_accuracy = val_accuracy
                checkpoint_dir = os.path.join(MODEL_SAVE_DIR, f"{TASK_NAME}_lora")
                model.save_pretrained(checkpoint_dir)

    print("\nTraining finished.")
