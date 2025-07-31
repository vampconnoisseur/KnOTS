"""
Evaluates and compares merged ArXiv classifiers against specialists and baselines.

This is the main script for running the final analysis. It performs a comprehensive
evaluation by:
1.  Loading the individual specialist adapters (CS, MATH) and evaluating their
    performance on the curated inter-domain test set.
2.  Loading the specialist adapters, merging them using various techniques
    (e.g., TIES with SVD), merging their classification heads, and evaluating the
    resulting generalist models.
3.  Loading the baseline adapters (Combined, Union) and evaluating their
    performance as a point of comparison.
4.  Saving detailed prediction files (JSON) and summary metrics (F1 scores)
    for each model to allow for in-depth qualitative and quantitative analysis.
"""
import os
import torch
import numpy as np
import json
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from tqdm.auto import tqdm
import transformers
from peft import PeftModel
from copy import deepcopy
from collections import OrderedDict

from transformers import AutoModelForSequenceClassification

from utils import get_config_from_name, prepare_experiment_config, write_to_csv, set_seed
from task_merger import get_merge_handler
from ft_handlers import LoRAScoreHandler
from dataset.arxiv_multilabel import prepare_dataloaders

os.environ["TOKENIZERS_PARALLELISM"] = "true"
transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

class StateDictModel(torch.nn.Module):
    def __init__(self, sd):
        super().__init__()
        self._sd = sd
    def state_dict(self):
        return self._sd

def evaluate_and_get_predictions(model, dataloader, device, id_to_label, threshold=0.5, top_k=5):
    """
    Performs a full evaluation of a model on a dataset.

    Calculates Macro and Micro F1 scores and generates a detailed log of predictions,
    including the raw text, true labels, and the model's top-k predicted labels
    with their confidence scores.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation set.
        device (torch.device): The device to run evaluation on.
        id_to_label (dict): Mapping from class ID to label name.
        threshold (float): Probability threshold for binary classification.
        top_k (int): Number of top predictions to save for qualitative analysis.

    Returns:
        tuple[dict, list]: A tuple containing a dictionary of scores and a list
                           of detailed prediction dictionaries.
    """
    model.to(device)
    model.eval()
    
    all_binary_preds = []
    all_true_labels = []
    detailed_predictions = []

    if hasattr(dataloader.dataset, 'data'):
        raw_texts = [item['Abstracts'] for item in dataloader.dataset.data]
    else:
        raw_texts = [dataloader.dataset.df.iloc[i]['Abstracts'] for i in range(len(dataloader.dataset))]

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating and Capturing Predictions", leave=False)):
            true_labels = batch.pop('labels').to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            
            logits = outputs.logits
            probs = torch.sigmoid(logits)
            
            binary_preds = (probs > threshold).int()
            all_binary_preds.extend(binary_preds.cpu().numpy())
            all_true_labels.extend(true_labels.cpu().numpy())

            top_k_scores, top_k_indices = torch.topk(probs, k=top_k, dim=-1)

            batch_size = len(inputs['input_ids'])
            for j in range(batch_size):
                global_idx = i * dataloader.batch_size + j
                true_indices = torch.where(true_labels[j] == 1)[0].cpu().numpy()
                
                pred_dict = {
                    "abstract": raw_texts[global_idx] if global_idx < len(raw_texts) else "N/A",
                    "true_labels": [id_to_label.get(idx, "Unknown") for idx in true_indices],
                    "predicted_top_k": [id_to_label.get(idx.item(), "Unknown") for idx in top_k_indices[j]],
                    "scores_top_k": [f"{score.item():.4f}" for score in top_k_scores[j]],
                }
                detailed_predictions.append(pred_dict)

    scores = {}
    if not all_true_labels:
        scores['micro_f1'] = 0.0
        scores['macro_f1'] = 0.0
    else:
        scores['micro_f1'] = f1_score(np.array(all_true_labels), np.array(all_binary_preds), average='micro', zero_division=0) * 100
        scores['macro_f1'] = f1_score(np.array(all_true_labels), np.array(all_binary_preds), average='macro', zero_division=0) * 100
        
    return scores, detailed_predictions


def run_arxiv_evaluation():
    """Main function to run the entire ArXiv evaluation pipeline."""
    CONFIG_NAME = 'arxiv_merge_config'
    COMPUTE_TRANSFORM = False

    SEED = 42

    set_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    raw_config = get_config_from_name(CONFIG_NAME, device=device)

    MODEL_DIR = raw_config['model_dir']
    HEADS_PATH = raw_config['heads_path']
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(raw_config['model']['name'])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.sep_token is None: tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    
    id_to_label = raw_config['model']['id_to_label']
    for d_config in raw_config['dataset']:
        d_config['tokenizer'], d_config['label_to_id'] = tokenizer, raw_config['model']['label_to_id']

    eval_config = raw_config['evaluation_dataset']
    eval_config['tokenizer'], eval_config['label_to_id'] = tokenizer, raw_config['model']['label_to_id']
    print("\nPreparing final evaluation loader...")

    eval_loader = prepare_dataloaders(eval_config)['union_eval']
    
    output_dir = "./predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    
    print('\n' + '='*50)
    print("--- Evaluating Specialist Adapters (Individual Performance) ---")
    print('='*50)

    specialist_adapters = {
        "CS": "arxiv_cs_lora",
        "MATH": "arxiv_math_lora",
    }

    for adapter_name, adapter_folder in specialist_adapters.items():
        print(f"\n--- Evaluating Specialist: {adapter_name} ---")
        adapter_path = os.path.join(MODEL_DIR, adapter_folder)
        
        if not os.path.exists(adapter_path):
            print(f"Warning: Adapter not found at '{adapter_path}', skipping.")
            continue
            
        print(f"  > Loading adapter from: {adapter_path}")
        fresh_base_model = AutoModelForSequenceClassification.from_pretrained(
            raw_config['model']['name'], num_labels=raw_config['model']['num_labels'],
            problem_type="multi_label_classification", cache_dir=raw_config['model'].get('cachedir'),
            torch_dtype=torch.bfloat16
        )
        fresh_base_model.resize_token_embeddings(len(tokenizer))
        if fresh_base_model.config.pad_token_id is None:
            fresh_base_model.config.pad_token_id = tokenizer.eos_token_id

        model_with_adapter = PeftModel.from_pretrained(fresh_base_model, adapter_path)

        scores, predictions = evaluate_and_get_predictions(
            model=model_with_adapter, dataloader=eval_loader, device=device, id_to_label=id_to_label)
            
        print(f"  > Macro F1-Score (Specialist: {adapter_name}): {scores['macro_f1']:.2f}%")
        print(f"  > Micro F1-Score (Specialist: {adapter_name}): {scores['micro_f1']:.2f}%")
        
        save_path = os.path.join(output_dir, f"arxiv_specialist_{adapter_name}_predictions.json")
        with open(save_path, 'w') as f:
            json.dump(predictions, f, indent=4)
        print(f"  > Saved detailed predictions to {save_path}")


    print('\n' + '='*50)
    print("--- Evaluating Merged Models ---")
    print('='*50)

    print("Preparing experiment config (loading base LoRA models for merging)...")
    config = prepare_experiment_config(raw_config, load_data=False)

    backbone_merge_config = config['task_merge_config']
    MergeClass = get_merge_handler(backbone_merge_config['representation'])
    BackboneMerger = MergeClass(
        finetuned_models=np.array([m.cpu().eval() for m in config['models']['bases']]),
        pretrained_model=config['models']['new'],
        param_handler=config['param_handler'],
        device=device, merge_config=backbone_merge_config)

    if COMPUTE_TRANSFORM and backbone_merge_config.get('ingredients_path'):
        print("Computing and saving SVD ingredients for backbones...")
        BackboneMerger.transform(backbone_merge_config)
        print(f"Backbone ingredients saved to {backbone_merge_config['ingredients_path']}")

    print(f"\nLoading task-specific heads from {HEADS_PATH}...")
    task_heads = torch.load(HEADS_PATH, map_location='cpu')

    print("Calculating simple averaged head...")
    avg_head_sd = OrderedDict()
    for key in task_heads[next(iter(task_heads))].keys():
        avg_head_sd[key] = torch.stack([head[key] for head in task_heads.values()]).mean(dim=0)
    
    print("\nInstantiating Merge Handler for SVD Head Merge...")
    head_merge_config = deepcopy(backbone_merge_config)
    head_merge_config['ingredients_path'] = None
    head_models = [StateDictModel(sd) for sd in task_heads.values()]
    template_head_sd = next(iter(task_heads.values()))
    zero_head_sd = OrderedDict([(k, torch.zeros_like(v)) for k, v in template_head_sd.items()])
    zero_reference_head_model = StateDictModel(zero_head_sd)
    HeadMerger = MergeClass(
        finetuned_models=np.array(head_models), pretrained_model=zero_reference_head_model, 
        param_handler=LoRAScoreHandler, device=device, merge_config=head_merge_config)
    HeadMerger.transform(head_merge_config)
    svd_head_sd = HeadMerger.merge(head_merge_config, return_state_dict_only=True)
    print("Head merge complete.")

    head_types_to_test = {
        "AVERAGED_HEAD": avg_head_sd,
        "SVD_MERGED_HEAD": svd_head_sd,
    }
    
    for head_name, head_sd in head_types_to_test.items():
        print(f"\n--- Evaluating Merged Backbone with '{head_name}' ---")
        
        print("  > Merging model backbones...")
        eval_model = BackboneMerger.merge(backbone_merge_config)
        
        print(f"  > Attaching '{head_name}'...")
        eval_model.score.load_state_dict(head_sd)
        
        scores, predictions = evaluate_and_get_predictions(
            model=eval_model, dataloader=eval_loader, device=device, id_to_label=id_to_label)
            
        print(f"  > Macro F1-Score ({head_name}): {scores['macro_f1']:.2f}%")
        print(f"  > Micro F1-Score ({head_name}): {scores['micro_f1']:.2f}%")
        
        save_path = os.path.join(output_dir, f"arxiv_merged_{head_name}_predictions.json")
        with open(save_path, 'w') as f:
            json.dump(predictions, f, indent=4)
        print(f"  > Saved detailed predictions to {save_path}")

    
    print('\n' + '='*50)
    print("--- Evaluating Single Adapters Trained on Mixed Data (Baselines) ---")
    print('='*50)

    baselines_to_test = {
        "Combined": "arxiv_combined_lora",
        "Union": "arxiv_union_lora",
    }
    
    for baseline_name, adapter_folder_name in baselines_to_test.items():
        print(f"\n--- Evaluating Baseline: {baseline_name} ---")
        adapter_path = os.path.join(MODEL_DIR, adapter_folder_name)
        
        if not os.path.exists(adapter_path):
            print(f"Warning: Adapter not found at '{adapter_path}', skipping.")
            continue
            
        print(f"  > Loading adapter from: {adapter_path}")
        fresh_base_model = AutoModelForSequenceClassification.from_pretrained(
            raw_config['model']['name'], num_labels=raw_config['model']['num_labels'],
            problem_type="multi_label_classification", cache_dir=raw_config['model'].get('cachedir'),
            torch_dtype=torch.bfloat16
        )
        fresh_base_model.resize_token_embeddings(len(tokenizer))
        if fresh_base_model.config.pad_token_id is None:
            fresh_base_model.config.pad_token_id = tokenizer.eos_token_id

        model_with_adapter = PeftModel.from_pretrained(fresh_base_model, adapter_path)

        scores, predictions = evaluate_and_get_predictions(
            model=model_with_adapter, dataloader=eval_loader, device=device, id_to_label=id_to_label)
            
        print(f"  > Macro F1-Score (Baseline: {baseline_name}): {scores['macro_f1']:.2f}%")
        print(f"  > Micro F1-Score (Baseline: {baseline_name}): {scores['micro_f1']:.2f}%")
        
        save_path = os.path.join(output_dir, f"arxiv_baseline_{baseline_name}_predictions.json")
        with open(save_path, 'w') as f:
            json.dump(predictions, f, indent=4)
        print(f"  > Saved detailed predictions to {save_path}")

    print("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    run_arxiv_evaluation()