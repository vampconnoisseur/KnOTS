import os
import torch
from copy import deepcopy
import numpy as np
from utils import *
from task_merger import get_merge_handler
from models.huggingface_clip import HFLoRACLIPVisionModel

def run_BIG_function():
    EVAL_SPLIT = 'val'
    EVAL_TEST = True
    BIGSEED = 420
    set_seed(BIGSEED)

    CONFIG_NAME = 'vitL_r16_knots_ties'
    print(f"--- Starting Hyperparameter Search for: {CONFIG_NAME} ---")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(CONFIG_NAME, device=device)

    print("\n--- Preparing DataLoaders and Preprocessing ---")
    dummy_model = HFLoRACLIPVisionModel(
        model_name=raw_config['model']['base_type'],
        cache_dir=raw_config['model']['cachedir'],
        lora_config=raw_config['model']['ft_config']
    )
    for dataset_config in raw_config['dataset']:
        dataset_config['train_preprocess'] = dummy_model.train_preprocess
        dataset_config['eval_preprocess'] = dummy_model.val_preprocess
    
    dataloaders = prepare_data(raw_config['dataset'], device=device)
    
    print("\n--- Preparing CLIP Text Encodings (Heads) ---")
    all_clip_encodings = []
    for idx, d_config in enumerate(raw_config['dataset']):
        encoding_path = d_config['clip_encodings']
        if os.path.exists(encoding_path):
            print(f"Loading existing CLIP encodings from {encoding_path}")
            encodings = get_clip_encodings(encoding_path, device=device)
        else:
            print(f"File not found: {encoding_path}. Generating new encodings...")
            os.makedirs(os.path.dirname(encoding_path), exist_ok=True)
            class_names = dataloaders[idx]['test']['class_names']
            encodings = load_clip_features(
                class_names, device, model_name=raw_config['model']['base_type']
            )
            print(f"Saving new encodings to {encoding_path}")
            torch.save(encodings, encoding_path)
        all_clip_encodings.append(encodings)

    print("\n--- Preparing Models for Merging ---")
    config = prepare_experiment_config(raw_config, preloaded_data=dataloaders)
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    print("--- All data and models prepared successfully. ---\n")

    order_of_processing_params = ['scaling_coeffs', 'topK']
    search_config = {
        'scaling_coeffs': np.arange(0.1, 1.1, step=0.1),
        'topK': np.arange(10, 101, step=10),
    }
    best_params_found = deepcopy(config['task_merge_config'])

    fine_tuned_acc = {
        'stanford_cars': 99.77, 'dtd': 70.05, 'eurosat': 98.59, 'gtsrb': 97.20,
        'mnist': 99.53, 'resisc45': 95.70, 'sun397': 79.60, 'svhn': 97.72
    }
    print("Using ViT-L/14 finetuned accuracies for normalization:")
    print(fine_tuned_acc)

    def merge_and_eval(Merge, EVAL_SPLIT='val', instance_config=None):
        set_seed(BIGSEED)
        print(f"\n--- Evaluating with config: {instance_config} ---")
        merged_model = Merge.merge(instance_config)
        
        run_results = {**instance_config}
        avg_norm_accuracy = 0.
        
        print(f"--- Per-task results on '{EVAL_SPLIT}' set ---")
        for i, loader_dict in enumerate(dataloaders):
            dataset_name = dataset_names[i]
            loader = loader_dict['test'][EVAL_SPLIT]
            acc = evaluate_cliphead(merged_model.to(device), loader, class_vectors=all_clip_encodings[i])
            
            raw_acc_percent = acc * 100
            norm_acc_percent = raw_acc_percent / fine_tuned_acc[dataset_name] * 100
            
            print(f"  - {dataset_name:<15}: Acc: {raw_acc_percent:.2f}%  |  Norm Acc: {norm_acc_percent:.2f}%")
            
            run_results[f'{dataset_name}_acc'] = raw_acc_percent
            run_results[f'{dataset_name}_norm_acc'] = norm_acc_percent
            avg_norm_accuracy += norm_acc_percent
        
        avg_norm_accuracy /= len(dataloaders)
        run_results['Average_norm_acc'] = avg_norm_accuracy
        
        print(f'---> Average Normalized Accuracy on {EVAL_SPLIT} set: {avg_norm_accuracy:.3f}')
        return run_results

    with torch.no_grad():
        print(f"\n--- Starting Linear Search for {CONFIG_NAME} ---")
        
        models = np.array([i.cpu().eval() for i in config['models']['bases']])
        MergeClass = get_merge_handler(config['task_merge_config']['representation'])
        Merge = MergeClass(
                deepcopy(models), pretrained_model=deepcopy(config['models']['new']),
                param_handler=config['param_handler'], device=device,
                merge_config=config['task_merge_config']
            )
        Merge.transform(config['task_merge_config'])

        for param_to_tune in order_of_processing_params:
            best_run_in_loop = {'Average_norm_acc': 0.0}
            print(f"\n>>> Tuning parameter: '{param_to_tune}' <<<")
            
            for value in search_config[param_to_tune]:
                current_params = deepcopy(best_params_found)
                current_params[param_to_tune] = value
                
                run_results = merge_and_eval(Merge, EVAL_SPLIT='val', instance_config=current_params)
                
                if run_results['Average_norm_acc'] > best_run_in_loop['Average_norm_acc']:
                    best_run_in_loop = deepcopy(run_results)
            
            best_params_found[param_to_tune] = best_run_in_loop[param_to_tune]
            print(f">>> Best value for '{param_to_tune}' found: {best_run_in_loop[param_to_tune]} with validation score: {best_run_in_loop['Average_norm_acc']:.3f} <<<")

        if EVAL_TEST:
            print("\n--- Evaluating final best parameters on the TEST set ---")
            print("Final best parameters found:", best_params_found)
            final_test_results = merge_and_eval(Merge, EVAL_SPLIT='test', instance_config=best_params_found)
            print("\n--- Final Test Set Results ---")
            print(f"Average Normalized Accuracy: {final_test_results['Average_norm_acc']:.3f}")

if __name__ == "__main__":
    run_BIG_function()