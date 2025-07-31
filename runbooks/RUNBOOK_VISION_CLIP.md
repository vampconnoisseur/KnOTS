
# KnOTS Project Runbook: Vision Tasks

This runbook provides a step-by-step guide to train, merge, and evaluate models using the KnOTS framework, focusing on a suite of 8 vision classification tasks with Vision Transformer (ViT) models.

---

## Prerequisites

### 1. Environment Setup
Ensure you have created and activated the Conda environment using the provided file:

```bash
conda env update --file conda_environment.yaml --prune
conda activate knots_env
```

### 2. Verify Key Library Versions

This codebase is sensitive to the versions of `peft`, `torch`, and `transformers`. Run the following to confirm:

```bash
pip list | grep -E 'peft|torch|transformers'
```

Expected versions:
- `peft`: **0.3.0**
- `torch`: **2.1.2**
- `transformers`: **4.28.1**

### 3. Hugging Face Login

Make sure you have your Hugging Face token set as an environment variable:

```bash
export HUGGINGFACE_TOKEN=your_token_here
```

Prefer to save it in your ~/.bashrc or ~/.zshrc file.

### 4. Vision Task Heads (Auto-Generated)

Unlike NLI tasks, the vision evaluation script will **automatically generate** the required classification heads (CLIP text encodings) if not found locally. Ensure paths in config files are writable.

---

## 🔧 Part 1: Training Vision LoRA Adapters

### Step 1: Create Directory for Adapters

```bash
mkdir -p lora_adaptors_vision
```

---

### Step 2: Configure Training Script for a Single Task

Modify `8vision_train.py` to isolate the task you want to train either by adding more maps to dataset list or by changing names if you are training the adaptors separately:

```python
# In 8vision_train.py
config = {
    'dataset': [
        {
            'name': 'dtd',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'type': 'dtd',
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'dtd_head.pt'),
            'batch_size': 32,
            'num_workers': 16,
            'val_fraction': 0.5,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/dtd_shuffled_idxs.pt')
        }
    ],
    'model': {
        'BASE_TYPE': 'ViT-L-14',  # or 'ViT-B-32'
        # other model config...
    },
    # ... rest of config
}
```

---

### Step 3: Run the Training Script

```bash
python -m training_scripts.8vision_training
```

The adapter will be saved under:  
`lora_adaptors_vision/dtd_lora` (for the DTD example).

---

### Step 4: Repeat for All Tasks

Repeat steps 2 and 3 for each task:

- EuroSAT
- DTD
- MNIST
- GTSRB
- SVHN
- Stanford Cars
- SUN397
- RESISC45

---

## Part 2: Merging Adapters & Evaluating with Linear Search

### Step 1: Configure the Merge & Eval Script

Edit `vitL_r16_knots_ties.py`:

```python
# Line 118 in vitL_r16_knots_ties.py
'bases': [
    'hoffman-lab/KnOTS-ViT-L-14_lora_R16_stanford_cars',
    'hoffman-lab/KnOTS-ViT-L-14_lora_R16_dtd',
    # Or your local adapters:
    './lora_adaptors_vision/eurosat_lora',
    './lora_adaptors_vision/dtd_lora',
    # ... add all 8
]
```

Also update model architecture if needed:

```python
# Line 116
'base_type': 'openai/clip-vit-large-patch14'
# or for base model:
# 'base_type': 'openai/clip-vit-base-patch32'
```

---

### Step 2: Linear Hyperparameter Search

The script `8vision_pertask_linearsearch.py` performs a grid search over:

- `scaling_coeffs`
- `topK`

Defined in:

```python
# Lines 53–56 in 8vision_pertask_linearsearch.py
search_config = {
    'scaling_coeffs': [0.2, 0.4, 0.6, 0.8, 1.0],
    'topK': [2, 4, 6, 8],
}
```

---

### Step 3: Run the Script

```bash
python -m eval_scripts.8vision_pertask_linearsearch
```

**What it does:**

- Prepares all 8 datasets and dataloaders.
- Generates `*_head.pt` files if not found.
- Performs linear search for best merge parameters.
- Evaluates merged model performance on test set.

Happy Experimenting! 🎯