
---

# KnOTS Project Runbook: ArXiv Multi-Label Classification

This runbook provides a step-by-step guide to prepare data, train specialist models, generate classification heads, merge adapters, and evaluate the final models for the ArXiv multi-label classification task suite.

---

## Prerequisites

### 1. Environment Setup
Ensure you have created and activated the Conda environment using the provided file.

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

Ensure you have a Hugging Face token set as an environment variable to download the base Llama 3 model.

```bash
export HUGGINGFACE_TOKEN='your_token_here'
```
It is recommended to save this in your `~/.bashrc` or `~/.zshrc` file for persistence.

### 4. Raw Dataset

This experiment requires the `arxiv90k.csv` dataset.

1.  **Download the dataset** from Kaggle at the following URL:  
    [Kaggle_Link](https://www.kaggle.com/datasets/kelixirr/arxiv-multi-label-text-classification-datasets?select=arxiv90k.csv)

2.  **Place the file** in the root directory of the project. The final path should be `./arxiv90k.csv`.

---

## Part 1: Data Preparation

This experiment requires a specific data preparation pipeline to create the specialist, baseline, and evaluation datasets.

### Step 1: Run the Master Data Preparation Script

Execute the all-in-one data preparation script from the project's root directory.

```bash
python -m data_prep_scripts.prepare_all_arxiv_data
```

**What this script does:**
1.  **Creates `./data` directory:** All generated files will be placed here.
2.  **Splits Raw Data:** Creates `arxiv90k_train.csv` and `arxiv90k_test.csv`.
3.  **Creates Training Sets:** Generates the four required training files:
    - `train_task_A_cs.csv` (CS Specialist)
    - `train_task_B_math.csv` (Math Specialist)
    - `train_task_C_combined.csv` (Combined Baseline)
    - `train_task_D_union.csv` (Union Baseline)
4.  **Creates Evaluation Set:** Generates `arxiv_curated_eval.csv` for the final multi-domain evaluation.

---

## Part 2: Training ArXiv LoRA Adapters

You will train four separate adapters: two specialists and two baselines.

### Step 1: Create Directory for Adapters

```bash
mkdir -p lora_adapters_arxiv
```

### Step 2: Determine Training Steps

Based on the output of the data preparation script, calculate the `max_steps` required to train each adapter for ~3 epochs.

**Formula:** `max_steps = (dataset_size / batch_size) * num_epochs`

*(Example calculation with a batch size of 4 and 3 epochs)*
-   **Specialists (CS & Math):** `(12367 / 4) * 3 ≈ 9300 steps`
-   **Combined:** `(24734 / 4) * 3 ≈ 18600 steps`
-   **Union:** `(59318 / 4) * 3 ≈ 44500 steps`

### Step 3: Run Training Commands

Execute the following commands from the project's root directory.

**1. Train the CS Specialist (Task A):**
```bash
python -m training_scripts.train_arxiv_classifier_lora \
    --config_name arxiv_merge_config \
    --task_index 0 \
    --max_steps 9300 \
    --eval_every 1500
```

**2. Train the Math Specialist (Task B):**
```bash
python -m training_scripts.train_arxiv_classifier_lora \
    --config_name arxiv_merge_config \
    --task_index 1 \
    --max_steps 9300 \
    --eval_every 1500
```

**3. Train the "Combined" Baseline (Task C):**
```bash
python -m training_scripts.train_arxiv_classifier_lora \
    --config_name arxiv_merge_config \
    --task_index 2 \
    --max_steps 18600 \
    --eval_every 3000
```

**4. Train the "Union" Baseline (Task D):**
```bash
python -m training_scripts.train_arxiv_classifier_lora \
    --config_name arxiv_merge_config \
    --task_index 3 \
    --max_steps 44500 \
    --eval_every 7500
```

---

## Part 3: Head Extraction and Final Evaluation

### Step 1: Extract Specialist Classification Heads

After training the two specialist adapters (CS and Math), run the head extraction script.

```bash
python -m head_extraction_scripts.generate_arxiv_heads
```

This will create the `head_layers/arxiv_heads.pt` file, which contains the classification layers from the two specialist models.

### Step 2: Configure and Run the Final Evaluation

The `evaluate_merged_arxiv_classifier.py` script is designed to run the full suite of evaluations.

**1. Configure the script:**
   - Open `eval_scripts/evaluate_merged_arxiv_classifier.py`.
   - On the first run, you may want to set `COMPUTE_TRANSFORM = True` (line 119) to generate the SVD ingredients file. For subsequent runs, set it to `False` to load the pre-computed file, which is much faster.

**2. Run the Evaluation Script:**
```bash
python -m eval_scripts.evaluate_merged_arxiv_classifier
```

**What this script does:**
-   Evaluates the individual **CS** and **Math** specialists.
-   Merges the specialists using **Averaged Head** and **SVD Merged Head** methods and evaluates them.
-   Evaluates the **Combined** and **Union** baselines.
-   Saves all results and detailed predictions to the `./predictions/` directory.

Happy Experimenting! 🎯