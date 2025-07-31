# **Reddit Classification Pipeline Runbook**

This runbook provides a step-by-step guide to train, merge, and evaluate models using the KnOTS framework, focusing on the Reddit subreddit classification task with Llama 3 models.

### **Prerequisites**

1.  **Environment Setup:** Ensure you have created and activated the Conda environment using the provided `conda_environment.yaml` file. This is the recommended way to ensure all dependencies are aligned.
    ```bash
    conda env update --file conda_environment.yaml --prune
    conda activate <your_env_name>
    ```

2.  **Verify Key Library Versions:** This codebase is sensitive to the versions of `peft`, `torch`, and `transformers`. Before running, please ensure your environment has the following specific versions installed:
    ```bash
    pip list | grep -E 'peft|torch|transformers'
    ```
    The output should match:
    *   **`peft`**: `0.3.0`
    *   **`torch`**: `2.1.2` (or a compatible version, check `conda_environment.yaml`)
    *   **`transformers`**: `4.28.1`

    If your versions differ, it is highly recommended to align them to avoid unexpected errors:
    ```bash
    pip install peft==0.3.0 torch==2.1.2 transformers==4.28.1
    ```

3.  **Hugging Face Login:** Make sure you have a Hugging Face token with access to Llama 3 set as an environment variable (`HUGGINGFACE_TOKEN`). The scripts will use this to log in and download the base model.
    ```bash
    export HUGGINGFACE_TOKEN=hf_...
    ```

4.  **Prepare Evaluation Data:** The final evaluation is performed on a curated, dual-label dataset.
    *   Ensure your curated evaluation file is named `reddit_curated_val_data.jsonl`.
    *   Place this file inside the `data/` directory at the root of your project: `KnOTS/data/reddit_curated_val_data.jsonl`.

---

### **Part 1: Training Specialist LoRA Adapters**

The first step is to train individual LoRA adapters specialized for different sets of subreddits (e.g., "lifestyle & culture" vs. "science & tech").

**1. Configure the Main Reddit Config:**

Open `configs/reddit_merge_classification_config.py`. This file controls the entire pipeline.

*   **`BASE_MODEL_NAME`**: Verify this points to the correct Llama 3 model you intend to use (e.g., `"meta-llama/Llama-3.2-3B"`).
*   **`num_train_samples` / `num_val_samples`**: Adjust the number of samples for training and validation. For a full run, `100,000` is a good starting point. For a quick test, you can reduce this significantly.
*   **`MODEL_DIR`**: Check that this points to the directory where you want your trained adapters to be saved (default: `lora_adapters_reddit/`).

**2. Run Training for Each Specialist:**

You will run the `train_reddit_classifier_lora.py` script multiple times, once for each specialist model defined in your config file's `dataset` list. You select the specialist using the `--task_index` argument.

*   **Train the "Lifestyle" Specialist (index 0):**
    ```bash
    python -m training_scripts.train_reddit_classifier_lora \
        --config_name reddit_merge_classification_config \
        --task_index 0
    ```
*   **Train the "Science/Tech" Specialist (index 1):**
    ```bash
    python -m training_scripts.train_reddit_classifier_lora \
        --config_name reddit_merge_classification_config \
        --task_index 1
    ```

*   **Train the "Mixed" Specialist (index 2):**
    This adapter is trained on all subreddits and serves as a baseline for comparison against the merged models.
    ```bash
    python -m training_scripts.train_reddit_classifier_lora \
        --config_name reddit_merge_classification_config \
        --task_index 2
    ```

After these commands complete, the `lora_adapters_reddit/` directory will contain the trained adapters.

---

### **Part 2: Preparing for Merging**

Before you can merge the models, you must extract their classification heads.

**1. Configure the Head Extraction Script:**

*   Open `generate_reddit_heads.py` in the `head_extraction_scripts` directory.
*   Ensure `CONFIG_NAME` is set to `'reddit_merge_classification_config'`.
*   Verify that the `ADAPTER_PATHS` dictionary correctly points to the specialist adapters you just trained.

**2. Run Head Extraction:**

Execute the script from the project root. This will create a `head_layers/` directory and save a `reddit_heads.pt` file inside it.

```bash
python -m head_extraction_scripts.generate_reddit_heads
```

---

### **Part 3: Merging Adapters and Evaluating**

This is the final step where the specialist models are merged and evaluated against the baselines on the curated dataset.

**1. Configure the Evaluation Script:**

*   Open `eval_scripts/evaluate_merged_reddit_classifier.py`.
*   Ensure `CONFIG_NAME` is set to `'reddit_merge_classification_config'`.
*   To compute the SVD "ingredients" for the first time, set **`COMPUTE_TRANSFORM = True`**. For all subsequent runs with the same adapters, you can set this to `False` to load the pre-computed ingredients and save time.

**2. Configure Merge Hyperparameters:**

*   Open `configs/reddit_merge_classification_config.py`.
*   Modify the `task_merge_config` dictionary to experiment with different merging strategies (`merge_method`), representations (`representation`), and scaling coefficients (`scaling_coeffs`).

**3. Run the Full Evaluation Pipeline:**

Execute the evaluation script from the project root.

```bash
python -m eval_scripts.evaluate_merged_reddit_classifier
```

**What the Script Does:**

1.  **Loads Specialist Models:** It loads the LoRA adapters specified in the `bases` list of your config file.
2.  **Computes SVDs (if `COMPUTE_TRANSFORM` is True):** It calculates and saves the SVD ingredients to the path specified by `INGREDIENTS_PATH` in your config.
3.  **Merges Models:** It merges the backbones using your chosen method (e.g., TIES) and merges the classification heads using both SVD and simple averaging.
4.  **Evaluates Merged Models:** It runs the merged model with each type of merged head on the curated dataset.
5.  **Evaluates Baselines:** It then evaluates the original, un-merged specialist adapters and the mixed-data adapter on the same curated dataset for a direct comparison.
6.  **Saves Results:** It saves the final performance metrics to a CSV file in the `csvs_reddit_classification_eval/` directory and detailed predictions to JSONL files in the `predictions_reddit/` directory.