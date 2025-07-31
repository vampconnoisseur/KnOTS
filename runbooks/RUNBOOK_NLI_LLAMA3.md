# **KnOTS Project Runbook**

This runbook provides a step-by-step guide to train, merge, and evaluate models using the KnOTS framework, focusing on the Natural Language Inference (NLI) task suite with Llama 3 models.

### **Prerequisites**

1.  **Environment Setup:** Ensure you have created and activated the Conda environment using the provided file. This is the recommended way to ensure all dependencies are aligned.
    ```bash
    conda env update --file conda_environment.yaml --prune
    conda activate <your_env_name>
    ```

2.  **Verify Key Library Versions:** This codebase is sensitive to the versions of `peft`, `torch`, and `transformers`. Before running any training or evaluation scripts, please ensure your environment has the following specific versions installed:
    ```bash
    pip list | grep -E 'peft|torch|transformers'
    ```
    The output should match:
    *   **`peft`**: `0.3.0`
    *   **`torch`**: `2.5.1` (or a compatible version, check `conda_environment.yaml`)
    *   **`transformers`**: `4.28.1`
    
    If your versions differ, it is highly recommended to align them to avoid unexpected errors:
    ```bash
    pip install peft==0.3.0 torch==2.1.2 transformers==4.28.1
    ```

3.  **Hugging Face Login:** Make sure you have a Hugging Face token set as an environment variable (`HUGGINGFACE_TOKEN`) to download models and adapters. The scripts will use this to log in.

4.  **Download Task Heads:** For NLI classification tasks, the final classification layer (the "head") of each fine-tuned model is required for evaluation. Download the `nli_heads.pt` file from the following link and place it in the root directory of the project:
    *   **Download Link:** [OneDrive Link for nli_heads.pt](https://gtvault-my.sharepoint.com/:f:/g/personal/pramesh39_gatech_edu/ElwcOO7eMGJKmm9QQKchbzIBXd-YvCnFMiBZ7mFDJDXqGw?e=R9mDWC) (Link from the official repo)

---

### **Part 1: Training NLI LoRA Adapters**

To train a LoRA adapter for a specific NLI task (e.g., SNLI, MNLI, QNLI), you will use the `training_scripts/nli_training.py` script.

**1. Configure the Training Script:**

Open `training_scripts/nli_training.py` and modify the configuration section (**lines 63-75**):

*   **`PTM_MODEL_PATH`**: Path to a pretrained model checkpoint. Can be left empty to start from the base Llama 3 model.
*   **`CACHE_DIR`**: Set the directory where Hugging Face models will be cached.
*   **`MODEL_SAVE_DIR`**: Specify the directory where your trained adapters will be saved.
*   **`TASK`**: **This is the most important setting.** Change this to the NLI dataset you want to train on (e.g., `'snli'`, `'mnli'`, `'qnli'`).
*   **Hyperparameters**: Adjust `MAX_NUM_EPOCHS`, `MAX_STEPS`, `LR`, `BATCH_SIZE`, etc., as needed for your training run.

**2. Choose a Saving Method:**

In the same file (`training_scripts/nli_training.py`), decide how you want to save your model by editing **lines 148-156**:

*   **To save only the LoRA adapter (Recommended & Efficient):**
    Ensure these lines are active:
    ```python
    adapter_save_path = os.path.join(MODEL_SAVE_DIR, TASK)
    model.save_pretrained(adapter_save_path)
    ```
*   **To save the entire fine-tuned model (Large file size):**
    Comment out the adapter-saving lines and uncomment these:
    ```python
    # model_save_path = os.path.join(MODEL_SAVE_DIR, TASK + '.pt')
    # torch.save(model.state_dict(), model_save_path)
    ```
    The `utils/prepare_llama.py` function is designed to correctly handle both cases based on whether the path provided during merging is a directory (for adapters) or a `.pt` file.

**3. (Optional) Customize Dataset Configs:**

If you need to change dataset-specific settings like tokenization length or preprocessing, you can modify the corresponding files in `dataset/configs/`.

**4. Run Training:**

Execute the training script from the root directory. It will start training a LoRA adapter for the specified `TASK`.

```bash
python -m training_scripts.nli_training
```

Repeat this process for each NLI task you wish to include in your model merge.

---

### **Part 2: Merging Adapters and Evaluating**

After training your adapters, you can merge them using various algorithms (like TIES-Merging with SVD) and evaluate their performance. This is done using the `eval_scripts/nli_pertask_linearsearch.py` script.

**1. First-Time Setup: Compute SVD "Ingredients"**

The KnOTS method relies on the Singular Value Decomposition (SVD) of the task adapters. This is a one-time computation per set of adapters.

*   Open `eval_scripts/nli_pertask_linearsearch.py`.
*   Set **`COMPUTE_TRANSFORM = True`** on **line 25**.
*   Open the experiment configuration file `configs/llama8B_r16_knots_ties.py`.
*   On **line 41**, update the `bases` list to point to the directories of the LoRA adapters you just trained (or use the pre-trained Hoffman Lab adapters provided in the config).
*   On **line 3**, set the `INGREDIENTS_PATH` to where you want to save the computed SVDs (e.g., `./nli_svd_ingredients.pt`).

**2. Run the Initial Evaluation:**

This run will compute the SVDs, save them to the `INGREDIENTS_PATH`, and then perform the merge and evaluation.

```bash
python -m eval_scripts.nli_pertask_linearsearch
```

**3. Subsequent Runs:**

After the SVD "ingredients" have been computed and saved, you can **set `COMPUTE_TRANSFORM = False`** in `eval_scripts/nli_pertask_linearsearch.py`. This will make all future runs much faster by loading the pre-computed SVDs from the `.pt` file instead of recalculating them.

**4. Configure the Hyperparameter Search:**

The `nli_pertask_linearsearch.py` script is designed to find the best merging hyperparameters by searching over a defined space.

*   Open `eval_scripts/nli_pertask_linearsearch.py`.
*   Modify the `search_config` dictionary (**lines 57-76**) to define the range of values you want to test for parameters like `scaling_coeffs` and `topK`.

**5. Configure Paths for Evaluation:**

*   In `eval_scripts/nli_pertask_linearsearch.py`, ensure the `TASK_HEADS_PATH` on **line 24** points to your `nli_heads.pt` file.
    ```python
    TASK_HEADS_PATH = "nli_heads.pt"
    ```
*   In `configs/llama8B_r16_knots_ties.py`, double-check that the `bases` list points to the correct set of adapters you want to merge.

**6. Run the Hyperparameter Search and Evaluation:**

Execute the script. It will loop through the hyperparameter combinations defined in `search_config`, merge the models using each combination, evaluate the performance, and save the results to a CSV file.

```bash
python -m eval_scripts.nli_pertask_linearsearch
```

The script will output the performance of each combination and identify the best-performing set of hyperparameters based on validation set accuracy, then run a final evaluation on the test set.