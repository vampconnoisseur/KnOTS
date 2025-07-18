import json
from datasets import load_dataset
from tqdm.auto import tqdm

# --- Configuration ---
SAMPLES_PER_CATEGORY = 100

# Define the output files and subreddit lists for each category
CATEGORIES_TO_CURATE = {
    "lifestyle": {
        "subreddits": [
            'movies', 'books', 'fitness', 'cooking', 'travel', 'music', 'art', 'history'
        ],
        "output_file": "lifestyle_samples.jsonl"
    },
    "science": {
        "subreddits": [
            'science', 'technology', 'askscience', 'gadgets', 'space', 'explainlikeimfive'
        ],
        "output_file": "science_samples.jsonl"
    }
}

def curate_data():
    print("--- Automatic Reddit Data Curation Script (Separate Files) ---")

    print("Connecting to fddemarco/pushshift-reddit stream...")
    full_dataset_stream = load_dataset("fddemarco/pushshift-reddit", streaming=True, split="train")

    for category_name, details in CATEGORIES_TO_CURATE.items():
        subreddit_list = details["subreddits"]
        output_file = details["output_file"]
        
        print(f"\nCurating {SAMPLES_PER_CATEGORY} samples for the '{category_name}' category...")
        
        filtered_stream = full_dataset_stream.filter(
            lambda example: example.get("subreddit") in subreddit_list
        )
        
        limited_stream = filtered_stream.take(SAMPLES_PER_CATEGORY)
        
        category_samples = []
        for sample in tqdm(limited_stream, total=SAMPLES_PER_CATEGORY, desc=f"Fetching {category_name} posts"):
            title = sample.get('title', '')
            selftext = sample.get('selftext', '')
            if not isinstance(title, str): title = ""
            if not isinstance(selftext, str): selftext = ""

            curated_sample = {
                "subreddit": sample.get("subreddit"),
                "title": title,
                "selftext": selftext
            }
            category_samples.append(curated_sample)
            
        print(f"Writing {len(category_samples)} '{category_name}' samples to '{output_file}'...")
        with open(output_file, 'w') as f:
            for sample in category_samples:
                f.write(json.dumps(sample) + '\n')
        print(f"Successfully saved '{output_file}'.")

    print("\nCuration complete!")

if __name__ == "__main__":
    curate_data()