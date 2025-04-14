from datasets import load_dataset

# A dictionary mapping display names to their Hugging Face dataset identifiers and desired split.
# Adjust the split name if necessary per dataset.
qa_datasets = {
    "SQuAD": ("squad", "validation"),
    "SQuAD_v2": ("squad_v2", "validation"),
    "Natural Questions": ("natural_questions", "validation"),
    "TriviaQA": ("trivia_qa", "validation"),
    "MS MARCO": ("ms_marco", "validation"),      # Note: this is for passage ranking; check available splits.
    "DROP": ("drop", "validation"),
    "NewsQA": ("newsqa", "validation"),
    "CoQA": ("coqa", "validation"),
    "QuAC": ("quac", "validation"),
    "HotpotQA": ("hotpot_qa", "validation"),
    "WikiHop": ("wikihop", "validation"),
    "NarrativeQA": ("narrative_qa", "validation"),
    "RACE": ("race", "validation"),
    "BioASQ": ("bioasq", "validation"),          # Ensure that the dataset and split exist as expected.
    "WikiQA": ("wiki_qa", "validation")
}

# Loop through the dictionary, load each dataset with streaming enabled, and print its summary.
for name, (hf_name, split) in qa_datasets.items():
    print(f"Loading {name} (dataset key: '{hf_name}', split: '{split}')...")
    try:
        dataset = load_dataset(hf_name, split=split, streaming=True)
        print(f"{name} dataset: ")
        print(dataset)
    except Exception as e:
        print(f"Error loading {name}: {e}")
    print("-" * 60)
