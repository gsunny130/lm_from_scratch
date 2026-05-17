# data_utils.py
# Shared dataset splitting logic used by train.py, evaluate.py, and sweep.py.
#
# Why centralize this? train.py and evaluate.py must agree on exactly which
# essays are in the test set. If each file computed its own split independently
# (even with the same seed), any future change to one file could silently
# create a mismatch — training on test data, or evaluating on training data.
# A single source of truth prevents that class of bug.

from datasets import load_dataset

def get_splits(data_file="essays_formatted.jsonl", seed=42):

    dataset = load_dataset("json", data_files=data_file, split="train")

    # Step 1: carve off 20% as a temporary hold-out pool
    # This gives us an 80% train set and a 20% pool to split further.
    split1 = dataset.train_test_split(test_size=0.2, seed=seed)
    train_ds = split1["train"]   # 80%

    # Step 2: split the 20% pool evenly into val and test
    # test_size=0.5 means "half of the 20% pool" = 10% of total each
    split2 = split1["test"].train_test_split(test_size=0.5, seed=seed)
    val_ds  = split2["train"]    # 10%
    test_ds = split2["test"]     # 10%

    return train_ds, val_ds, test_ds
