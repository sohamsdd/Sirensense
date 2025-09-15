import os
import shutil
import random

SEED = 42
random.seed(SEED)

DATA_DIR = "data/processed"
OUTPUT_DIR = "dataset"
CLASSES = ["emergency", "urban", "emergency2urban", "urban2emergency"]

SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}

def ensure_dirs():
    for split in SPLITS.keys():
        for c in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, c), exist_ok=True)

def split():
    for c in CLASSES:
        files = os.listdir(os.path.join(DATA_DIR, c))
        random.shuffle(files)

        n = len(files)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:]
        }

        for split, split_files in splits.items():
            for f in split_files:
                src = os.path.join(DATA_DIR, c, f)
                dst = os.path.join(OUTPUT_DIR, split, c, f)
                shutil.copy(src, dst)
        print(f"Split {c}: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")

if __name__ == "__main__":
    ensure_dirs()
    split()
