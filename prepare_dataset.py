import os
import librosa
import soundfile as sf
import numpy as np
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = "data"
ORIG_DIRS = {
    "emergency": os.path.join(BASE_DIR, "original", "emergency"),
    "urban": os.path.join(BASE_DIR, "original", "urban")
}
PROC_DIRS = {
    "emergency": os.path.join(BASE_DIR, "processed", "emergency"),
    "urban": os.path.join(BASE_DIR, "processed", "urban"),
    "emergency2urban": os.path.join(BASE_DIR, "processed", "emergency2urban"),
    "urban2emergency": os.path.join(BASE_DIR, "processed", "urban2emergency")
}

SAMPLES_PER_CLASS = 2500
TARGET_DURATION = 5.0 

for d in PROC_DIRS.values():
    os.makedirs(d, exist_ok=True)

def create_fixed_clips(input_dir, output_dir, label):
    files = os.listdir(input_dir)
    counter = 1
    for file in files:
        path = os.path.join(input_dir, file)
        y, sr = librosa.load(path, sr=None)
        total_samples = int(TARGET_DURATION * sr)

        for _ in range(SAMPLES_PER_CLASS // len(files)):
            if len(y) > total_samples:
                start = random.randint(0, len(y) - total_samples)
                clip = y[start:start + total_samples]
            else:
                clip = librosa.util.fix_length(y, total_samples)
            out_path = os.path.join(output_dir, f"{label}_{counter}.wav")
            sf.write(out_path, clip, sr)
            counter += 1
    print(f"Created {counter-1} files for {label}")

def create_transition_samples(dir1, dir2, output_dir, label):
    files1, files2 = os.listdir(dir1), os.listdir(dir2)
    counter = 1
    while counter <= SAMPLES_PER_CLASS:
        f1, f2 = random.choice(files1), random.choice(files2)
        y1, sr1 = librosa.load(os.path.join(dir1, f1), sr=None)
        y2, sr2 = librosa.load(os.path.join(dir2, f2), sr=None)
        assert sr1 == sr2
        total_samples = int(TARGET_DURATION * sr1)

        if len(y1) > total_samples:
            start1 = random.randint(0, len(y1) - total_samples)
            y1 = y1[start1:start1 + total_samples]
        else:
            y1 = librosa.util.fix_length(y1, total_samples)

        if len(y2) > total_samples:
            start2 = random.randint(0, len(y2) - total_samples)
            y2 = y2[start2:start2 + total_samples]
        else:
            y2 = librosa.util.fix_length(y2, total_samples)

        mixed = 0.5 * y1 + 0.5 * y2
        out_path = os.path.join(output_dir, f"{label}_{counter}.wav")
        sf.write(out_path, mixed, sr1)
        counter += 1
    print(f"Created {counter-1} files for {label}")

def main():
    create_fixed_clips(ORIG_DIRS["emergency"], PROC_DIRS["emergency"], "emergency")
    create_fixed_clips(ORIG_DIRS["urban"], PROC_DIRS["urban"], "urban")
    create_transition_samples(ORIG_DIRS["emergency"], ORIG_DIRS["urban"], PROC_DIRS["emergency2urban"], "emergency2urban")
    create_transition_samples(ORIG_DIRS["urban"], ORIG_DIRS["emergency"], PROC_DIRS["urban2emergency"], "urban2emergency")

if __name__ == "__main__":
    main()
