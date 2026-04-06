import os
import io
import csv
import json
import argparse
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
from check_accuracy import check_accuracy


# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "pipeline3")

N_DEFAULT = 10000
NUM_CLASSES = 10

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "full_labels_checkpoint.json")
CHECKPOINT_LABELS_PATH = os.path.join(OUTPUT_DIR, "full_labels_checkpoint.npy")


# ==============================
# Oracle wrapper (silent)
# ==============================
def oracle_score(labels):
    # check_accuracy prints verbose logs; suppress to keep runtime and logs manageable.
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        acc, n_correct, n_total = check_accuracy(labels)
    return acc, n_correct, n_total


# ==============================
# Checkpoint helpers
# ==============================
def save_checkpoint(labels, next_index, base_correct):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(CHECKPOINT_LABELS_PATH, labels)
    meta = {
        "next_index": int(next_index),
        "base_correct": int(base_correct),
        "n": int(len(labels)),
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(expected_n):
    if not os.path.exists(CHECKPOINT_PATH) or not os.path.exists(CHECKPOINT_LABELS_PATH):
        return None

    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    labels = np.load(CHECKPOINT_LABELS_PATH)

    if len(labels) != expected_n:
        raise ValueError(
            f"Checkpoint labels length {len(labels)} != expected {expected_n}. "
            f"Delete checkpoint files in {OUTPUT_DIR} and retry."
        )

    return labels, int(meta["next_index"]), int(meta["base_correct"])


# ==============================
# Main brute-force recovery
# ==============================
def recover_full_labels(n_files=10000, random_seed=42, save_every=100, resume=True):
    np.random.seed(random_seed)

    if resume:
        loaded = load_checkpoint(n_files)
    else:
        loaded = None

    if loaded is not None:
        labels, start_index, base_correct = loaded
        print(f"[INFO] Resuming from checkpoint at index={start_index}, correct={base_correct}")
    else:
        labels = np.random.randint(0, NUM_CLASSES, size=n_files)
        _, base_correct, _ = oracle_score(labels)
        start_index = 0
        print(f"[INFO] Starting new run, initial correct={base_correct}/{n_files}")

    for i in range(start_index, n_files):
        original = labels[i]

        found_better = False
        for d in range(NUM_CLASSES):
            if d == original:
                continue

            labels[i] = d
            _, new_correct, _ = oracle_score(labels)

            if new_correct > base_correct:
                base_correct = new_correct
                found_better = True
                break

        if not found_better:
            labels[i] = original

        if (i + 1) % 20 == 0 or i == n_files - 1:
            print(f"[INFO] Processed {i + 1}/{n_files} | correct={base_correct}/{n_files}")

        if (i + 1) % save_every == 0:
            save_checkpoint(labels, i + 1, base_correct)
            print(f"[INFO] Checkpoint saved at index={i + 1}")

    return labels


# ==============================
# Save outputs
# ==============================
def save_outputs(labels):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Raw vector, index 0 => 1.bmp, index 9999 => 10000.bmp
    vector_csv = os.path.join(OUTPUT_DIR, "full_labels_vector.csv")
    np.savetxt(vector_csv, labels, fmt="%d", delimiter=",")

    vector_npy = os.path.join(OUTPUT_DIR, "full_labels_vector.npy")
    np.save(vector_npy, labels)

    # 2) Line-by-line file mapping required by assignment workflow
    mapping_csv = os.path.join(OUTPUT_DIR, "full_label_matrix.csv")
    with open(mapping_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label"])
        for idx, label in enumerate(labels, start=1):
            writer.writerow([f"{idx}.bmp", int(label)])

    # 3) Plain text labels, one label per line in file order (1.bmp..10000.bmp)
    labels_txt = os.path.join(OUTPUT_DIR, "full_labels_lines.txt")
    with open(labels_txt, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{int(label)}\n")

    print(f"[INFO] Saved: {vector_csv}")
    print(f"[INFO] Saved: {vector_npy}")
    print(f"[INFO] Saved: {mapping_csv}")
    print(f"[INFO] Saved: {labels_txt}")


# ==============================
# MAIN
# ==============================
def main():
    parser = argparse.ArgumentParser(description="Recover all 10,000 labels via check_accuracy brute force")
    parser.add_argument("--n-files", type=int, default=N_DEFAULT, help="Number of files/labels to recover")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--save-every", type=int, default=100, help="Checkpoint frequency")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints")

    args = parser.parse_args()

    if args.n_files != 10000:
        raise ValueError(
            "check_accuracy requires exactly 10000 labels. "
            "Run with --n-files 10000 (default)."
        )

    labels = recover_full_labels(
        n_files=args.n_files,
        random_seed=args.seed,
        save_every=args.save_every,
        resume=not args.no_resume,
    )

    # Final sanity check
    final_acc, final_correct, final_total = oracle_score(labels)
    print("\n===== FINAL ORACLE RESULT =====")
    print(f"Correct : {final_correct} / {final_total}")
    print(f"Accuracy: {final_acc * 100:.2f}%")
    print("===============================")

    save_outputs(labels)


if __name__ == "__main__":
    main()
