import os
import re
import csv
import time
import base64
import argparse
import unicodedata

import cv2
import numpy as np
import requests
from dotenv import load_dotenv


# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "Indian_Digits_Train")
MATRIX_PATH = os.path.join(BASE_DIR, "pipeline3", "label_matrix_500.csv")
OUT_DIR = os.path.join(BASE_DIR, "pipeline3")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ==============================
# ARGUMENTS
# ==============================
def parse_args():
    parser = argparse.ArgumentParser(description="OpenRouter benchmark on practical 500-label matrix")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-2.0-flash-001",
        help="OpenRouter model id",
    )
    parser.add_argument(
        "--samples-per-digit",
        type=int,
        default=2,
        help="Balanced sample count per class from label_matrix_500.csv",
    )
    parser.add_argument("--resize", type=int, default=224, help="Square resize edge length")
    parser.add_argument("--delay", type=float, default=0.4, help="Delay in seconds between requests")
    parser.add_argument("--max-retries-429", type=int, default=4, help="Maximum retries on HTTP 429")
    parser.add_argument(
        "--max-retry-wait",
        type=float,
        default=8.0,
        help="Maximum seconds to wait on a single 429 retry",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Maximum completion tokens per request",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="openrouter_benchmark_results.csv",
        help="Output CSV file name under pipeline3/",
    )
    parser.add_argument(
        "--matrix-path",
        type=str,
        default=MATRIX_PATH,
        help="Path to label matrix CSV",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional seed for reproducible random sampling",
    )
    return parser.parse_args()


# ==============================
# HELPERS
# ==============================
def parse_digit(text):
    if text is None:
        return None

    text = text.strip()

    # Fast path for plain ASCII responses like "7".
    m = re.fullmatch(r"[0-9]", text)
    if m:
        return int(text)

    # Normalize Arabic-Indic / Eastern Arabic-Indic digits to ASCII.
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"))

    m = re.search(r"([0-9])", text)
    if m:
        return int(m.group(1))

    # Last-resort unicode numeric parsing for any decimal digit character.
    for ch in text:
        try:
            d = unicodedata.digit(ch)
            if 0 <= d <= 9:
                return int(d)
        except Exception:
            continue

    return None


def build_prompt():
    return (
        "Task: Classify the handwritten Indian (Hindi/Devanagari) digit in the provided image. "
        "Image Context: This image is an upscaled version of a 28x28 grayscale handwritten digit. "
        "The digit belongs to the Indian (Hindi/Devanagari) numbering system (٠,١,٢,٣,٤,٥,٦,٧,٨,٩). "
        "Critical Instruction - Confusing Pairs: These are Indian digits, not Western/Arabic numerals. "
        "Pay extra attention to shapes that look like Western numbers but have different meanings in this script: "
        "٤ (Indian 4) can look like a Western 3. "
        "٥ (Indian 5) can look like a Western 0. "
        "٦ (Indian 6) can look like a Western 7. "
        "Output Requirement: Respond with only the predicted digit as a single integer (0-9). "
        "Do not include any text, labels, punctuation, or explanations in your response. "
        "Response Example: 5"
    )


def collect_balanced_samples(matrix_path, samples_per_digit, random_seed=None):
    matrix = np.loadtxt(matrix_path, delimiter=",", dtype=int)
    if matrix.ndim == 1:
        matrix = matrix.reshape(10, -1)

    rng = np.random.default_rng(seed=random_seed)

    samples = []
    for digit in range(10):
        class_indices = np.array(matrix[digit], dtype=int)
        if samples_per_digit > len(class_indices):
            raise ValueError(
                f"samples_per_digit={samples_per_digit} exceeds available references for class {digit} ({len(class_indices)})"
            )

        picked = rng.choice(class_indices, size=samples_per_digit, replace=False)
        for idx in picked:
            samples.append((int(idx), digit))

    rng.shuffle(samples)

    return samples


def preprocess_image(path, resize):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
    ok, enc = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG: {path}")

    return enc.tobytes()


def call_openrouter(headers, model, prompt, query_png_bytes, max_tokens):
    query_b64 = base64.b64encode(query_png_bytes).decode("ascii")
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{query_b64}"},
        },
    ]

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }

    if not model.startswith("openai/"):
        payload["temperature"] = 0

    return requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)


# ==============================
# MAIN
# ==============================
def main():
    args = parse_args()
    # Force .env values to override any stale exported key in terminal session.
    load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in .env")

    prompt = build_prompt()
    samples = collect_balanced_samples(
        args.matrix_path,
        args.samples_per_digit,
        random_seed=args.random_seed,
    )
    if not samples:
        raise RuntimeError("No samples found in label matrix")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    rows = []
    attempted = 0
    correct = 0
    parse_fail = 0
    errors = 0
    started = time.time()

    usage_prompt_tokens = 0
    usage_completion_tokens = 0
    usage_total_tokens = 0

    for i, (index_0, true_digit) in enumerate(samples, start=1):
        image_path = os.path.join(DATA_DIR, f"{index_0 + 1}.bmp")

        try:
            png_bytes = preprocess_image(image_path, args.resize)
        except Exception as exc:
            errors += 1
            rows.append([args.model, index_0 + 1, true_digit, "", 0, "", f"image_error: {exc}", 0, 0, 0])
            continue

        tries = 0
        while True:
            tries += 1
            response = call_openrouter(
                headers,
                args.model,
                prompt,
                png_bytes,
                args.max_tokens,
            )

            if response.status_code == 429 and tries <= args.max_retries_429:
                retry_after = response.headers.get("retry-after", "5")
                try:
                    wait_s = float(retry_after)
                except Exception:
                    wait_s = 5.0
                wait_s = min(max(wait_s, 1.0), args.max_retry_wait)
                print(
                    f"[WARN] 429 rate-limit on item {i}/{len(samples)} | "
                    f"retry {tries}/{args.max_retries_429} | waiting {wait_s:.1f}s"
                )
                time.sleep(wait_s)
                continue
            break

        if response.status_code != 200:
            errors += 1
            rows.append([args.model, index_0 + 1, true_digit, "", 0, "", f"http_{response.status_code}", 0, 0, 0])
        else:
            data = response.json()
            message = data["choices"][0]["message"]
            raw = message.get("content")
            if raw is None:
                raw = message.get("reasoning")
            if raw is None:
                raw = ""
            pred = parse_digit(raw)
            attempted += 1

            usage = data.get("usage", {})
            pt = int(usage.get("prompt_tokens", 0) or 0)
            ct = int(usage.get("completion_tokens", 0) or 0)
            tt = int(usage.get("total_tokens", 0) or 0)

            usage_prompt_tokens += pt
            usage_completion_tokens += ct
            usage_total_tokens += tt

            if pred is None:
                parse_fail += 1
                rows.append([args.model, index_0 + 1, true_digit, "", 0, raw, "parse_fail", pt, ct, tt])
            else:
                is_correct = int(pred == true_digit)
                correct += is_correct
                rows.append([args.model, index_0 + 1, true_digit, pred, is_correct, raw, "", pt, ct, tt])

        if i % 10 == 0 or i == len(samples):
            print(
                f"{i}/{len(samples)} attempted={attempted} "
                f"correct={correct} parse_fail={parse_fail} errors={errors}"
            )

        if args.delay > 0:
            time.sleep(args.delay)

    elapsed = time.time() - started
    acc = (correct / attempted) if attempted else 0.0

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, args.output)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "file_number",
                "true_label",
                "pred_label",
                "is_correct",
                "raw_response",
                "error",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ]
        )
        writer.writerows(rows)

    print("\n===== SUMMARY =====")
    print(f"model={args.model}")
    print(f"samples={len(samples)} ({args.samples_per_digit} per digit)")
    print(f"resize={args.resize}")
    print(f"attempted={attempted}, correct={correct}, accuracy={acc:.4f}")
    print(f"parse_fail={parse_fail}, errors={errors}, elapsed_sec={elapsed:.1f}")
    print(
        "usage_tokens="
        f"prompt={usage_prompt_tokens}, completion={usage_completion_tokens}, total={usage_total_tokens}"
    )
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
