import os
import re
import csv
import time
import base64
import argparse

import cv2
import numpy as np
import requests
from dotenv import load_dotenv


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "Indian_Digits_Train")
MATRIX_PATH = os.path.join(BASE_DIR, "pipeline3", "label_matrix_500.csv")
OUT_DIR = os.path.join(BASE_DIR, "pipeline3")


def build_prompt(style: str):
    if style == "strict":
        return (
            "Identify the handwritten Indian digit in the image and return it as exactly one digit from 0 to 9. "
            "Output only one character, no words or punctuation."
        )

    # pdf style
    return (
        "Task: classify a handwritten Indian digit image into one class from 0 to 9. "
        "Classes are: 0,1,2,3,4,5,6,7,8,9. "
        "The input image is grayscale and may be low-resolution. "
        "Return only the predicted digit as a single character from 0 to 9. "
        "Do not return words, punctuation, markdown, or explanation."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Groq vision model with preprocessing")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        help="Groq model id",
    )
    parser.add_argument(
        "--samples-per-digit",
        type=int,
        default=2,
        help="Balanced sample count per class from label_matrix_500.csv",
    )
    parser.add_argument("--resize", type=int, default=224, help="Square resize edge length")
    parser.add_argument("--equalize", action="store_true", help="Apply histogram equalization")
    parser.add_argument(
        "--prompt-style",
        choices=["pdf", "strict"],
        default="pdf",
        help="Prompt template",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.2,
        help="Delay between requests in seconds",
    )
    parser.add_argument(
        "--max-retries-429",
        type=int,
        default=4,
        help="Maximum retries on HTTP 429",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="groq_benchmark_resize_results.csv",
        help="Output csv filename under pipeline3 folder",
    )
    return parser.parse_args()


def parse_digit(text: str):
    if text is None:
        return None

    text = text.strip()
    match = re.fullmatch(r"[0-9]", text)
    if match:
        return int(text)

    match = re.search(r"([0-9])", text)
    return int(match.group(1)) if match else None


def collect_balanced_samples(matrix_path: str, samples_per_digit: int):
    matrix = np.loadtxt(matrix_path, delimiter=",", dtype=int)
    if matrix.ndim == 1:
        matrix = matrix.reshape(10, -1)

    samples = []
    for digit in range(10):
        count = 0
        for idx in matrix[digit]:
            samples.append((int(idx), digit))
            count += 1
            if count >= samples_per_digit:
                break

    return samples


def preprocess_image(path: str, resize: int, equalize: bool):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
    if equalize:
        img = cv2.equalizeHist(img)

    ok, enc = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG: {path}")

    return enc.tobytes()


def call_groq(headers, model: str, prompt: str, png_bytes: bytes):
    b64 = base64.b64encode(png_bytes).decode("ascii")
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
    }

    return requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=90,
    )


def main():
    args = parse_args()
    load_dotenv(os.path.join(BASE_DIR, ".env"))

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in .env")

    prompt = build_prompt(args.prompt_style)
    samples = collect_balanced_samples(MATRIX_PATH, args.samples_per_digit)
    if not samples:
        raise RuntimeError("No samples found in matrix")

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

    for i, (index_0, true_digit) in enumerate(samples, start=1):
        image_path = os.path.join(DATA_DIR, f"{index_0 + 1}.bmp")
        try:
            png_bytes = preprocess_image(image_path, args.resize, args.equalize)
        except Exception as exc:
            errors += 1
            rows.append([args.model, index_0 + 1, true_digit, "", 0, "", f"image_error: {exc}"])
            continue

        tries = 0
        while True:
            tries += 1
            response = call_groq(headers, args.model, prompt, png_bytes)

            if response.status_code == 429 and tries <= args.max_retries_429:
                retry_after = response.headers.get("retry-after", "5")
                try:
                    wait_s = float(retry_after)
                except Exception:
                    wait_s = 5.0
                time.sleep(max(wait_s, 1.0))
                continue

            break

        if response.status_code != 200:
            errors += 1
            rows.append([args.model, index_0 + 1, true_digit, "", 0, "", f"http_{response.status_code}"])
        else:
            data = response.json()
            raw = data["choices"][0]["message"]["content"]
            pred = parse_digit(raw)
            attempted += 1

            if pred is None:
                parse_fail += 1
                rows.append([args.model, index_0 + 1, true_digit, "", 0, raw, "parse_fail"])
            else:
                is_correct = int(pred == true_digit)
                correct += is_correct
                rows.append([args.model, index_0 + 1, true_digit, pred, is_correct, raw, ""])

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
        writer.writerow([
            "model",
            "file_number",
            "true_label",
            "pred_label",
            "is_correct",
            "raw_response",
            "error",
        ])
        writer.writerows(rows)

    print("\n===== SUMMARY =====")
    print(f"model={args.model}")
    print(f"samples={len(samples)} ({args.samples_per_digit} per digit)")
    print(f"resize={args.resize}, equalize={args.equalize}, prompt={args.prompt_style}")
    print(f"attempted={attempted}, correct={correct}, accuracy={acc:.4f}")
    print(f"parse_fail={parse_fail}, errors={errors}, elapsed_sec={elapsed:.1f}")
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
