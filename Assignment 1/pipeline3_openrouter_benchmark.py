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
REFERENCE_IMAGE_PATH = os.path.join(OUT_DIR, "reference_digits_0_to_9_labeled.png")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LOCKED_MODEL = "google/gemma-4-31b-it"


# ==============================
# ARGUMENTS
# ==============================
def parse_args():
    parser = argparse.ArgumentParser(description="OpenRouter benchmark on practical 500-label matrix")
    parser.add_argument(
        "--model",
        type=str,
        default=LOCKED_MODEL,
        help="OpenRouter model id",
    )
    parser.add_argument(
        "--allow-any-model",
        action="store_true",
        help="Allow running models other than the locked safe default",
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
        "--max-retries-timeout",
        type=int,
        default=3,
        help="Maximum retries on request timeout/network errors",
    )
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
        "--request-timeout",
        type=float,
        default=180.0,
        help="HTTP request timeout in seconds",
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
        "--reference-image-path",
        type=str,
        default=REFERENCE_IMAGE_PATH,
        help="Path to labeled reference image (0..9)",
    )
    parser.add_argument(
        "--api-key-env-var",
        type=str,
        default="OPENROUTER_API_KEY",
        help="Environment variable name to read OpenRouter API key from",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional seed for reproducible random sampling",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of images per API request",
    )
    parser.add_argument(
        "--compact-prompt",
        action="store_true",
        help="Use a shorter prompt to reduce token usage",
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


def build_prompt(compact=False):
    if compact:
        return (
            "Classify the handwritten Indian (Devanagari) digit in each image. "
            "The first image is a labeled reference for digits 0..9. "
            "Return digits only (0-9)."
        )

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


def build_prompt_for_batch(base_prompt, batch_size):
    if batch_size <= 1:
        return base_prompt

    return (
        base_prompt
        + " "
        + f"This request contains {batch_size} images. "
        + f"Return exactly {batch_size} digits in order as a comma-separated list. "
        + "Example output format: 3,1,9,0"
    )


def parse_batch_digits(text, expected_count):
    if text is None:
        return None

    cleaned = text.strip().translate(str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"))
    matches = re.findall(r"[0-9]", cleaned)
    if len(matches) < expected_count:
        return None

    return [int(x) for x in matches[:expected_count]]


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


def load_reference_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read reference image: {path}")

    ok, enc = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"Failed to encode reference image: {path}")

    return enc.tobytes()


def call_openrouter(headers, model, prompt, query_png_bytes_list, max_tokens, request_timeout, reference_png_bytes=None):
    content = [{"type": "text", "text": prompt}]
    if reference_png_bytes is not None:
        ref_b64 = base64.b64encode(reference_png_bytes).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{ref_b64}"},
            }
        )

    for query_png_bytes in query_png_bytes_list:
        query_b64 = base64.b64encode(query_png_bytes).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{query_b64}"},
            }
        )

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

    return requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=request_timeout,
    )


def extract_message_text(data):
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, "missing_choices"

    first = choices[0]
    if not isinstance(first, dict):
        return None, "invalid_choice_item"

    message = first.get("message")
    if not isinstance(message, dict):
        return None, "missing_message"

    raw = message.get("content")
    if raw is None:
        raw = message.get("reasoning")
    if raw is None:
        raw = ""
    return raw, ""


# ==============================
# MAIN
# ==============================
def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if not args.allow_any_model and args.model != LOCKED_MODEL:
        raise ValueError(
            f"Model '{args.model}' is blocked by safety lock. "
            f"Use '{LOCKED_MODEL}' or pass --allow-any-model to override."
        )
    # Force .env values to override any stale exported key in terminal session.
    load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

    api_key = os.getenv(args.api_key_env_var, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing {args.api_key_env_var} in .env")

    prompt = build_prompt(compact=args.compact_prompt)
    reference_png_bytes = None
    if args.reference_image_path:
        reference_png_bytes = load_reference_image(args.reference_image_path)

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
    batch_rows = []
    attempted = 0
    correct = 0
    parse_fail = 0
    errors = 0
    started = time.time()

    usage_prompt_tokens = 0
    usage_completion_tokens = 0
    usage_total_tokens = 0

    batch_id = 0
    for start in range(0, len(samples), args.batch_size):
        batch_id += 1
        batch = samples[start : start + args.batch_size]
        valid_items = []

        for index_0, true_digit in batch:
            image_path = os.path.join(DATA_DIR, f"{index_0 + 1}.bmp")
            try:
                png_bytes = preprocess_image(image_path, args.resize)
                valid_items.append((index_0, true_digit, png_bytes))
            except Exception as exc:
                errors += 1
                rows.append([args.model, index_0 + 1, true_digit, "", 0, "", f"image_error: {exc}", 0, 0, 0])

        if not valid_items:
            continue

        batch_file_numbers = [str(index_0 + 1) for index_0, _, _ in valid_items]
        batch_true_labels = [str(true_digit) for _, true_digit, _ in valid_items]
        batch_pred_labels = ["" for _ in valid_items]
        batch_raw = ""
        batch_error = ""
        batch_http_status = ""
        batch_attempted = 0
        batch_correct = 0
        batch_pt = 0
        batch_ct = 0
        batch_tt = 0

        request_prompt = build_prompt_for_batch(prompt, len(valid_items))

        tries = 0
        timeout_retries = 0
        response = None
        while True:
            tries += 1
            try:
                response = call_openrouter(
                    headers,
                    args.model,
                    request_prompt,
                    [x[2] for x in valid_items],
                    args.max_tokens,
                    args.request_timeout,
                    reference_png_bytes,
                )
            except requests.exceptions.RequestException as exc:
                timeout_retries += 1
                if timeout_retries <= args.max_retries_timeout:
                    wait_s = min(2.0 * timeout_retries, args.max_retry_wait)
                    print(
                        f"[WARN] network/timeout on batch {start + 1}-{min(start + len(batch), len(samples))}/{len(samples)} | "
                        f"retry {timeout_retries}/{args.max_retries_timeout} | waiting {wait_s:.1f}s | {type(exc).__name__}"
                    )
                    time.sleep(wait_s)
                    continue

                for index_0, true_digit, _ in valid_items:
                    errors += 1
                    rows.append(
                        [
                            args.model,
                            index_0 + 1,
                            true_digit,
                            "",
                            0,
                            "",
                            f"request_error: {type(exc).__name__}",
                            0,
                            0,
                            0,
                        ]
                    )
                response = None
                break

            if response.status_code == 429 and tries <= args.max_retries_429:
                retry_after = response.headers.get("retry-after", "5")
                try:
                    wait_s = float(retry_after)
                except Exception:
                    wait_s = 5.0
                wait_s = min(max(wait_s, 1.0), args.max_retry_wait)
                print(
                    f"[WARN] 429 rate-limit on batch {start + 1}-{min(start + len(batch), len(samples))}/{len(samples)} | "
                    f"retry {tries}/{args.max_retries_429} | waiting {wait_s:.1f}s"
                )
                time.sleep(wait_s)
                continue
            break

        if response is None:
            batch_error = "request_error"
            batch_rows.append(
                [
                    batch_id,
                    args.model,
                    len(valid_items),
                    "|".join(batch_file_numbers),
                    "|".join(batch_true_labels),
                    "|".join(batch_pred_labels),
                    batch_correct,
                    batch_attempted,
                    batch_error,
                    batch_http_status,
                    batch_raw,
                    batch_pt,
                    batch_ct,
                    batch_tt,
                ]
            )
            continue

        if response.status_code != 200:
            batch_error = f"http_{response.status_code}"
            batch_http_status = str(response.status_code)

            # If a multi-image request is rejected by provider validation,
            # retry each item individually instead of failing the whole batch.
            if response.status_code == 400 and len(valid_items) > 1:
                print(
                    f"[WARN] 400 on batch {start + 1}-{min(start + len(batch), len(samples))}/{len(samples)} | "
                    "retrying items one-by-one"
                )
                batch_error = "http_400_fallback"
                batch_raw = "fallback_to_single"

                for j, (index_0, true_digit, png_bytes) in enumerate(valid_items):
                    single_response = None
                    single_tries = 0
                    single_timeout_retries = 0

                    while True:
                        single_tries += 1
                        try:
                            single_response = call_openrouter(
                                headers,
                                args.model,
                                build_prompt_for_batch(prompt, 1),
                                [png_bytes],
                                args.max_tokens,
                                args.request_timeout,
                                reference_png_bytes,
                            )
                        except requests.exceptions.RequestException as exc:
                            single_timeout_retries += 1
                            if single_timeout_retries <= args.max_retries_timeout:
                                wait_s = min(2.0 * single_timeout_retries, args.max_retry_wait)
                                print(
                                    f"[WARN] network/timeout on fallback item {index_0 + 1} | "
                                    f"retry {single_timeout_retries}/{args.max_retries_timeout} | waiting {wait_s:.1f}s | {type(exc).__name__}"
                                )
                                time.sleep(wait_s)
                                continue

                            errors += 1
                            rows.append(
                                [
                                    args.model,
                                    index_0 + 1,
                                    true_digit,
                                    "",
                                    0,
                                    "",
                                    f"request_error: {type(exc).__name__}",
                                    0,
                                    0,
                                    0,
                                ]
                            )
                            single_response = None
                            break

                        if single_response.status_code == 429 and single_tries <= args.max_retries_429:
                            retry_after = single_response.headers.get("retry-after", "5")
                            try:
                                wait_s = float(retry_after)
                            except Exception:
                                wait_s = 5.0
                            wait_s = min(max(wait_s, 1.0), args.max_retry_wait)
                            print(
                                f"[WARN] 429 on fallback item {index_0 + 1} | "
                                f"retry {single_tries}/{args.max_retries_429} | waiting {wait_s:.1f}s"
                            )
                            time.sleep(wait_s)
                            continue
                        break

                    if single_response is None:
                        continue

                    if single_response.status_code != 200:
                        errors += 1
                        rows.append(
                            [
                                args.model,
                                index_0 + 1,
                                true_digit,
                                "",
                                0,
                                "",
                                f"http_{single_response.status_code}",
                                0,
                                0,
                                0,
                            ]
                        )
                        continue

                    try:
                        single_data = single_response.json()
                    except ValueError:
                        errors += 1
                        rows.append(
                            [
                                args.model,
                                index_0 + 1,
                                true_digit,
                                "",
                                0,
                                "",
                                "bad_json",
                                0,
                                0,
                                0,
                            ]
                        )
                        continue

                    single_raw, single_err = extract_message_text(single_data)
                    if single_err:
                        errors += 1
                        rows.append(
                            [
                                args.model,
                                index_0 + 1,
                                true_digit,
                                "",
                                0,
                                "",
                                f"bad_payload:{single_err}",
                                0,
                                0,
                                0,
                            ]
                        )
                        continue

                    single_usage = single_data.get("usage", {})
                    pt = int(single_usage.get("prompt_tokens", 0) or 0)
                    ct = int(single_usage.get("completion_tokens", 0) or 0)
                    tt = int(single_usage.get("total_tokens", 0) or 0)

                    usage_prompt_tokens += pt
                    usage_completion_tokens += ct
                    usage_total_tokens += tt
                    batch_pt += pt
                    batch_ct += ct
                    batch_tt += tt

                    attempted += 1
                    batch_attempted += 1

                    pred = parse_digit(single_raw)
                    if pred is None:
                        parse_fail += 1
                        rows.append([args.model, index_0 + 1, true_digit, "", 0, single_raw, "parse_fail", pt, ct, tt])
                    else:
                        batch_pred_labels[j] = str(pred)
                        is_correct = int(pred == true_digit)
                        correct += is_correct
                        batch_correct += is_correct
                        rows.append([args.model, index_0 + 1, true_digit, pred, is_correct, single_raw, "", pt, ct, tt])

                batch_rows.append(
                    [
                        batch_id,
                        args.model,
                        len(valid_items),
                        "|".join(batch_file_numbers),
                        "|".join(batch_true_labels),
                        "|".join(batch_pred_labels),
                        batch_correct,
                        batch_attempted,
                        batch_error,
                        batch_http_status,
                        batch_raw,
                        batch_pt,
                        batch_ct,
                        batch_tt,
                    ]
                )
            else:
                for index_0, true_digit, _ in valid_items:
                    errors += 1
                    rows.append([args.model, index_0 + 1, true_digit, "", 0, "", f"http_{response.status_code}", 0, 0, 0])
                batch_rows.append(
                    [
                        batch_id,
                        args.model,
                        len(valid_items),
                        "|".join(batch_file_numbers),
                        "|".join(batch_true_labels),
                        "|".join(batch_pred_labels),
                        batch_correct,
                        batch_attempted,
                        batch_error,
                        batch_http_status,
                        batch_raw,
                        batch_pt,
                        batch_ct,
                        batch_tt,
                    ]
                )
        else:
            try:
                data = response.json()
            except ValueError:
                batch_error = "bad_json"
                for index_0, true_digit, _ in valid_items:
                    errors += 1
                    rows.append([args.model, index_0 + 1, true_digit, "", 0, "", "bad_json", 0, 0, 0])
                batch_rows.append(
                    [
                        batch_id,
                        args.model,
                        len(valid_items),
                        "|".join(batch_file_numbers),
                        "|".join(batch_true_labels),
                        "|".join(batch_pred_labels),
                        batch_correct,
                        batch_attempted,
                        batch_error,
                        batch_http_status,
                        batch_raw,
                        batch_pt,
                        batch_ct,
                        batch_tt,
                    ]
                )
                processed = min(start + len(batch), len(samples))
                if processed % 10 == 0 or processed == len(samples):
                    print(
                        f"{processed}/{len(samples)} attempted={attempted} "
                        f"correct={correct} parse_fail={parse_fail} errors={errors}"
                    )
                if args.delay > 0:
                    time.sleep(args.delay)
                continue

            raw, payload_err = extract_message_text(data)
            if payload_err:
                batch_error = f"bad_payload:{payload_err}"
                for index_0, true_digit, _ in valid_items:
                    errors += 1
                    rows.append([args.model, index_0 + 1, true_digit, "", 0, "", batch_error, 0, 0, 0])
                batch_rows.append(
                    [
                        batch_id,
                        args.model,
                        len(valid_items),
                        "|".join(batch_file_numbers),
                        "|".join(batch_true_labels),
                        "|".join(batch_pred_labels),
                        batch_correct,
                        batch_attempted,
                        batch_error,
                        batch_http_status,
                        batch_raw,
                        batch_pt,
                        batch_ct,
                        batch_tt,
                    ]
                )
                processed = min(start + len(batch), len(samples))
                if processed % 10 == 0 or processed == len(samples):
                    print(
                        f"{processed}/{len(samples)} attempted={attempted} "
                        f"correct={correct} parse_fail={parse_fail} errors={errors}"
                    )
                if args.delay > 0:
                    time.sleep(args.delay)
                continue

            batch_raw = raw
            batch_http_status = str(response.status_code)

            usage = data.get("usage", {})
            pt = int(usage.get("prompt_tokens", 0) or 0)
            ct = int(usage.get("completion_tokens", 0) or 0)
            tt = int(usage.get("total_tokens", 0) or 0)
            batch_pt, batch_ct, batch_tt = pt, ct, tt

            usage_prompt_tokens += pt
            usage_completion_tokens += ct
            usage_total_tokens += tt

            attempted += len(valid_items)
            batch_attempted = len(valid_items)

            if len(valid_items) == 1:
                preds = [parse_digit(raw)]
            else:
                preds = parse_batch_digits(raw, len(valid_items))

            if preds is None:
                parse_fail += len(valid_items)
                batch_error = "parse_fail"
                for index_0, true_digit, _ in valid_items:
                    rows.append([args.model, index_0 + 1, true_digit, "", 0, raw, "parse_fail", pt, ct, tt])
            else:
                for j, ((index_0, true_digit, _), pred) in enumerate(zip(valid_items, preds)):
                    batch_pred_labels[j] = str(pred)
                    is_correct = int(pred == true_digit)
                    correct += is_correct
                    batch_correct += is_correct
                    rows.append([args.model, index_0 + 1, true_digit, pred, is_correct, raw, "", pt, ct, tt])

            batch_rows.append(
                [
                    batch_id,
                    args.model,
                    len(valid_items),
                    "|".join(batch_file_numbers),
                    "|".join(batch_true_labels),
                    "|".join(batch_pred_labels),
                    batch_correct,
                    batch_attempted,
                    batch_error,
                    batch_http_status,
                    batch_raw,
                    batch_pt,
                    batch_ct,
                    batch_tt,
                ]
            )

        processed = min(start + len(batch), len(samples))
        if processed % 10 == 0 or processed == len(samples):
            print(
                f"{processed}/{len(samples)} attempted={attempted} "
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

    out_root, out_ext = os.path.splitext(args.output)
    batch_out_path = os.path.join(OUT_DIR, f"{out_root}_batches{out_ext or '.csv'}")
    with open(batch_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "batch_id",
                "model",
                "batch_size",
                "file_numbers",
                "true_labels",
                "pred_labels",
                "batch_correct",
                "batch_attempted",
                "error",
                "http_status",
                "raw_response",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ]
        )
        writer.writerows(batch_rows)

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
    print(f"saved_batches={batch_out_path}")


if __name__ == "__main__":
    main()
