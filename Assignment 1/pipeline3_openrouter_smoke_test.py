import os
import re
import base64
import argparse
import unicodedata

import cv2
import requests
from dotenv import load_dotenv


# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "Indian_Digits_Train")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ==============================
# ARGUMENTS
# ==============================
def parse_args():
    parser = argparse.ArgumentParser(description="Single-image OpenRouter smoke test for Pipeline 3")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-2.0-flash-001",
        help="OpenRouter model id",
    )
    parser.add_argument(
        "--image-number",
        type=int,
        default=1,
        help="Image file number from Indian_Digits_Train (1..10000)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=224,
        help="Square size to upscale before sending",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate env and image processing only; do not call API",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Maximum completion tokens per request",
    )
    return parser.parse_args()


# ==============================
# HELPERS
# ==============================
def parse_digit(text):
    if text is None:
        return None

    text = text.strip()

    m = re.fullmatch(r"[0-9]", text)
    if m:
        return int(text)

    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"))

    m = re.search(r"([0-9])", text)
    if m:
        return int(m.group(1))

    for ch in text:
        try:
            d = unicodedata.digit(ch)
            if 0 <= d <= 9:
                return int(d)
        except Exception:
            continue

    return None


def preprocess_image(path, resize):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")

    return encoded.tobytes()


def build_payload(model, query_png_bytes, max_tokens):
    prompt = (
        "Task: Classify the handwritten Indian (Hindi/Devanagari) digit in the provided image. "
        "Image Context: This image is an upscaled version of a 28x28 grayscale handwritten digit. "
        "The digit belongs to the Indian (Hindi/Devanagari) numbering system (0, 1, 2, 3, 4, 5, 6, 7, 8, or 9). "
        "Critical Instruction - Confusing Pairs: These are Indian digits, not Western/Arabic numerals. "
        "Pay extra attention to shapes that look like Western numbers but have different meanings in this script: "
        "4 (Indian 4) can look like a Western 3. "
        "5 (Indian 5) can look like a Western 0. "
        "6 (Indian 6) can look like a Western 7. "
        "Output Requirement: Respond with only the predicted digit as a single integer (0-9). "
        "Do not include any text, labels, punctuation, or explanations in your response. "
        "Response Example: 5"
    )

    content = [{"type": "text", "text": prompt}]

    query_b64 = base64.b64encode(query_png_bytes).decode("ascii")
    content.append({"type": "text", "text": "Target image to classify:"})
    content.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{query_b64}"},
        }
    )

    return {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }


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

    image_number = args.image_number
    if image_number < 1 or image_number > 10000:
        raise ValueError("--image-number must be in [1, 10000]")

    image_path = os.path.join(DATA_DIR, f"{image_number}.bmp")
    png_bytes = preprocess_image(image_path, args.resize)

    print(f"[INFO] model={args.model}")
    print(f"[INFO] image={image_path}")
    print(f"[INFO] resize={args.resize}")

    if args.dry_run:
        print("[INFO] Dry run OK. API call skipped.")
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = build_payload(args.model, png_bytes, args.max_tokens)
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")

    data = response.json()
    raw = data["choices"][0]["message"]["content"]
    pred = parse_digit(raw)

    print(f"[INFO] raw_response={raw!r}")
    print(f"[INFO] parsed_digit={pred}")

    if pred is None:
        print("[WARNING] Could not parse a digit from model response.")


if __name__ == "__main__":
    main()
