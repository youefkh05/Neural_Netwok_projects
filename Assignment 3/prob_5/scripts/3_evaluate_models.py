#!/usr/bin/env python3
"""
Model Evaluation Script
Evaluates Arabic text using multiple LLMs via OpenRouter and Ollama.
Processes one sentence at a time.
"""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    exit(1)

DATA_DIR = Path(__file__).parent.parent / "data"
LOGS_DIR = Path(__file__).parent.parent / "logs"
INPUT_FILE = DATA_DIR / "to_annotate.csv"
OUTPUT_FILE = LOGS_DIR / "evaluation.json"

MAX_ITEMS = int(os.environ.get("MAX_ITEMS", "100"))
OLLAMA_TIMEOUT = 180

SYSTEM_PROMPT = "You are an Arabic toxicity detector. Analyze the following text. Is the text toxic, hostile, or offensive? Your ENTIRE response MUST be ONLY a single JSON object: {is_toxic: yes} or {is_toxic: no}. Do not output any other text, explanation, or conversational filler."

USER_PROMPT = 'Text: "{text}"'

OPENROUTER_MODELS = {
    "chatgpt": "openai/gpt-4o-mini",
    "gemini": "google/gemini-2.5-flash",
}

OLLAMA_MODELS = ["iKhalid/ALLaM:7b", "jwnder/jais-adaptive:7b", "yasserrmd/Fanar-1-9B-Instruct"]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/api/chat"


def load_data():
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found: {INPUT_FILE}")
        return []
    
    with open(INPUT_FILE, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_toxicity_output(text: str) -> dict:
    import re
    
    try:
        match = re.search(r'{"is_toxic"[^}]*}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    
    text_lower = text.lower().strip()
    
    if 'yes' in text_lower or 'نعم' in text_lower:
        return {"is_toxic": "yes"}
    
    if 'no' in text_lower or 'لا' in text_lower:
        return {"is_toxic": "no"}
    
    return {"is_toxic": "parse_error", "raw": text}


def call_openrouter(model_id: str, user_prompt: str, api_key: str):
    if not api_key:
        return {"is_toxic": "error", "error": "OPENROUTER_API_KEY not set"}
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/arabic-eval",
        "X-Title": "Arabic Toxicity Evaluation"
    }
    
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 50
    }
    
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            raw_content = data["choices"][0]["message"]["content"]
            return parse_toxicity_output(raw_content)
        else:
            return {"is_toxic": "error", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"is_toxic": "error", "error": "Timeout"}
    except Exception as e:
        return {"is_toxic": "error", "error": str(e)}


def call_ollama(model: str, user_prompt: str):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            raw_content = data.get("message", {}).get("content", "")
            return parse_toxicity_output(raw_content)
        else:
            return {"is_toxic": "error", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"is_toxic": "error", "error": "Ollama not running"}
    except requests.exceptions.Timeout:
        return {"is_toxic": "error", "error": "Timeout"}
    except Exception as e:
        return {"is_toxic": "error", "error": str(e)}


def save_results(results: list, metadata: dict):
    output_data = {
        "metadata": metadata,
        "results": results
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def evaluate_item(item: dict, api_key: str) -> dict:
    text_id = item["id"]
    text = item["text"]
    user_prompt = USER_PROMPT.format(text=text)
    
    result = {
        "id": text_id,
        "text": text,
        "variety": item["variety"],
        "timestamp": datetime.now().isoformat(),
        "outputs": {}
    }
    
    for name, model_id in OPENROUTER_MODELS.items():
        output = call_openrouter(model_id, user_prompt, api_key)
        result["outputs"][name] = output
        time.sleep(0.3)
    
    for model in OLLAMA_MODELS:
        output = call_ollama(model, user_prompt)
        result["outputs"][model] = output
        time.sleep(0.1)
    
    return result


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    
    print("Loading data...")
    data = load_data()
    if not data:
        print("No data to process.")
        return
    
    data = data[:MAX_ITEMS]
    print(f"Processing {len(data)} items one at a time...")
    
    all_results = []
    
    metadata = {
        "start_time": datetime.now().isoformat(),
        "total_items": len(data),
        "models": {
            "openrouter": {name: model_id for name, model_id in OPENROUTER_MODELS.items()},
            "ollama": OLLAMA_MODELS
        }
    }
    
    for i, item in enumerate(data, 1):
        print(f"  [{i}/{len(data)}] ID {item['id']}...", end=" ", flush=True)
        result = evaluate_item(item, api_key)
        all_results.append(result)
        save_results(all_results, metadata)
        print("done")
    
    metadata["end_time"] = datetime.now().isoformat()
    save_results(all_results, metadata)
    
    print(f"Completed: {len(all_results)} results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
