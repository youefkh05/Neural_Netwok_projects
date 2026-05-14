#!/usr/bin/env python3
"""
Create Annotation Template
Converts raw_data.csv to to_annotate.csv with annotation columns.
"""

import csv
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_FILE = DATA_DIR / "raw_data.csv"
OUTPUT_FILE = DATA_DIR / "to_annotate.csv"


def main():
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found: {INPUT_FILE}")
        return
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Read {len(rows)} rows from {INPUT_FILE}")
    
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "variety", "annotator1", "annotator2", "gold_label"])
        
        for row in rows:
            writer.writerow([
                row["id"],
                row["text"],
                row["variety"],
                "",
                "",
                ""
            ])
    
    print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
