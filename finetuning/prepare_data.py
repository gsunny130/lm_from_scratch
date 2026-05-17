# prepare_data.py
# Converts your essays CSV into a JSONL file ready for fine-tuning.
#
# Why JSONL? Each line is one JSON object (one essay). This format is
# standard for LLM training — easy to stream, easy to debug, one example per line.

import pandas as pd  # For reading the CSV
import json          # For writing JSONL
import os            # For file path operations

# --------------------------------------------------------------------------
# 1. Load the CSV
# --------------------------------------------------------------------------
# pd.read_csv reads your file into a DataFrame — think of it as a table in memory.
df = pd.read_csv("essays.csv")

print(f"Loaded {len(df)} essays.")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 100 chars of essay 1:\n{df['text'][0][:100]}\n")

# --------------------------------------------------------------------------
# 2. Basic cleaning
# --------------------------------------------------------------------------
# Drop any rows where the text column is empty (NaN = "not a number", pandas'
# way of representing missing values).
df = df.dropna(subset=["text"])

# Strip leading/trailing whitespace from each essay.
# .str accessor lets you apply string methods to a whole column at once.
df["text"] = df["text"].str.strip()

# Drop essays that are suspiciously short (under 100 characters).
# A real essay won't be that short — these are likely data errors.
df = df[df["text"].str.len() >= 100]

print(f"After cleaning: {len(df)} essays remain.")

# --------------------------------------------------------------------------
# 3. Format each essay into a training prompt
# --------------------------------------------------------------------------
# LLMs learn by predicting the next token. We wrap each essay in a prompt
# template so the model learns: "when I see this instruction, write an essay."
#
# This specific format (### Instruction / ### Response) is what Mistral-7B
# was instruction-tuned with, so it recognizes it naturally.

def format_essay(text):
    """Wrap a single essay in the instruction-following template."""
    return (
        "### Instruction:\n"
        "Write a compelling college application essay.\n\n"
        "### Response:\n"
        f"{text}"
    )

df["formatted"] = df["text"].apply(format_essay)

# --------------------------------------------------------------------------
# 4. Save as JSONL
# --------------------------------------------------------------------------
# We write one JSON object per line. Each object has a single key "text"
# containing the formatted essay. This is the format expected by the
# Hugging Face `trl` library's SFTTrainer (Supervised Fine-Tuning Trainer).

output_path = "essays_formatted.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for essay in df["formatted"]:
        # json.dumps() converts a Python dict to a JSON string.
        # ensure_ascii=False preserves any non-ASCII characters (accents, etc.)
        json_line = json.dumps({"text": essay}, ensure_ascii=False)
        f.write(json_line + "\n")  # Each example on its own line

print(f"\nSaved {len(df)} formatted essays to: {output_path}")
print("\nSample formatted essay (first 300 chars):")
print(df["formatted"].iloc[0][:300])
