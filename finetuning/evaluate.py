# evaluate.py
# Compares base Mistral-7B-v0.1 vs. your fine-tuned LoRA adapter.
#
# Three types of evaluation:
#   1. PERPLEXITY   — statistical measure on the held-out TEST split only
#                     (not training data — that would be cheating)
#   2. QUALITATIVE  — side-by-side text generation on fixed prompts
#   3. CLAUDE SCORE — Claude Sonnet judges each output on narrative quality,
#                     specificity, and emotional resonance (1–5 each)
#
# Run on Lambda AFTER training:
#   ~/env/bin/python evaluate.py
#
# Claude scoring requires ANTHROPIC_API_KEY in environment:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   ~/env/bin/python evaluate.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from data_utils import get_splits
from eval_utils import compute_perplexity, score_outputs


# --------------------------------------------------------------------------
# 0. Config
# --------------------------------------------------------------------------

BASE_MODEL   = "mistralai/Mistral-7B-v0.1"
ADAPTER_PATH = "./mistral-essays-finetuned"
DATA_FILE    = "./essays_formatted.jsonl"

# Fixed prompts for qualitative comparison.
# These are the exact opening lines the two models will continue.
EVAL_PROMPTS = [
    "It",
    "January",
    "Harry",
    "I",
]

MAX_NEW_TOKENS = 250   # tokens to generate per prompt


# --------------------------------------------------------------------------
# 1. Model loading helpers
# --------------------------------------------------------------------------

def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    return tok


def bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_base_model():
    print("Loading base model (no adapter)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config(),
        device_map={"": 0},        # force everything to GPU 0
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_finetuned_model(base_model):
    print("Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model


# --------------------------------------------------------------------------
# 2. Generation helpers
# --------------------------------------------------------------------------

def format_prompt(opening_line):
    """Same template used during training."""
    return (
        "### Instruction:\n"
        "Write a compelling college application essay.\n\n"
        "### Response:\n"
        f"{opening_line}"
    )


def generate(model, tokenizer, opening_line):
    prompt = format_prompt(opening_line)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_output.split("### Response:\n")[-1].strip()


# --------------------------------------------------------------------------
# 3. Main
# --------------------------------------------------------------------------

def main():
    tokenizer = load_tokenizer()

    # ---- Load the test split (same fixed split used by train.py) ----
    # test_dataset contains essays the model never saw during training.
    # Using training data for perplexity would give an overly optimistic score
    # (the model has partially memorised those examples).
    _, _, test_dataset = get_splits(DATA_FILE)
    test_texts = [ex["text"] for ex in test_dataset]
    print(f"Test split: {len(test_texts)} essays (held-out, never seen during training)")

    # ---- Load base model once, then layer the adapter on top ----
    base_model = load_base_model()

    # ---- PERPLEXITY: base model ----
    base_perp = None
    if test_texts:
        print("\nComputing perplexity on base model (test split)...")
        base_perp = compute_perplexity(base_model, tokenizer, test_texts)
        print(f"  Base model perplexity:       {base_perp:.2f}")

    # ---- QUALITATIVE: base model ----
    print("\n" + "=" * 70)
    print("QUALITATIVE COMPARISON — BASE MODEL (no fine-tuning)")
    print("=" * 70)
    base_outputs = []
    for prompt in EVAL_PROMPTS:
        out = generate(base_model, tokenizer, prompt)
        base_outputs.append(out)

    # ---- Swap to fine-tuned model ----
    finetuned_model = load_finetuned_model(base_model)

    # ---- PERPLEXITY: fine-tuned model ----
    ft_perp = None
    if test_texts:
        print("\nComputing perplexity on fine-tuned model (test split)...")
        ft_perp = compute_perplexity(finetuned_model, tokenizer, test_texts)
        print(f"  Fine-tuned model perplexity: {ft_perp:.2f}")

    # ---- QUALITATIVE: fine-tuned model ----
    print("\n" + "=" * 70)
    print("QUALITATIVE COMPARISON — FINE-TUNED MODEL (LoRA adapter)")
    print("=" * 70)
    ft_outputs = []
    for prompt in EVAL_PROMPTS:
        out = generate(finetuned_model, tokenizer, prompt)
        ft_outputs.append(out)

    # ---- CLAUDE SCORING ----
    # Requires ANTHROPIC_API_KEY. Gracefully skipped if the key is absent.
    print("\n" + "=" * 70)
    print("CLAUDE SCORING (narrative quality / specificity / emotional resonance / coherence / task completion)")
    print("=" * 70)
    print("\nScoring base model outputs...")
    base_scores = score_outputs(base_outputs, "base")

    print("\nScoring fine-tuned model outputs...")
    ft_scores = score_outputs(ft_outputs, "fine-tuned")

    # ---- Print results ----
    print("\n\n" + "#" * 70)
    print("RESULTS SUMMARY")
    print("#" * 70)

    # Perplexity table
    if base_perp is not None and ft_perp is not None:
        improvement = ((base_perp - ft_perp) / base_perp) * 100
        print(f"\nPerplexity on test split (lower is better):")
        print(f"  Base model:       {base_perp:.2f}")
        print(f"  Fine-tuned model: {ft_perp:.2f}")
        if ft_perp < base_perp:
            print(f"  => Fine-tuned is {improvement:.1f}% lower — model fits essay style better.")
        else:
            print(f"  => Fine-tuned perplexity is HIGHER — possible overfitting or prompt mismatch.")

    # Claude scores table
    if base_scores and ft_scores:
        dims = ["narrative_quality", "specificity", "emotional_resonance", "coherence", "task_completion"]
        col_w = 22
        print(f"\nClaude scores (avg over {len(EVAL_PROMPTS)} prompts, 1–5 each):")
        print(f"  {'Dimension':<{col_w}} {'Base':>6}  {'Fine-tuned':>10}  {'Delta':>7}")
        print(f"  {'-'*col_w} {'------':>6}  {'----------':>10}  {'-------':>7}")
        for dim in dims:
            b = base_scores[dim]
            f = ft_scores[dim]
            delta = f - b
            sign = "+" if delta >= 0 else ""
            label = dim.replace("_", " ").title()
            print(f"  {label:<{col_w}} {b:>6.2f}  {f:>10.2f}  {sign}{delta:>6.2f}")

    # Side-by-side generation
    print("\nSide-by-side generation:")
    for i, prompt in enumerate(EVAL_PROMPTS):
        print(f"\n{'─' * 70}")
        print(f"PROMPT {i+1}: {prompt}")
        print(f"{'─' * 70}")
        print(f"\n[BASE MODEL]\n{base_outputs[i]}\n")
        print(f"\n[FINE-TUNED]\n{ft_outputs[i]}\n")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
