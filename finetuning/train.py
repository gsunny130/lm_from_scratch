
# train.py
# Fine-tunes Mistral-7B on your essays using QLoRA.
#
# QLoRA = Quantized Low-Rank Adaptation
#   - "Quantized": the base model weights are compressed to 4-bit integers,
#     which cuts memory usage by ~75%. This lets a 7B model fit on a 24GB GPU.
#   - "Low-Rank Adaptation (LoRA)": instead of updating all 7 billion weights,
#     we insert small trainable matrices ("adapters") into the model. Only these
#     adapters are trained — the base model stays frozen.
#   - Result: we train ~0.1% of the parameters, saving memory and compute,
#     while still meaningfully shifting the model's behavior.

import argparse
import json
import os

import torch                             # PyTorch — the deep learning framework everything runs on
from transformers import (
    AutoModelForCausalLM,                # Loads the Mistral model
    AutoTokenizer,                       # Loads the matching tokenizer
    BitsAndBytesConfig,                  # Configures 4-bit quantization
)
from peft import LoraConfig, get_peft_model, TaskType  # LoRA adapter setup
from trl import SFTTrainer, SFTConfig    # SFTConfig replaces TrainingArguments in trl 1.x

from data_utils import get_splits        # shared 80/10/10 split — same split used in evaluate.py
from eval_utils import compute_perplexity


# --------------------------------------------------------------------------
# 1. Configuration — constants that don't change between sweep runs
# --------------------------------------------------------------------------

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
# The base model we're starting from. Hugging Face will download it (~14GB).
# It's stored in ~/.cache/huggingface after the first download.

DATA_FILE = "essays_formatted.jsonl"
# The file we created with prepare_data.py


# --------------------------------------------------------------------------
# 2. Command-line arguments
# --------------------------------------------------------------------------
# Defaults match the original hand-tuned config.
# sweep.py overrides these to explore different hyperparameter combinations.

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning of Mistral-7B on college essays")
    parser.add_argument(
        "--r", type=int, default=8,
        help="LoRA rank — size of the low-rank adapter matrices (default: 8)",
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=16,
        help="LoRA alpha — scaling factor for the adapter update (default: 16 = 2*r)",
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.05,
        help="LoRA dropout — fraction of adapter weights zeroed during training (default: 0.05)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./mistral-essays-finetuned",
        help="Directory to save the trained adapter and tokenizer",
    )
    parser.add_argument(
        "--metrics-file", type=str, default=None,
        help="If set, write train/eval/perplexity metrics as JSON to this path (used by sweep.py)",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# 3. Main training function
# --------------------------------------------------------------------------

def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # 3a. Load and split the dataset
    # -----------------------------------------------------------------------
    # get_splits() gives us an 80 / 10 / 10 train / val / test split with a
    # fixed seed. We only use train + val here. The test split is held out
    # entirely during training — we only touch it at the very end to measure
    # final perplexity on data the model has never seen.

    train_dataset, val_dataset, test_dataset = get_splits(DATA_FILE)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples:   {len(val_dataset)}")
    print(f"Test examples:  {len(test_dataset)}  (held out — not used during training)")

    # -----------------------------------------------------------------------
    # 3b. Tokenizer
    # -----------------------------------------------------------------------
    # A tokenizer converts raw text into numbers (token IDs) that the model
    # understands. "The cat sat" might become [450, 5255, 6348].
    # Every model has its own tokenizer — you must use the matching one.

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Mistral's tokenizer doesn't define a padding token by default.
    # Padding is needed when batching examples of different lengths together —
    # shorter examples get padded to match the longest in the batch.
    # We use the end-of-sequence token as the pad token (common practice).
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Pad on the right side of sequences

    # -----------------------------------------------------------------------
    # 3c. Quantization config (the "Q" in QLoRA)
    # -----------------------------------------------------------------------
    # This tells the model loader to compress base model weights to 4-bit.
    # The model itself never trains in 4-bit — computations are upcast to
    # bfloat16 (a 16-bit float format) on the fly. This is handled automatically.

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,               # Load model in 4-bit precision
        bnb_4bit_quant_type="nf4",       # "NormalFloat4" — best quantization type for LLMs
        bnb_4bit_compute_dtype=torch.bfloat16,  # Upcast to bfloat16 for math ops
        bnb_4bit_use_double_quant=True,  # Quantize the quantization constants too (saves a bit more memory)
    )

    # -----------------------------------------------------------------------
    # 3d. Load the base model
    # -----------------------------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,  # Apply 4-bit quantization
        device_map="auto",               # Automatically place model layers on available GPUs
        trust_remote_code=True,
    )

    # This is needed when using gradient checkpointing with quantized models.
    # Gradient checkpointing trades compute for memory: instead of storing all
    # intermediate activations (needed for backprop), it recomputes them on demand.
    model.config.use_cache = False
    model.enable_input_require_grads()  # ensures gradients flow properly into LoRA adapters

    # -----------------------------------------------------------------------
    # 3e. LoRA config (the "LoRA" in QLoRA)
    # -----------------------------------------------------------------------
    # LoRA inserts two small matrices (A and B) into specific layers of the model.
    # Instead of updating a weight matrix W directly, we learn: W + A*B
    # where A and B are much smaller. Their product approximates the update to W.

    lora_config = LoraConfig(
        r=args.r,
        # "Rank" of the LoRA matrices. Higher = more parameters to train = more
        # expressive but more prone to overfitting on small datasets.

        lora_alpha=args.lora_alpha,
        # Scaling factor for the LoRA update. The effective update is scaled by
        # lora_alpha / r. Setting alpha = 2*r is a common heuristic.

        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # Which layers to insert LoRA adapters into. These are the attention
        # projection layers (query, key, value, output) — the core of the
        # transformer's attention mechanism. This is where style is encoded.

        lora_dropout=args.lora_dropout,
        # Randomly zero out this fraction of LoRA weights during training.
        # Regularization — prevents overfitting by forcing the model to
        # not rely on any single weight too heavily.

        bias="none",
        # Don't train bias terms — not necessary for fine-tuning.

        task_type=TaskType.CAUSAL_LM,
        # We're doing causal language modeling: predict the next token.
    )

    model = get_peft_model(model, lora_config)
    # Wraps the model: freezes all base weights, adds trainable LoRA adapters.

    model.print_trainable_parameters()
    # Prints something like: "trainable params: 8,388,608 || all params: 3,752,071,168 || trainable%: 0.22"
    # This confirms only ~0.2% of parameters are actually being trained.

    # -----------------------------------------------------------------------
    # 3f. SFTConfig — training arguments (trl 1.x style)
    # -----------------------------------------------------------------------
    # In trl 1.x, SFTConfig replaces TrainingArguments and absorbs all the
    # SFT-specific settings that used to be passed directly to SFTTrainer
    # (max_seq_length, dataset_text_field, packing). Everything lives in one place.

    training_args = SFTConfig(
        output_dir=args.output_dir,

        num_train_epochs=3,
        # How many times to loop through the full training set.
        # 3 epochs is a safe starting point for a small dataset.

        per_device_train_batch_size=2,
        # How many essays to process at once per GPU. Small because essays are long.

        gradient_accumulation_steps=4,
        # Instead of updating weights every 2 examples, accumulate gradients
        # over 4 steps, then update. Effective batch size = 2 * 4 = 8.
        # This simulates a larger batch without needing more GPU memory.

        learning_rate=2e-4,
        # How big each weight update step is. 2e-4 (0.0002) is standard for LoRA.
        # this is the max learning rate, ramps up linearly during warm_up steps

        lr_scheduler_type="cosine",
        # The learning rate isn't constant — it follows a cosine curve, starting
        # at learning_rate, warming up, then gradually decaying to near zero.
        # This helps the model converge smoothly.

        warmup_steps=4,
        # Linearly ramp up the learning rate for the first 4 steps.
        # Starting with a small LR prevents early instability.

        fp16=False,
        bf16=True,
        # Use bfloat16 mixed precision for training computations (not the model weights).
        # bfloat16 has a wider dynamic range than float16 and is more stable for training.
        # fp16 and bf16 can't both be True.

        logging_steps=3,
        # Print a training loss update every 3 steps.

        eval_strategy="epoch",
        # Run evaluation on the val set at the end of each epoch.

        save_strategy="epoch",
        # Save a checkpoint at the end of each epoch.

        load_best_model_at_end=False,
        # Disabled: loading best model at end is unreliable with quantized models.

        report_to="none",
        # Don't send logs to wandb/tensorboard. Keep it simple.

        dataset_text_field="text",
        # The field in each JSONL record that contains the text to train on.

        max_length=1024,
        # Maximum number of tokens per example. Essays longer than this get truncated.
        # Mistral supports up to 32768, but 1024 keeps memory usage reasonable.
        # Most of these essays will be under 1024 tokens.

        packing=False,
        # If True, multiple short examples are packed into one sequence to fill
        # the context window. We leave it False since essays are long enough.
    )

    # -----------------------------------------------------------------------
    # 3g. SFTTrainer — Supervised Fine-Tuning Trainer
    # -----------------------------------------------------------------------
    # SFTTrainer from the `trl` library is a convenience wrapper around
    # Hugging Face's Trainer. It handles tokenization and the training loop.
    # In trl 1.x, all config lives in SFTConfig above — SFTTrainer is lean.

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,        # val split only — test is held out
        processing_class=tokenizer,
    )

    # -----------------------------------------------------------------------
    # 3h. Train!
    # -----------------------------------------------------------------------

    print("\nStarting training...")
    trainer.train()

    # -----------------------------------------------------------------------
    # 3i. Save the final adapter weights
    # -----------------------------------------------------------------------
    # We only save the LoRA adapter — not the full 14GB model.
    # The adapter is just a few MB. To use it, you load the base model + adapter together.

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nTraining complete. Adapter saved to: {args.output_dir}")

    # -----------------------------------------------------------------------
    # 3j. Test perplexity on the held-out test split
    # -----------------------------------------------------------------------
    # Now that training is done we run the trained model on the test split —
    # essays it has never seen. This gives an unbiased measure of how well
    # the adapter generalised to the essay style (vs. just memorising the
    # training examples).

    print("\nComputing test perplexity on held-out test split...")
    test_texts = [ex["text"] for ex in test_dataset]
    test_perplexity = compute_perplexity(model, tokenizer, test_texts)
    print(f"  Test perplexity: {test_perplexity:.2f}")

    # -----------------------------------------------------------------------
    # 3k. Save metrics JSON (consumed by sweep.py)
    # -----------------------------------------------------------------------
    if args.metrics_file:
        history = trainer.state.log_history

        # Walk backward through the log to find the last recorded train loss
        # and eval loss. The log mixes step-level and epoch-level entries.
        train_loss = next(
            (e["loss"] for e in reversed(history) if "loss" in e and "eval_loss" not in e),
            None,
        )
        eval_loss = next(
            (e["eval_loss"] for e in reversed(history) if "eval_loss" in e),
            None,
        )

        metrics = {
            "r":               args.r,
            "lora_alpha":      args.lora_alpha,
            "lora_dropout":    args.lora_dropout,
            "train_loss":      round(train_loss, 4)      if train_loss      is not None else None,
            "eval_loss":       round(eval_loss, 4)       if eval_loss       is not None else None,
            "test_perplexity": round(test_perplexity, 2),
        }

        # Make sure the parent directory exists (output_dir was already created by the trainer)
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_file)), exist_ok=True)
        with open(args.metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics saved to: {args.metrics_file}")


if __name__ == "__main__":
    main()
