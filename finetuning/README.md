# Mistral-7B Essay Fine-tuning with QLoRA

Fine-tunes [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) on 53 Ivy League college application essays using QLoRA. The trained LoRA adapter teaches the model to write in the voice and style of high-quality college essays.

---

## What is QLoRA?

**Q** = Quantization: the base model weights are compressed to 4-bit integers (~75% memory reduction), letting a 7B model fit on a 24GB GPU.

**LoRA** = Low-Rank Adaptation: instead of updating all 7 billion weights, we insert small trainable matrices ("adapters") into the attention layers. Only ~0.2% of parameters are trained — saving compute while still shifting the model's behavior.

---

## Project structure

```
finetuning/
├── essays.csv                    # raw essays dataset (53 essays)
├── essays_formatted.jsonl        # formatted for training (produced by prepare_data.py)
│
├── prepare_data.py               # CSV → JSONL with instruction template
├── data_utils.py                 # shared 80/10/10 train/val/test split
├── train.py                      # QLoRA fine-tuning (supports argparse for sweep)
├── evaluate.py                   # perplexity + qualitative + Claude scoring
├── inference.py                  # interactive essay generator
├── sweep.py                      # hyperparameter sweep (r, lora_alpha, lora_dropout)
├── eval_utils.py                 # compute_perplexity, claude_score_essay helpers
│
├── mistral-essays-finetuned/     # saved LoRA adapter (after training)
└── sweep_results.csv             # produced by sweep.py
```

---

## Setup (Lambda Labs A10)

### 1. Pick instance
- Type: **1x A10**
- Image: **Lambda Stack 22.04** (NOT plain Ubuntu — it won't have CUDA)

### 2. Transfer files

```bash
ssh-keyscan -H <instance-ip> >> ~/.ssh/known_hosts
scp -i ~/.ssh/id_ed25519 \
    essays.csv prepare_data.py data_utils.py eval_utils.py \
    train.py evaluate.py inference.py sweep.py \
    ubuntu@<instance-ip>:~/
```

### 3. Create virtual environment (run once)

```bash
python3 -m venv ~/env --system-site-packages
~/env/bin/pip install -q 'numpy==1.26.4' 'transformers==4.46.2' 'peft==0.13.2' \
    'trl==0.11.4' 'accelerate>=0.34.0' 'datasets' 'pandas' 'anthropic'
~/env/bin/pip install --upgrade bitsandbytes
```

### 4. Verify GPU

```bash
~/env/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# Should print: CUDA: True  GPU: NVIDIA A10
```

---

## How to run

### Step 1: Format the data

```bash
~/env/bin/python prepare_data.py
# Produces essays_formatted.jsonl
```

### Step 2: Train

```bash
~/env/bin/python train.py
# Uses default config: r=8, lora_alpha=16, lora_dropout=0.05
# Saves adapter to ./mistral-essays-finetuned/
# ~95 seconds on an A10
```

### Step 3: Evaluate

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # optional — enables Claude scoring
~/env/bin/python evaluate.py
```

### Step 4: Interactive inference

```bash
~/env/bin/python inference.py
# Type an opening line → get an essay continuation
```

### Step 5: Hyperparameter sweep (optional, ~1 hour)

```bash
~/env/bin/python sweep.py
# Runs 6 configs, saves sweep_results.csv
```

### Download adapter to your Mac

```bash
scp -r -i ~/.ssh/id_ed25519 ubuntu@<instance-ip>:~/mistral-essays-finetuned \
    /Users/graceyoon/finetuning/
```

Then terminate the instance.

---

## Dataset split

| Split | Size | Purpose |
|-------|------|---------|
| Train | 80% (~42 essays) | Gradient updates |
| Val   | 10% (~5 essays)  | eval_loss during training |
| Test  | 10% (~5 essays)  | Final perplexity — held out entirely |

The same fixed seed (42) is used in `train.py` and `evaluate.py` via `data_utils.get_splits()`, so both scripts always agree on which essays are in the test set.

---

## Hyperparameter sweep results

Run with `sweep.py`. Results sorted by test perplexity (lower = better style fit).

| r  | lora_alpha | lora_dropout | train_loss | eval_loss | test_ppl |
|----|-----------|-------------|-----------|----------|---------|
| 8  | 16        | 0.05        | 2.10      | 2.17     | —       |
| 16 | 32        | 0.05        | —         | —        | —       |
| 16 | 32        | 0.10        | —         | —        | —       |
| 8  | 16        | 0.10        | —         | —        | —       |
| 4  | 8         | 0.05        | —         | —        | —       |
| 4  | 8         | 0.10        | —         | —        | —       |

*Fill in after running `sweep.py`.*

### What each hyperparameter does

**`r` (rank):** Controls how many parameters the LoRA adapters add. `r=8` adds ~8M trainable params. Higher rank = more expressive but more prone to overfitting on small datasets. On 53 essays, lower rank (r=4 or r=8) often generalises better.

**`lora_alpha`:** Scales the adapter's contribution to the model output — the effective scale is `alpha/r`. By always setting `alpha = 2*r`, we keep this ratio constant across runs, so rank and dropout are the variables being tested.

**`lora_dropout`:** Regularization — randomly zeros out adapter weights during training. Higher dropout (0.1 vs 0.05) reduces overfitting at the cost of a noisier training signal.

---

## Evaluation: what the numbers mean

**Perplexity** measures how "surprised" the model is by the test essays — lower means it assigns higher probability to the essay-style text. The fine-tuned model should score lower than the base model on the held-out test split.

**Claude scores** (1–5 each, averaged over 4 prompts):

| Metric | Measures |
|--------|---------|
| Narrative quality | Specific, memorable story vs. generic plot |
| Specificity | Concrete details vs. vague generalities |
| Emotional resonance | Authentic voice vs. flat and formulaic |
| Coherence | Logically consistent, plausible for a real person's life vs. contradictory/nonsensical |
| Task completion | Actually writes a first-person personal essay vs. gives writing advice or goes off-task |

---

## Example outputs (base vs. fine-tuned)

*Fill in after running `evaluate.py`.*

**Prompt:** `"I"`

**Base model:**
> I want to study medicine because I have always been passionate about helping people...

**Fine-tuned:**
> I pressed my ear to the linoleum floor of the hospital hallway, listening for the IV drip...

---

## Healthy training loss numbers

From the baseline run (r=8, alpha=16, dropout=0.05):

| Epoch | Train loss | Eval loss |
|-------|-----------|---------|
| 1     | 2.31      | 2.21    |
| 2     | 2.21      | 2.17    |
| 3     | 2.10      | 2.18    |

Eval loss ticking up at epoch 3 = mild overfitting, expected with only 53 examples.

---

## Library versions (confirmed working)

| Library        | Version  |
|----------------|---------|
| torch          | 2.7.0   |
| transformers   | 4.46.2  |
| peft           | 0.13.2  |
| trl            | 0.11.4  |
| bitsandbytes   | 0.49.2  |
| accelerate     | 1.13.0  |
| anthropic      | latest  |
| Python         | 3.10    |
| CUDA           | 12.8    |

---

## Key gotchas

- **Use Lambda Stack 22.04**, not plain Ubuntu — plain Ubuntu has no CUDA drivers
- **Use `--system-site-packages`** when creating the venv — inherits CUDA-enabled PyTorch from Lambda Stack
- **Do NOT run `pip install --upgrade numpy`** — breaks system scipy/torch ABI
- **Do NOT install bitsandbytes before upgrading it** — 0.43.x lacks CUDA 12.8 binary
- **`device_map={"": 0}`** for inference, `device_map="auto"` for training
- **trl 0.11.4 uses `tokenizer=`**, not `processing_class=` — check your version if you see warnings
