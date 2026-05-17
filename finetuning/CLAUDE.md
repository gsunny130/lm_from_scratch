# Fine-tuning Setup Guide

## What this project does
Fine-tunes Mistral-7B-v0.1 on college essays using QLoRA (started with 53, later expanded to ~106).
The trained LoRA adapter is saved in `mistral-essays-finetuned/`.

## Lambda Labs instance setup

### Instance to pick
- Type: **1x A10**
- Image: **Lambda Stack 22.04** (NOT plain Ubuntu — it won't have CUDA)

### SSH
```bash
# Your public key is already registered with Lambda
ssh ubuntu@<instance-ip>
```

### Transfer files
```bash
ssh-keyscan -H <instance-ip> >> ~/.ssh/known_hosts
scp -i ~/.ssh/id_ed25519 essays.csv prepare_data.py train.py inference.py ubuntu@<instance-ip>:~/
```

### Set up the venv (run once on the instance)
```bash
python3 -m venv ~/env --system-site-packages
~/env/bin/pip install -q 'numpy==1.26.4' 'transformers==4.46.2' 'peft==0.13.2' 'trl==0.11.4' 'accelerate>=0.34.0' 'datasets' 'pandas'
~/env/bin/pip install --upgrade bitsandbytes
```

### Verify GPU is working
```bash
~/env/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```
Should print: `CUDA: True` and `GPU: NVIDIA A10`

## Baseline (run BEFORE training)

Before fine-tuning, run inference on the base model so you can compare outputs:

```bash
~/env/bin/python inference.py  # but point ADAPTER_PATH at the base model only
```

Or temporarily comment out the PeftModel lines in inference.py and run the base model directly.
Save a few example outputs. After fine-tuning, run the same prompts and compare.
This proves the fine-tuning actually changed something.

## Training

```bash
~/env/bin/python prepare_data.py   # formats essays.csv → essays_formatted.jsonl
~/env/bin/python train.py          # downloads model (~20s), trains (~95s)
```

### What to expect during training
- Model downloads to `~/.cache/huggingface` (~14GB, fast on Lambda)
- Training prints loss every 5 steps, eval loss every epoch
- 3 epochs, 18 total steps, ~95 seconds
- Adapter saved to `./mistral-essays-finetuned/`


## Inference

```bash
~/env/bin/python inference.py
```

Interactive CLI — type an opening line, get an essay continuation. Type `quit` to exit.

## After training: download adapter to Mac
```bash
scp -r -i ~/.ssh/id_ed25519 ubuntu@<instance-ip>:~/mistral-essays-finetuned /Users/graceyoon/finetuning/
```

Then terminate the instance.

## Library versions (confirmed working)
| Library        | Version  |
|----------------|----------|
| torch          | 2.7.0    |
| transformers   | 4.46.2   |
| peft           | 0.13.2   |
| trl            | 0.11.4   |
| bitsandbytes   | 0.49.2   |
| accelerate     | 1.13.0   |
| Python         | 3.10     |
| CUDA           | 12.8     |

## Evaluation results (114-essay dataset)

### Training loss (3 epochs — config TBD, verify in train.py)
| Epoch | Train loss | Eval loss |
|-------|-----------|-----------|
| 1     | 2.235     | 2.356     |
| 2     | 2.127     | 2.351     |
| 3     | 2.085     | 2.354     |

Eval loss flat across epochs = less overfitting than 53-essay run. Training took ~252s (36 steps).

### Perplexity (held-out test split)
| Model        | Perplexity | Improvement |
|--------------|------------|-------------|
| Base model   | 10.35      |             |
| Fine-tuned   | 9.16       | 11.5%       |

### Claude scores (avg over 4 prompts, 1–5 each)
| Dimension          | Base | Fine-tuned | Delta  |
|--------------------|------|------------|--------|
| Narrative Quality  | 1.25 | 2.50       | +1.25  |
| Specificity        | 1.50 | 3.00       | +1.50  |
| Emotional Resonance| 1.25 | 2.25       | +1.00  |
| Coherence          | 3.25 | 3.50       | +0.25  |
| Task Completion    | 2.00 | 4.25       | +2.25  |

Key insight: base model failed task completely on 2/4 prompts (gave essay-writing tips instead of writing essays).
Fine-tuned model wrote actual personal essays every time.

## Evaluation results (53-essay dataset)

### Training loss (3 epochs, r=16, alpha=32, dropout=0.05)
| Epoch | Train loss | Eval loss |
|-------|-----------|-----------|
| 1     | 2.31      | 2.21      |
| 2     | 2.21      | 2.17      |
| 3     | 2.10      | 2.18      |

### Perplexity (honest numbers — held-out test split)
| Model        | Perplexity | Notes                        |
|--------------|------------|------------------------------|
| Base model   | 12.47      |                              |
| Fine-tuned   | 11.05      | 11% improvement              |

(Earlier run showed base=10.42, fine-tuned=8.36 / 19.8% — invalid, was using training data)

### LoRA hyperparameter sweep (test perplexity)
| r  | alpha | dropout | Test perplexity |
|----|-------|---------|-----------------|
| 16 | 32    | 0.05    | 11.01 ← best    |
| 16 | 32    | 0.10    | 11.01           |
| 8  | 16    | 0.05    | 11.04           |
| 8  | 16    | 0.10    | 11.04           |
| 4  | 8     | 0.05    | 11.22           |
| 4  | 8     | 0.10    | 11.23           |

Best config: r=16, alpha=32, dropout=0.05

## Key gotchas learned the hard way
- **Use Lambda Stack 22.04**, not plain Ubuntu — plain Ubuntu has no CUDA drivers
- **Use `--system-site-packages`** when creating the venv — inherits CUDA-enabled PyTorch from Lambda Stack
- **Do NOT run `pip install --upgrade numpy`** — breaks system scipy/torch ABI
- **Do NOT install bitsandbytes before upgrading it** — 0.43.x lacks CUDA 12.8 binary; install then `--upgrade`
- **`device_map={"": 0}`** not `device_map="auto"` — auto splits layers to CPU with 4-bit models
- **trl 0.11.4 uses `tokenizer=`**, not `processing_class=` (that's trl 1.x)
- **trl 0.11.4 uses `max_seq_length=`** in SFTConfig (trl 1.x uses `max_length=`)
