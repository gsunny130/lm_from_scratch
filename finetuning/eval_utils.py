# eval_utils.py
# Shared evaluation helpers used by train.py and evaluate.py.
#
# Four kinds of evaluation live here:
#   1. compute_perplexity   — statistical measure of how well a model fits text
#   2. compute_mauve_score  — distributional similarity vs. human references (0–1)
#   3. claude_score_essay   — model-based qualitative scoring via Claude API
#   4. score_outputs        — convenience wrapper to score a list of essays

import json
import math
import os

import torch


# --------------------------------------------------------------------------
# Perplexity
# --------------------------------------------------------------------------

def compute_perplexity(model, tokenizer, texts):
    total_loss   = 0.0
    total_tokens = 0

    for text in texts:
        # Tokenize with a max_length cap to avoid OOM on very long essays.
        # truncation=True silently drops tokens beyond the cap rather than
        # raising an error.
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        input_ids = enc["input_ids"]
        n_tokens  = input_ids.shape[1]

        with torch.no_grad():
            # Passing labels=input_ids tells the model to compute
            # cross-entropy loss internally (next-token prediction).
            # outputs.loss is the mean NLL per token for this sample.
            # We weight by essay length so longer essays don't get ignored.
            outputs = model(input_ids=input_ids, labels=input_ids)
            total_loss   += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_nll = total_loss / total_tokens
    return math.exp(avg_nll)


# --------------------------------------------------------------------------
# MAUVE
# --------------------------------------------------------------------------

def compute_mauve_score(p_texts, q_texts, device_id=0):
    """
    Compute MAUVE between a reference distribution (p) and a generated
    distribution (q).  Higher is better (max 1.0).

    p_texts: list of human-written reference strings (e.g. test_texts)
    q_texts: list of model-generated strings
    device_id: GPU id (0) or -1 for CPU

    Returns the MAUVE score as a float, or None if mauve-text is not installed.

    Install: pip install mauve-text
    Note: uses GPT-2 embeddings internally (~500 MB download on first run).
    MAUVE is most reliable with 100+ samples per side; results with <20 may
    be noisy.
    """
    try:
        import mauve as mauve_lib
    except ImportError:
        print("  mauve-text not installed — skipping MAUVE (pip install mauve-text)")
        return None

    result = mauve_lib.compute_mauve(
        p_text=p_texts,
        q_text=q_texts,
        device_id=device_id,
        max_text_length=256,   # truncate to keep memory manageable
        verbose=False,
        featurize_model_name="gpt2",
    )
    return result.mauve


# --------------------------------------------------------------------------
# Claude-based qualitative scoring
# --------------------------------------------------------------------------

def claude_score_essay(client, essay_text):
    """
    Ask Claude Sonnet to score one essay on three dimensions (1–5 each).

    Dimensions:
        narrative_quality    — specific, compelling story vs. generic plot
        specificity          — concrete details vs. vague generalities
        emotional_resonance  — authentic, moving vs. flat and formulaic

    Args:
        client:     anthropic.Anthropic() client instance
        essay_text: the generated essay string

    Returns:
        dict with keys: narrative_quality, specificity, emotional_resonance
             each an int in [1, 5]
    """
    prompt = (
        "You are evaluating a college application essay excerpt. Imagine you are an admissions officer evaluating on if this student should be accepted."
        "Score it on four dimensions, each from 1 to 5, and keep in mind, this student wants to gain admission to your institution and was instructed to write an essay about a personal experience:\n\n"
        "1. narrative_quality — Does it tell a specific, compelling story? "
        "(1 = generic/vague, 5 = vivid and memorable)\n"
        "2. specificity — Does it use concrete details rather than generic statements? "
        "(1 = all generalities, 5 = rich concrete detail)\n"
        "3. emotional_resonance — Does it feel authentic and emotionally engaging? "
        "(1 = flat/formulaic, 5 = genuinely moving)\n"
        "4. coherence — Does it make logical sense? Are the facts internally consistent "
        "and plausible for a real person's life? "
        "(1 = contradictory/nonsensical, 5 = completely coherent)\n"
        "5. task_completion — Does it actually attempt to write a first-person personal essay? "
        "A response that gives writing advice, discusses essays in the third person, lists instructions, "
        "or talks about essay-writing instead of just writing one should score very low. "
        "(1 = completely off-task, 5 = clearly a personal essay written in first person)\n\n"
        "Reply with ONLY valid JSON in this exact format, no other text:\n"
        '{"narrative_quality": <1-5>, "specificity": <1-5>, "emotional_resonance": <1-5>, "coherence": <1-5>, "task_completion": <1-5>}\n\n'
        f"Essay:\n{essay_text[:1500]}"
        # Cap at 1500 chars — scores are based on style/voice, not full content.
        # 1500 chars is roughly the first 3 paragraphs of an essay.
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )

    if not response.content:
        raise ValueError("Claude returned an empty response")
    raw = response.content[0].text.strip()
    if not raw:
        raise ValueError("Claude returned an empty string")
    # Strip markdown code fences if Claude wrapped the JSON in them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"Claude response was not valid JSON: {raw!r}")


def score_outputs(outputs, label):
    """
    Score a list of generated essays with Claude and return averaged scores.

    Requires ANTHROPIC_API_KEY in the environment. If the key is missing,
    prints a warning and returns None so the rest of evaluation still runs.

    Args:
        outputs: list of essay strings (one per eval prompt)
        label:   "base" or "fine-tuned" — used only for progress printing

    Returns:
        dict with averaged scores, or None if scoring was skipped/failed
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(f"  ANTHROPIC_API_KEY not set — skipping Claude scoring for {label}.")
        return None

    client = anthropic.Anthropic(api_key=api_key)

    scores = []
    for i, essay in enumerate(outputs):
        print(f"  Scoring {label} output {i+1}/{len(outputs)}...", end=" ", flush=True)
        try:
            s = claude_score_essay(client, essay)
            scores.append(s)
            print(
                f"NQ={s['narrative_quality']}  "
                f"SP={s['specificity']}  "
                f"ER={s['emotional_resonance']}  "
                f"CO={s['coherence']}  "
                f"TC={s['task_completion']}"
            )
        except Exception as e:
            print(f"error: {e}")

    if not scores:
        return None

    return {
        "narrative_quality":   round(sum(s["narrative_quality"]   for s in scores) / len(scores), 2),
        "specificity":         round(sum(s["specificity"]         for s in scores) / len(scores), 2),
        "emotional_resonance": round(sum(s["emotional_resonance"] for s in scores) / len(scores), 2),
        "coherence":           round(sum(s["coherence"]           for s in scores) / len(scores), 2),
        "task_completion":     round(sum(s["task_completion"]     for s in scores) / len(scores), 2),
    }
