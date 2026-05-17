# inference.py
# Load your fine-tuned model and generate a college essay.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel  # For loading LoRA adapters on top of the base model

# --------------------------------------------------------------------------
# 1. Paths
# --------------------------------------------------------------------------

BASE_MODEL   = "mistralai/Mistral-7B-v0.1"   # The original base model
ADAPTER_PATH = "./mistral-essays-finetuned"   # The LoRA adapter we trained

# --------------------------------------------------------------------------
# 2. Load tokenizer
# --------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# --------------------------------------------------------------------------
# 3. Load base model in 4-bit (same quantization as training)
# --------------------------------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# --------------------------------------------------------------------------
# 4. Load the LoRA adapter on top of the base model
# --------------------------------------------------------------------------
# PeftModel.from_pretrained takes the frozen base model and layers our
# trained adapter weights on top. This is how LoRA inference works —
# the adapter modifies the base model's behavior without changing its weights.

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()  # disables dropout and batch normalization updates (not important here)

print("Model loaded successfully.")

# --------------------------------------------------------------------------
# 5. Generate an essay
# --------------------------------------------------------------------------

def generate_essay(prompt_topic: str, max_new_tokens: int = 500) -> str:
    # Format the prompt exactly as we did during training.
    prompt = (
        "### Instruction:\n"
        "Write a compelling college application essay.\n\n"
        "### Response:\n"
        f"{prompt_topic}"
    )

    # Tokenize: convert the prompt string into a tensor of token IDs.
    # return_tensors="pt" means return PyTorch tensors (not numpy or lists).
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # .to(model.device) moves the input tensor to the same device (GPU) as the model.

    with torch.no_grad():
        # torch.no_grad() tells PyTorch not to track gradients during this
        # forward pass. We're just doing inference, not training, so we don't
        # need gradients. This saves memory and speeds things up.

        output_ids = model.generate(
            **inputs,                    # Pass the tokenized input
            max_new_tokens=max_new_tokens,
            do_sample=True,              # Sample from the probability distribution
                                         # (vs. greedy decoding which always picks the top token)
            temperature=0.8,             # Controls randomness. 1.0 = normal, <1 = more focused,
                                         # >1 = more random. 0.8 gives fluent but varied output.
            top_p=0.9,                   # "Nucleus sampling": at each step, only consider tokens
                                         # whose cumulative probability is in the top 90%.
                                         # Cuts off very unlikely tokens.
            repetition_penalty=1.1,      # Penalize repeating the same tokens. >1 discourages
                                         # repetition. 1.1 is a mild penalty.
            pad_token_id=tokenizer.eos_token_id,  # Needed to avoid a warning
        )

    # Decode: convert token IDs back to a string.
    # output_ids[0] = first (and only) item in the batch.
    # skip_special_tokens=True removes <s>, </s>, etc. from the output.
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # The output includes the prompt. We only want the generated part.
    # Split on "### Response:\n" and take everything after it.
    response = full_output.split("### Response:\n")[-1].strip()

    return response


# --------------------------------------------------------------------------
# 6. Run it
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== College Essay Generator ===")
    print("Type the opening line of your essay, then press Enter.")
    print("Type 'quit' to exit.\n")

    while True:
        topic = input("Opening line: ").strip()

        if topic.lower() == "quit":
            print("Bye!")
            break

        if not topic:
            print("Please enter an opening line.\n")
            continue

        print("\nGenerating...\n")
        print("-" * 60)
        essay = generate_essay(topic, max_new_tokens=200)
        print(essay)
        print("-" * 60 + "\n")
