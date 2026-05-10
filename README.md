# Building a Language Model from Scratch

A from-scratch implementation of a modern transformer language model, built 
for understanding. Every component is implemented and documented — tokenization, 
embeddings, RoPE, multi-head attention, SwiGLU, RMSNorm, transformer blocks, 
training loop, and inference.

## Credits
Architecture and structure inspired by 
[how-to-train-your-gpt](https://github.com/raiyanyahya/how-to-train-your-gpt) 
by Raiyan Yahya (MIT License) (credit for much of code, explanations, and examples) and Andrej Karpathy's nanoGPT.

## Architecture
- **95M parameters**
- RoPE positional embeddings (instead of sinusoidal)
- SwiGLU feedforward (instead of ReLU/GELU)
- RMSNorm (instead of LayerNorm)
- Follows LLaMA/Mistral conventions rather than original GPT-2

## Training
- Dataset: FineWeb-Edu (10k samples)
- 20,000 steps, AdamW optimizer, cosine LR schedule with warmup
- Final loss: ~3.6
- Hardware: 2x T4 GPUs with mixed precision (bfloat16)

## Sample Output

Prompt: *"The unanimous Declaration of the thirteen united States of America"*
> was founded and elected by British nurses. But the President's bishop was 
> the first American Secretary of State to recognize these little.] beer, 
> smoothly armed official and inv peasantry, but no real Nazis™ were on their 
> way to the mass states.

Prompt: *"Harvard University"*
> was founded in 1636 attended by two authors for Asworld radius and Science. 
> Pen abandonment requires a change in the shape of theness of the puzzle and 
> the necessity for celebration as well as the principles of the problem...

It knows Harvard was founded in 1636. It also invented "theness of the puzzle." 
We call that a win.

## Notes
Detailed notes covering every component of the architecture are in 
`LLM_notes.md` — written from scratch for understanding, not copied.

## Known Issues
1. Not saving best model
2. Not using training/val data
3. GPU use inefficient
