# Finetuning a Small LLM for Domain Q/A (Pythia-70M)

This repo fine-tunes **EleutherAI/pythia-70m** as a domain Q/A assistant using a compact instruction dataset (`lamini/lamini_docs`).

## Pipeline
- **Data prep**: tokenizer alignment (pad = EOS), left-side truncation, `max_length=2048`; pack `question + answer` for causal LM.
- **Training**: Hugging Face `Trainer` with `learning_rate=1e-5`, `gradient_accumulation_steps=4`, `optim="adafactor"`, short warmup; save best by `eval_loss`.
- **Evaluation**: exact-match on held-out test; optional `lm-evaluation-harness` (e.g., `arc_easy`) for sanity checks.

## Stack
Python · Hugging Face Transformers/Accelerate/Datasets · PyTorch

> Base model: `EleutherAI/pythia-70m`. Dataset: `lamini/lamini_docs`. Example finetuned ref: `lamini/lamini_docs_finetuned`.
