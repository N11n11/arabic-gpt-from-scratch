# Arabic Stories Language Model

A GPT-style autoregressive language model trained from scratch on Arabic narrative text. The model is trained in two stages: unsupervised pretraining on a short story corpus, followed by supervised instruction tuning for story generation tasks.

## Overview

This implementation does not rely on any pretrained weights or external model libraries. The transformer architecture, tokenizer, training loop and evaluation are all written from scratch using PyTorch.

The training corpus consists of Arabic short stories covering themes of wisdom, justice, travel, friendship and everyday life. The instruction tuning dataset contains 200 curated pairs for tasks including story writing, narrative completion, moral analysis, character description and scene writing.

## Running the Notebook

Open `arabic_gpt_stories.ipynb` in Google Colab. Switch to GPU runtime (T4) before running. All cells execute sequentially with no manual steps required.

## Architecture

The model class is called `StoryGPT`. Unlike many implementations that use a single monolithic attention class, this one separates each attention head into its own `AttentionHead` module which makes the forward pass easier to follow.

Component breakdown:

**AttentionHead** — a single scaled dot-product attention head with causal masking applied directly in the head forward pass

**MultiHeadAttention** — runs N heads in parallel and concatenates their outputs before projection

**FFN** — two linear layers with GELU activation and dropout in between

**Block** — one full transformer block combining attention and FFN with pre-norm and residual connections

**StoryGPT** — the full model with token and positional embeddings, stacked blocks, final layer norm and language model head

Model configuration:

| parameter | value |
|-----------|-------|
| d_model | 192 |
| n_heads | 6 |
| n_layers | 4 |
| d_ff | 768 |
| seq_len | 128 |
| dropout | 0.1 |

## Tokenizer

Character-level tokenization. The vocabulary is built by scanning all characters in both the pretraining and fine-tuning data. Special tokens are PAD, UNK, BOS, EOS and SEP. Encoding and decoding are implemented as standalone functions rather than a class.

## Training Details

Pretraining objective is standard next-token prediction (cross entropy loss). The optimizer is AdamW with weight decay 0.01 and gradient clipping at norm 1.0. Learning rate follows a cosine annealing schedule.

Fine-tuning uses the same objective but the input format wraps each sample as BOS + instruction + SEP + response + EOS with PAD tokens to fill the context window.

| phase | lr | epochs | batch size |
|-------|----|--------|------------|
| pretraining | 2e-4 | 25 | 16 |
| fine-tuning | 8e-5 | 15 | 8 |

## Evaluation

Perplexity is calculated on both the pretraining and fine-tuning sets after training completes. An error analysis runs the model on three edge cases: a long output request, a very short instruction and an out-of-distribution topic. Each response is checked for minimum length, Arabic character ratio and excessive newlines.

## Outputs

The notebook saves the following on completion:

```
checkpoints/
    pretrained/model.pt
    finetuned/model.pt
    vocab.json
results/
    plots/pretrain_curves.png
    plots/sft_curves.png
    plots/combined_training_curves.png
    sample_generations/pretrain_stories_generations.json
    sample_generations/sft_stories_generations.json
    error_analysis.json
    final_report.json
data/
    pretrain/data.txt
    finetune/stories/sft_data.json
```

## Dependencies

```
torch >= 2.0.0
numpy >= 1.24.0
matplotlib >= 3.7.0
tqdm >= 4.65.0
```

Transformers and tokenizers libraries are installed but not used for the core model. They are listed in requirements.txt for completeness.
