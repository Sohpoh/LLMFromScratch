# LLM From Scratch

A module-by-module implementation of a GPT-style large language model, built from the ground up using PyTorch.

## Overview

Each module builds on the last, starting from basic neural network primitives and ending with a working RAG pipeline.

| Module | Topic |
|--------|-------|
| Module 1 | Custom linear layer |
| Module 2 | BPE tokenizer training |
| Module 3 | BPE tokenizer inference + custom embedding layer |
| Module 4 | Multi-head attention (MHA) |
| Module 5 | Full GPT model (transformer decoder blocks) |
| Module 6 | Training loop with cosine LR schedule |
| Module 7 | Sampling strategies (top-k, top-p, repetition penalties) |
| Module 8 | Retrieval-Augmented Generation (RAG) |

## Module Breakdown

### Module 1 — Linear Layer
Implements `CustomLinear`, a `torch.nn.Module` equivalent to `torch.nn.Linear` with manually defined weight and bias parameters.

### Module 2 — BPE Tokenizer Training
Implements Byte Pair Encoding (BPE) from scratch. Reads raw text, iteratively merges the most frequent character pairs, and saves a `vocab.txt` and `merges.json`. Ties in merge order are broken lexicographically.

### Module 3 — Tokenizer Inference + Embeddings
- `Tokenizer`: loads a trained vocab/merges and implements `encode(string) → [int]` and `decode([int]) → string`.
- `CustomEmbedding`: a `torch.nn.Module` that wraps a learnable embedding table.

### Module 4 — Multi-Head Attention
Implements `CustomMHA` with a fused QKV projection (`W_qkv` of shape `(3D, D)`) and output projection (`W_o`). Splits heads, computes scaled dot-product attention, and recombines.

### Module 5 — GPT Model
Assembles the full architecture:
- `TransformerDecoderBlock`: pre-norm MHA + two-layer MLP with residual connections and dropout.
- `GPTModel`: token + position embeddings → N decoder blocks → layer norm → linear projection to vocab logits.

### Module 6 — Training
`train_model.py` trains the GPT model on a pre-tokenized dataset (`training_data.npy`). Uses AdamW with a cosine learning rate schedule and linear warmup. Saves a loss curve to `loss_curve.png` and model weights to `model_weights.pt`.

**Dependencies** (managed with `uv`):
```
torch, datasets, transformers, matplotlib, tqdm
```

### Module 7 — Sampling
`Sampler` implements:
- **top-k** and **top-p** (nucleus) sampling
- **Frequency penalty**: scales logits by how often each token has appeared
- **Presence penalty**: scales logits by whether each token has appeared at all

Includes `example_use_with_gpt2.py` which applies the custom sampler to GPT-2 via Hugging Face.

### Module 8 — RAG
`SimpleRAGNews` builds a minimal retrieval-augmented generation pipeline:
1. Loads 100 BBC News articles from the `permutans/fineweb-bbc-news` dataset.
2. Embeds them with `ibm-granite/granite-embedding-30m-english`.
3. At query time, embeds the user query, finds the most similar article via cosine similarity, and summarizes it using an LLM via the Hugging Face Inference API.

Requires a `HFTOKEN` environment variable (set in `.env`).

## Running

Each module is self-contained. Example for Module 3:
```bash
cd Module3
python tokenizer.py
```

For Module 6 (training), first build the dataset:
```bash
cd Module6
python download_data.py
python construct_dataset.py
python train_model.py
```

For Module 8 (RAG), add your Hugging Face token to `Module8/.env`:
```
HFTOKEN=your_token_here
```
Then:
```bash
cd Module8
python simple_rag.py
```
