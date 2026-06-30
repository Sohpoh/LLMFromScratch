# LLM From Scratch

A ground-up implementation of a GPT-style large language model in PyTorch, covering everything from basic primitives to a working RAG pipeline.

## What's Here

| Folder | Description |
|--------|-------------|
| `linear_layer` | Custom linear layer |
| `bpe_tokenizer` | BPE tokenizer training |
| `embeddings` | BPE tokenizer inference + custom embedding layer |
| `attention` | Multi-head attention |
| `gpt_architecture` | Full GPT model (transformer decoder blocks) |
| `training_pipeline` | Training loop with cosine LR schedule |
| `text_generation` | Sampling strategies (top-k, top-p, repetition penalties) |
| `rag` | Retrieval-Augmented Generation |

## Breakdown

### Linear Layer
Implements `CustomLinear`, a `torch.nn.Module` equivalent to `torch.nn.Linear` with manually defined weight and bias parameters.

### BPE Tokenizer
Implements Byte Pair Encoding from scratch. Reads raw text, iteratively merges the most frequent character pairs, and saves a `vocab.txt` and `merges.json`. Ties in merge order are broken lexicographically.

### Embeddings
- `Tokenizer`: loads a trained vocab/merges and implements `encode(string) → [int]` and `decode([int]) → string`.
- `CustomEmbedding`: a `torch.nn.Module` wrapping a learnable embedding table.

### Attention
Implements `CustomMHA` with a fused QKV projection (`W_qkv` of shape `(3D, D)`) and output projection (`W_o`). Splits heads, computes scaled dot-product attention, and recombines.

### GPT Architecture
Assembles the full model:
- `TransformerDecoderBlock`: pre-norm MHA + two-layer MLP with residual connections and dropout.
- `GPTModel`: token + position embeddings → N decoder blocks → layer norm → linear projection to vocab logits.

### Training Pipeline
`train_model.py` trains the GPT model on a pre-tokenized dataset (`training_data.npy`). Uses AdamW with a cosine learning rate schedule and linear warmup. Saves a loss curve to `loss_curve.png` and model weights to `model_weights.pt`.

**Dependencies** (managed with `uv`):
```
torch, datasets, transformers, matplotlib, tqdm
```

### Text Generation
`Sampler` implements:
- **top-k** and **top-p** (nucleus) sampling
- **Frequency penalty**: scales logits by how often each token has appeared
- **Presence penalty**: scales logits by whether each token has appeared at all

Includes `example_use_with_gpt2.py` which applies the custom sampler to GPT-2 via Hugging Face.

### RAG
`SimpleRAGNews` builds a minimal retrieval-augmented generation pipeline:
1. Loads 100 BBC News articles from the `permutans/fineweb-bbc-news` dataset.
2. Embeds them with `ibm-granite/granite-embedding-30m-english`.
3. At query time, embeds the user query, finds the most similar article via cosine similarity, and summarizes it using an LLM via the Hugging Face Inference API.

Requires a `HFTOKEN` environment variable (set in `.env`).

## Running

Each folder is self-contained. Example for the embeddings:
```bash
cd embeddings
python tokenizer.py
```

For the training pipeline, first build the dataset:
```bash
cd training_pipeline
python download_data.py
python construct_dataset.py
python train_model.py
```

For RAG, add your Hugging Face token to `rag/.env`:
```
HFTOKEN=your_token_here
```
Then:
```bash
cd rag
python simple_rag.py
```
