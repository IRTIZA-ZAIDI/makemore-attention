# makemore-attention

Minimal, readable character-level language models in PyTorch, progressing from a simple bigram baseline to a GPT-style Transformer with masked self-attention.

This repo trains on `input.txt` (Tiny Shakespeare by default) and includes both runnable Python scripts and Jupyter notebooks for learning and experimentation.

---

## What’s in this repo

### Models

* **Bigram baseline** (`bigram.py`)

  * Character-level language model using token embeddings (and a small positional embedding table).
  * Trains quickly and provides a simple reference point.

* **Transformer/GPT-style model** (`gpt.py`)

  * Token + positional embeddings
  * Stacked Transformer blocks:

    * masked multi-head self-attention
    * feed-forward MLP
    * residual connections + LayerNorm
    * dropout
  * Autoregressive generation via sampling

* **Bigram v2 (Transformer-like)** (`bigram_v2.py`)

  * A GPT-style architecture similar to `gpt.py` but kept under the “bigram” naming (useful for comparing evolutions).

### Notebooks

* `bigram.ipynb` and `gpt-dev.ipynb` mirror the scripts for interactive runs and step-by-step understanding.

### Dataset

* `input.txt` is the training corpus (character-level). The scripts assume it is in the repo root.

---

## Project structure

```
.
├── bigram.py
├── bigram_v2.py
├── gpt.py
├── bigram.ipynb
├── gpt-dev.ipynb
└── input.txt
```

---

## Requirements

* Python 3.9+ (recommended: 3.10+)
* PyTorch

No other dependencies are required for the scripts in this repo.

---

## Setup

Create a virtual environment and install PyTorch:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows PowerShell

pip install --upgrade pip
pip install torch
```

If you want to run notebooks:

```bash
pip install jupyter
```

---

## Running the models

All scripts automatically select a device in this order:

1. CUDA (NVIDIA GPU)
2. MPS (Apple Silicon)
3. CPU

### 1) Train + generate with the bigram baseline

```bash
python bigram.py
```

This will:

* read `input.txt`
* build a character vocabulary
* train a small model for a few thousand iterations
* print a generated sample at the end

### 2) Train + generate with the GPT-style Transformer

```bash
python gpt.py
```

This will:

* train a multi-layer Transformer (masked self-attention)
* report train/val loss every `eval_interval`
* generate text from an initial context token

### 3) Train + generate with the Transformer-style “bigram v2”

```bash
python bigram_v2.py
```

This uses GPT-like blocks and generation but keeps the “bigram” naming for comparison.

---

## Using a different dataset

Replace `input.txt` with your own plain-text file (UTF-8). Since this is character-level, any unicode characters present become part of the vocabulary.

If you want to use Tiny Shakespeare, the scripts include the reference download link in comments:

* `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`

---

## Key hyperparameters (where to change them)

Each script defines hyperparameters at the top, for example in `gpt.py`:

* `batch_size`: sequences per batch
* `block_size`: context length (max tokens the model can attend to)
* `n_embd`: embedding dimension
* `n_head`: number of attention heads
* `n_layer`: number of Transformer blocks
* `dropout`
* `learning_rate`
* `max_iters`, `eval_interval`, `eval_iters`

If you hit OOM, reduce `batch_size`, `block_size`, `n_embd`, `n_layer`, or `n_head`.

---

## Notes on implementation

* Masked self-attention is implemented with a precomputed lower-triangular causal mask (`tril`) and `masked_fill(..., -inf)` before softmax.
* The Transformer block uses pre-norm LayerNorm with residual connections:

  * `x = x + SA(LN(x))`
  * `x = x + FFN(LN(x))`
* Generation is standard autoregressive sampling from the last timestep’s softmax distribution and appending the sampled token.

---

## Notebook workflow

To explore interactively:

```bash
jupyter notebook
```

Then open:

* `bigram.ipynb`
* `gpt-dev.ipynb`


