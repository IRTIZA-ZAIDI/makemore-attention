import torch
import torch.nn as nn
from torch.nn import functional as F

# --------------------
# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
eval_iters = 200
n_embd = 384
# every head is 384/6 = 64 dimensional
n_head = 6
n_layer = 6
# 20% of weights are disabled
dropout = 0.2
# --------------------

torch.manual_seed(1337)

# download dataset if needed:
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# train / val split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# --------------------
# data loader
def get_batch(split):
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i : i + block_size] for i in ix])
    y = torch.stack([data_split[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# --------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# --------------------
# Attention Head
class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        # wei[b, i, j] = (q[b,i] · k[b,j]) / sqrt(n_head)
        # head_size = q.size(-1)
        wei = (
            q @ k.transpose(-2, -1) * (q.size(-1) ** -0.5)
        )  # (B,T,C) @ (B,C,T) --> (B,T,T)
        # self.tril[:T,:T] -> Slices the mask to match current input size
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        # randomly prevent some of the nodes from communicating
        wei = self.dropout(wei)
        # perform weighted aggregation of values
        v = self.value(x)
        out = wei @ v  # (B,T,T) @ (B,T,C) --> (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(num_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# --------------------
# FeedForward layer
class FeedForward(nn.Module):
    """ " a simple feed forward layer followed by non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # ffwd is applied per token
        # Same weights, no token mixing
        return self.net(x)


# --------------------
# Block
class Block(nn.Module):
    """transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        # n_embd: embedding dimension
        # n_head: the number of heads
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# --------------------
# Bigram + Self-Attention Model
class BigramLanguageModel(nn.Module):
    """
    A GPT-style language model:
    - token + position embeddings
    - stacked Transformer blocks
    - final LayerNorm + linear head
    """

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, device):
        super().__init__()
        self.device = device
        self.block_size = block_size
        # Each token index maps to a learnable embedding vector
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Positional embeddings give the model a sense of order
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Stack of Transformer blocks (self-attention + MLP)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        # Final normalization before output projection
        self.ln_f = nn.LayerNorm(n_embd)
        # Maps final hidden states to vocabulary logits
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) integer token indices
        targets: (B, T) target token indices (optional)
        """
        B, T = idx.shape
        # Token embeddings: (B, T, C)
        tok_emb = self.token_embedding_table(idx)
        # Position embeddings: (T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        # Combine token identity + position information
        x = tok_emb + pos_emb  # (B, T, C)
        # Apply stacked Transformer blocks
        x = self.blocks(x)  # (B, T, C)
        # Final layer normalization
        x = self.ln_f(x)  # (B, T, C)
        # Project to vocabulary size to get logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute cross-entropy loss if targets are provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Autoregressively generate tokens.
        idx: (B, T) current context
        """
        for _ in range(max_new_tokens):
            # Keep only the last block_size tokens
            idx_cond = idx[:, -self.block_size :]
            # Forward pass to get logits
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (B, vocab_size)
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample the next token
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# --------------------
# model + optimizer
model = BigramLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer,
    device=device,
)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --------------------
# training loop
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --------------------
# generation
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
