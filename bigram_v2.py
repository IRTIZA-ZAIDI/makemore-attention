import torch
import torch.nn as nn
from torch.nn import functional as F

# --------------------
# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # maximum context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
eval_iters = 200
n_embd = 32
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

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        # wei[b, i, j] = (q[b,i] · k[b,j]) / sqrt(head_size)
        # head_size = q.size(-1)
        wei = (
            q @ k.transpose(-2, -1) * (q.size(-1) ** -0.5)
        )  # (B,T,C) @ (B,C,T) --> (B,T,T)
        # self.tril[:T,:T] -> Slices the mask to match current input size
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        # perform weighted aggregation of values
        v = self.value(x)
        out = wei @ v  # (B,T,T) @ (B,T,C) --> (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


# --------------------
# FeedForward layer
class FeedForward(nn.Module):
    """ " a simple feed forward layer followed by non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, n_embd), nn.ReLU())

    def forward(self, x):
        return self.net(x)

# --------------------
# Block
class Block(nn.Module):
    """ transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        # n_embd: embedding dimension 
        # n_head: the number of heads 
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x

# --------------------
# Bigram + Self-Attention Model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embdding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embdding_table = nn.Embedding(block_size, n_embd)
        # self.sa_head = Head(
        #     n_embd
        # )  # passing n_embd head size so taking B, T, C = x.shape in Head forward makes sense
        self.sa_heads = MultiHeadAttention(
            4, n_embd // 4
        )  # i.e. 4 heads of 8-dimentional self-attention
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embdding_table(idx)  # (B, T, C)
        pos_emb = self.position_embdding_table(torch.arange(T, device=device))  # (T, C)

        x = tok_emb + pos_emb  # (B, T, C)
        x = self.sa_heads(x)  # (B, T, C)
        # ffwd is applied per token
        # Same weights, no token mixing
        x = self.ffwd(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens for pos embedding
            # : → take all batches
            # -block_size: → take last block_size tokens
            idx_cond = idx[:, -block_size:]

            # get the predictions
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :]  # (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


# --------------------
# model + optimizer
model = BigramLanguageModel()
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
