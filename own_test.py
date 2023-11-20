import torch
import torch.nn as nn
from torch.nn import functional as F
import preprocess_data as P
from preprocess_data import read_vocab, create_mappings, encode_string, decode_list
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"The device being used is: {device}")
# Independent sequences to process in parellel
num_batch = 16
# maximum context length
# TODO: May make 64 on cuda system
context_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

# Create train and test splits
text = read_vocab("combined_sql.txt")
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
stoi, itos = create_mappings(vocab)
encoded = encode_string(stoi, "hello")
decoded = decode_list(itos, encoded)

# Make the train and test split

data = torch.tensor(encode_string(stoi, text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# Data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    batch_indices = torch.randint(len(data) - context_size, (num_batch,))
    x, y = [], []

    for i in batch_indices:
        x.append(data[i:i + context_size])
        y.append(data[i + 1:i + context_size + 1])

    x = torch.stack(x).to(device)
    y = torch.stack(y).to(device)
    return x, y


# Loss estimation (Preventing gradient accumulation)
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # self.tril = torch.tril(torch.ones(context_size, context_size))
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = (q @ k.transpose(-2, -1)) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = self.dropout(F.softmax(wei, dim=-1))

        return wei @ v


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        cat_tensors = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(cat_tensors))


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        return self.ffwd(self.ln2(x + self.sa(self.ln1(x))))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None if targets is None else F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = GPT()
print('loaded model parameters...')
model_path = "SQL_writer_GPTV1.pkl"
model = torch.load(model_path)
m = model.to(device)
print("Model loaded successfully")
print(f'{sum(p.numel() for p in m.parameters()) / 1e6} Million parameters')

print(model)
# generate with prompt
count = 2
prompt = "SELECT COUNT"
for cnt in range(1, count):
    # Encode the prompt
    encoded_prompt = encode_string(stoi, prompt)

    # Reshape the tensor if necessary (e.g., reshape it to (1, len(encoded_prompt)))
    context = torch.tensor(encoded_prompt, dtype=torch.long, device=device).reshape(1, -1)

    # Generate text based on the prompt
    generated_text = decode_list(mapping=itos, l=m.generate(context, max_new_tokens=1000)[0].tolist())
    prompt = str(generated_text)
    # Print the generated text
    print(generated_text)


# device = "mps"
# model_path = "song_writer_GPTV1.pkl"
# model = torch.load(model_path)
# m = model.to(device)
# print(f'{sum(p.numel() for p in m.parameters()) / 1e6} Million parameters')
