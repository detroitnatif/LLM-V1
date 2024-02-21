import os
os.environ['PATH'] += ':/opt/local/bin'
import torch
import subprocess
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
# g = torch.Generator()
# g.manual_seed(1337)


# subprocess.run(['wget', 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'])

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)


s2i = {i:s for s, i in enumerate(chars)}
i2s = {s:i for s, i in enumerate(chars)}
encode = lambda s: [s2i[c] for c in s]
decode = lambda l: ''.join([i2s[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)

n = int(.9*len(data))
train_data = data[:n]
val_data = data[n:]

# HYPER PARAMETERS
torch.manual_seed(1337)
batch_size = 32
block_size = 32
max_iters = 5000
eval_interval = 200
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
n_head = 4
n_layer = 4
drop_out = .2


def get_batch(split):
    data = train_data if split == "test_data" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 0 to len of data - block size
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y

xb, yb = get_batch('train')

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()                         # SWITCH TO EVAL STAGE
    for split in ['train', 'val']:       # CHOOSE TRAIN OR VAL
        losses = torch.zeros(eval_iters) # ZEROS ARRAY TO KEEP VALS IN
        for k in range(eval_iters):      # LOOP FOR EACH INDEX
            Xb, Yb = get_batch(split)    # GET BATCHS
            logits, loss = model(Xb, Yb) # PREDICT LOSS
            losses[k] = loss.item()      # PLACE INTO ARRAY
        out[split] = losses.mean()       # MAKE DICT OF MEAN OF LOSS FOR TRAIN/VAL
    model.train()                        #  # SWITCH TO TRAIN STAGE
    return out


head_size = 16


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * C**-.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class Multipleheadattention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
    

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.feed = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Dropout(drop_out)
        )
    def forward(self, x):
        out = self.feed(x)
        return out
    

class Block(nn.Module):
    
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = Multipleheadattention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(drop_out)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        x = self.dropout(x)
        return x

    
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.f_layernorm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):

        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # CREATES B,T,C array which is Batch(4) x Time(8) x Channel(65)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.f_layernorm(x)
        # x = self.feed(x)
        logits = self.lm_head(x)

        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # Flattening this to B*T, C
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # Idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
#             print(logits.shape)
            logits = logits[:, -1, :] # this slices the entire 1st and 3rd rows but only the last element of the 2nd
#             print(logits.shape)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            
        return idx
    
model = BigramLanguageModel()
logits, loss = model(xb, yb) # Taking 4 samples of 8 context and making 65 dimensions of embedding
# print(loss)  # WITHOUT TRAINING THE LOSS SHOULD BE "Negative Log Liklihood" => log {-(1/65)}

idx = torch.zeros((1, 1), dtype=torch.long)

# n = model.generate(idx, 25)[0].tolist()


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch = 32 


for steps in range(50):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

n = model.generate(idx, 500)[0].tolist()

print(decode(n))

