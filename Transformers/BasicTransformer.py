#!/usr/bin/env python
# coding: utf-8

# In[29]:


import os
os.environ['PATH'] += ':/opt/local/bin'
import torch


# In[9]:


get_ipython().system('wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')


# In[10]:


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# In[12]:


print("Characters in Dataset")
print(len(text))


# In[13]:


print(text[:500])


# In[18]:


chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars), len(chars))


# In[28]:


s2i = {i:s for s, i in enumerate(chars)}
i2s = {s:i for s, i in enumerate(chars)}
encode = lambda s: [s2i[c] for c in s]
decode = lambda l: ''.join([i2s[i] for i in l])

a = encode("hiii there!")
print(a)
print(decode(a))


# In[33]:


data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:5])


# In[34]:


n = int(.9*len(data))
train_data = data[:n]
val_data = data[n:]


# In[37]:


block_size = 8
train_data[:block_size + 1]


# In[36]:


x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):  # WHEN YOU INDEX INTO THE FIRST ELEMENT
    context = x[:t+1]        # ITS train_data[:1] not train_data[0]
    target = y[t]
    print(f'when input is {context}, target is {target}')


# In[ ]:





# In[106]:


torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == "test_data" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 0 to len of data - block size
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
#     print('ix')
#     print(ix)
#     print("4 random numbers from len(data)\n")
    
#     print('x')
#     print("Index into 8 consectutive numbers")
#     print(x)
    
#     print('y')
#     print("Index into 8 consectutive numbers offset by 1\n")
#     print(y)
    
    
    return x, y

xb, yb = get_batch('train')

print("inputs")
print(xb.shape)
print(xb)

print("targets")
print(yb.shape)
print(yb)


# In[107]:


import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
g = torch.Generator()
g.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # CREATES B,T,C array which is Batch(4) x Time(8) x Channel(65)
        B, T, C = logits.shape
        
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
            logits, loss = self(idx)
#             print(logits.shape)
            logits = logits[:, -1, :] # this slices the entire 1st and 3rd rows but only the last element of the 2nd
#             print(logits.shape)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1, generator=g )
            idx = torch.cat((idx, idx_next), dim=1)

            
        return idx
    


# In[102]:


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb) # Taking 4 samples of 8 context and making 65 dimensions of embedding
print(loss)  # WITHOUT TRAINING THE LOSS SHOULD BE "Negative Log Liklihood" => log {-(1/65)}

idx = torch.zeros((1, 1), dtype=torch.long)

n = m.generate(idx, 25)[0].tolist()
print(decode(n))


# In[108]:


# Create a PyTorch Optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


# In[114]:


batch = 32 
for steps in range(100):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
print(loss.item())

n = m.generate(idx, 100)[0].tolist()
print(decode(n))
    


# In[ ]:





# In[ ]:




