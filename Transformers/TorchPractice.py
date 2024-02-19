import torch
from torch.nn import functional as F
import torch.nn as nn
torch.manual_seed(1337)

B, T, C = 4,8,32


# print(x.shape)

x = torch.randn(B,T,C)     # BATCH, TIME (Block Size), CHANNELS (NEURONS)
xbow = torch.zeros((B,T,C))

for b in range(B):     # LOOP THROUGH ALL THE BATCHES
    for t in range(T): # LOOP THROUGH THE "TIME" => BLOCK_SIZE
        xprev = x[b, :t+1]  # INDEX INTO 'x' at each batch and get up to the t of its block_size
        xbow[b,t] = torch.mean(xprev, 0) # index into 0's matrix and place the mean of all previous 
# print(x)
# print(xbow)
        
    

head_size = 16
key = nn.Linear(C, head_size, bias=False)    
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)    # x = (4, 8, 32) x (32, 16)  ==> 4, 8, 16
q = query(x)  # x = (B,T,C)    x (C, 16)   ==> B, T, 16
wei = q @ k.transpose(-2, -1)  # (B, T, 16) x (B, 16, T)  ==> B, T, T
                               # (4, 8, 16) x (4, 16, 8) ==> (4, 8, 8)
                               # This is a matrix which contains the affinity for each batch and its letters
        
# wei = torch.tril(torch.ones(T,T))       # SAME THING DONE BELOW BUT IS LESS EXPENSIVE
# wei = wei / wei.sum(1, keepdim=True)
# xbow2 = wei @ x 

tril = torch.tril(torch.ones(T,T))
#  wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
v = value(x)
out = wei @ v
print(out.shape)
# torch.manual_seed(42)                # SHOWING HOW TO USE TRIANGULAR MATRICES TO 
# a = torch.tril(torch.ones(3,3))      # CREATE RUNNING AVERAGES WITHOUT LOOPS
# a = a / torch.sum(a, 1, keepdim=True)
# b = torch.randint(0, 10, (3,2)).float()
# c = a @ b
# print('a=')
# print(a)
# print('b=')
# print(b)
# print('c=')
# print(c)
