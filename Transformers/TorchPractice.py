import torch

torch.manual_seed(1337)

B, T, C = 4,8,2

x = torch.randn(B,T,C)

# print(x.shape)

xbow = torch.zeros((B,T,C))

for b in range(B):     # LOOP THROUGH ALL THE BATCHES
    for t in range(T): # LOOP THROUGH THE "TIME" => BLOCK_SIZE
        xprev = x[b, :t+1]  # INDEX INTO 'x' at each batch and get up to the t of its block_size
        xbow[b,t] = torch.mean(xprev, 0) # index into 0's matrix and place the mean of all previous 
# print(x)
# print(xbow)
        
wei = torch.tril(torch.ones(T,T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x 

        
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