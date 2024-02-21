import numpy as np
# print("hello warld") 

import torch
from torch.distributions.categorical import Categorical

logits = torch.tensor([ 0.25, 0.2, 0.55 ])

m = Categorical(probs = logits)
sample = m.sample()
print(torch.exp(m.log_prob(sample)).item())

a = torch.tensor([1,2,3.1])
b = torch.tensor([4,5,6])
# print((a*b).mean())
print(torch.tensor([[1,2,3.01],[4,5,6]]).mean())
# for i in range(10000):
    # assert m.sample().item() != 0, i

    # if m.sample().item() == 0:
    #     print(i)

    # print(m.sample().item())
# print(logits[1].type())

# print(torch.tensor([ 0.1, 0.9, 0.3 ]).dim())