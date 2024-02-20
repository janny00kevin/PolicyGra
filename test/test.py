import numpy as np
# print("hello warld") 

import torch
from torch.distributions.categorical import Categorical

logits = torch.tensor([ 0.25, 0.75 ])

m = Categorical(probs = logits)
print(m.log_prob())
# for i in range(10000):
    # assert m.sample().item() != 0, i

    # if m.sample().item() == 0:
    #     print(i)

    # print(m.sample().item())
# print(logits[1].type())

# print(torch.tensor([ 0.1, 0.9, 0.3 ]).dim())