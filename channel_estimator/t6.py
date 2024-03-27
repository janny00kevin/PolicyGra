import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.distributions.normal import Normal
# import matplotlib.pyplot as plt 
# import time

num_trajectories = 2
H_mean = 0; H_sigma = 1
n_R = 1; n_T = 2; T = 2
hidden_sizes=[64, 32]; lr = 1e-4
device = torch.device('cpu')

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Softplus):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

lamb, t = torch.tensor([1, 0]).to(device)
logits_net = mlp(sizes=[2*n_R*T]+hidden_sizes+[2*2*n_R*n_T]).to(device)
optimizer = Adam(logits_net.parameters(), lr=lr)
X = torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T)).to(device)
tau_h_hat = tau_h = tau_y= torch.zeros(0)
## generate trajectories
for k in range(num_trajectories):
    ## generate communication data
    H = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(n_R, n_T, 2))).to(device)
        # H = np.random.normal(H_mean, H_sigma, [n_R,2*n_T]).view(np.complex128)
    W = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(n_R, T, 2))).to(device)
        # W = np.random.normal(W_mean, W_sigma, [n_R,T*2]).view(np.complex128)
    Y = H.matmul(X) + W
        # Y = np.matmul(H,X) + W

    ## forward propagation
        # y = Y.transpose().reshape(-1).view(np.double)
    y = torch.flatten(torch.view_as_real(Y))
    tau_y = torch.cat((tau_y, y.unsqueeze(0)), 0)
    logits = logits_net(y)
    h_hat = Normal(logits[:len(logits)//2], logits[len(logits)//2:]).sample()
        # h = H.transpose().reshape(-1).view(np.double)
    h = torch.flatten(torch.view_as_real(H))
        # tau.append([h, h_hat])
    # tau = torch.cat((tau, torch.cat((h.unsqueeze(0), h_hat.unsqueeze(0)), 0).unsqueeze(0)), 0)
    tau_h = torch.cat((tau_h, h.unsqueeze(0)), 0)
    tau_h_hat = torch.cat((tau_h_hat, h_hat.unsqueeze(0)), 0)
    # print(torch.cat((h.unsqueeze(0), h_hat.unsqueeze(0)), 0))
    
# print(tau_h)
# print(tau_h_hat)
# print((tau_h - tau_h_hat))
# print(torch.view_as_complex((tau_h - tau_h_hat).reshape(num_trajectories, n_T*n_R ,2)))
# print(tau[1][0][2])
# print(tau.shape)
## compute loss and update
    # tau = [[row[0] for row in tau], [row[1] for row in tau]]  # reshape #####
    # print(np.array(ct[0])-np.array(ct[1]))
# norm = np.linalg.norm(h-np.double(np.array(h_hat)).view(np.complex128))
norm = torch.norm(torch.view_as_complex((tau_h - tau_h_hat).reshape(num_trajectories, n_T*n_R ,2)), dim=1)
# print(norm)
# batch_loss = lamb*norm.mean()*Normal(logits[:len(logits)//2],logits[len(logits//2)]).log_prob(h_hat[l])

batch_loss = lamb*norm.mean()*Normal(logits_net(tau_y)[:, :logits.size(dim=0)//2], 
                                     logits_net(tau_y)[:, logits.size(dim=0)//2:]).log_prob(tau_h_hat)
# print(Normal(logits_net(tau_y)[:, :logits.size(dim=0)//2], logits_net(tau_y)[:, logits.size(dim=0)//2:]))
# print(tau_h_hat.shape)
print(batch_loss.mean())
print(len(logits))

# print(logits_net(tau_y))
# print(logits_net(tau_y)[:, :logits.size(dim=0)//2])
# print(logits_net(tau_y)[:, logits.size(dim=0)//2:])
# print(logits.size(dim=0))
# print(logits_net(tau_y)[0, :tau_h_hat.size(dim=1)//2].size())
# print(Normal(torch.tensor([1, 11]), torch.tensor([0.1, 0.01])))
# print(Normal(torch.))

# print(logits)

# print(batch_loss)
# print(tau_h_hat)
# print(tau_h_hat[:, :tau_h_hat.size(dim=1)//2])
# print(tau_h_hat.size(dim=1)//2)

# a = 3
# print(type(a))
# a += norm.mean()
# print(a)
# print(a.type())

# a, b = torch.tensor([1, 2])
# print(a, b)
