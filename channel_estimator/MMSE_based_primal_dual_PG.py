import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.distributions.normal import Normal

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Softplus):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(size, hidden_sizes=[64, 32], lr=1e-2):
    # generate the multilayer peceptron
    n_T, n_R, T = size
    logits_net = mlp(sizes=[2*n_R*T]+hidden_sizes+[2*2*n_R*n_T])
    optimizer = Adam(logits_net.parameters(), lr=lr)
    lamb, t = [0, 0]

    # generate the data
    H_sigma = 5
    H_mean = 0
    H = np.random.normal(H_mean, H_sigma, [n_R,2*n_T]).view(np.complex128)
    W_mean = 0
    W_sigma = 3
    W = np.random.normal(W_mean, W_sigma, [n_R,T*2]).view(np.complex128)
    X = np.tile(np.eye(n_T), int(T/n_T))
    Y = np.matmul(H,X) + W

    # forward propagation
    y = Y.transpose().reshape(-1).view(np.double)
    h_hat = []
    # print(y.shape)
    # print(logits_net)
    # dist = logits_net(torch.as_tensor(y, dtype=torch.float32))
    ## dist = dist.detach().numpy().reshape(2*n_R*n_T,2)
    logits = logits_net(torch.as_tensor(y, dtype=torch.float32))
    # print(Normal(-1, 0).sample())
    for i in range(int(len(logits)/2)):
        # print(logits[2*i], logits[2*i+1])
        h_hat.append(Normal(logits[2*i],logits[2*i+1]).sample())
        # print(Normal(logits[2*i], logits[2*i+1]).sample())
    # # h_hat = np.zeros(2*n_R*n_T)
    # print(dist)
    # print(len(dist))
    # for i in range(len(dist)):
    #     h_hat[i] = np.random.normal(dist[i][0],dist[i][1])
    # h_hat = h_hat.view(np.complex128)
    # print(h_hat.shape)

    # compute loss and update
    # optimizer.zero_grad()
    # h = H.transpose().reshape(-1)
    # norm = np.linalg.norm(h-h_hat)
    # print(PG)
    # lamb += 



if __name__ == '__main__':
    size = [2, 4, 8]   # n_T, n_R, T
    train(size)