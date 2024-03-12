import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt 
import time

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Softplus):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(size, hidden_sizes=[64, 32], lr=1e-3, epoch=1000, batch_size=10):
    start_time = time.time()
    # generate the multilayer peceptron
    n_T, n_R, T = size
    logits_net = mlp(sizes=[2*n_R*T]+hidden_sizes+[2*2*n_R*n_T])
    optimizer = Adam(logits_net.parameters(), lr=lr)
    lamb, t = [1, 0]

    H_mean = 5
    H_sigma = 0.2
    
    W_mean = 0
    W_sigma = 0.1
    
    X = np.tile(np.eye(n_T), int(T/n_T))
    print(X)

    batch_loss = []
    epochs = []
    

    for j in range(epoch):
        # generate the data
        H = np.random.normal(H_mean, H_sigma, [n_R,2*n_T]).view(np.complex128)
        W = np.random.normal(W_mean, W_sigma, [n_R,T*2]).view(np.complex128)
        Y = np.matmul(H,X) + W

        # forward propagation
        y = Y.transpose().reshape(-1).view(np.double)
        h_hat = []
        logits = logits_net(torch.as_tensor(y, dtype=torch.float32))
        for i in range(int(len(logits)/2)):
            h_hat.append(Normal(logits[2*i],logits[2*i+1]).sample())

        # compute loss and update
        optimizer.zero_grad()
        h = H.transpose().reshape(-1)
        norm = np.linalg.norm(h-np.double(np.array(h_hat)).view(np.complex128))
        lamb += lr*(norm-t)
        PG = torch.zeros(int(len(logits)/2))
        for i in range(int(len(logits)/2)):
            PG[i] = lamb*norm*Normal(logits[2*i],logits[2*i+1]).log_prob(h_hat[i])
        loss = PG.mean()
        if (j)%batch_size == 0:
            # print(j)
            batch_loss.append(norm)
            epochs.append(j)
        loss.backward()
        optimizer.step()
        t -= lr*(1-lamb)

    end_time = time.time()
    print("time:", round((end_time-start_time),2), " sec")
    # plot the loss
    plt.plot(epochs,batch_loss)
    plt.show()

if __name__ == '__main__':
    epoch = 10000
    batch_size = 100
    n_T = 2
    n_R = 1
    T = n_T*1
    size = [n_T, n_R, T]
    hidden_sizes=[64,32]
    lr = 5e-4
    train(size, epoch=epoch, batch_size=batch_size, hidden_sizes=hidden_sizes , lr=lr)
    
    