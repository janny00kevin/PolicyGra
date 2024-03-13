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

def train(size, hidden_sizes=[64, 32], lr=1e-3, iteration=1000, batch_size=10, 
          channel_information = [5, 0.2, 0, 0.01], epoch = 100):
    start_time = time.time()
    n_T, n_R, T = size
    # generate the multilayer peceptron

    H_mean, H_sigma, W_mean, W_sigma = channel_information
    
    X = np.tile(np.eye(n_T), int(T/n_T))
    print(X)
    
    epoch_loss = []
    for i in range(epoch):
        logits_net = mlp(sizes=[2*n_R*T]+hidden_sizes+[2*2*n_R*n_T])
        optimizer = Adam(logits_net.parameters(), lr=lr)
        lamb, t = [1, 0]
        iter_loss = []
        iterations = []
        for j in range(iteration):
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
                iter_loss.append(norm)
                iterations.append(j)
            loss.backward()
            optimizer.step()
            t -= lr*(1-lamb)
        epoch_loss.append(iter_loss)
    

    end_time = time.time()
    print("time:", round((end_time-start_time),2), " sec")
    # plot the loss
    epoch_loss = np.array(epoch_loss).mean(-2)
    plt.plot(iterations, epoch_loss)
    plt.title('epoch:%s, $[n_T,n_R,T]$:[%s,%s,%s], lr:%s' %(epoch,n_T,n_R,T,lr) )
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()

if __name__ == '__main__':
    iteration = 4000
    epoch = 100
    batch_size = 50
    n_T = 1
    n_R = 1
    T = n_T*1
    size = [n_T, n_R, T]
    hidden_sizes=[64,32]
    lr = 1e-3
    H_mean = 5
    H_sigma = 0.2
    W_mean = 0
    W_sigma = 0.1
    channel_information = [H_mean, H_sigma, W_mean, W_sigma]
    train(size, hidden_sizes, lr, iteration, batch_size, channel_information, epoch)
    
    