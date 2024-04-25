import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt 
import time
from tqdm import tqdm

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Softplus):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(size, hidden_sizes=[64, 32], lr=1e-3, num_iterations=1000, itr_batch_size=10, 
          channel_information = [5, 0.2, 0, 0.01], num_epochs = 3, num_trajectories = 10, num_minibatch = 10):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    n_T, n_R, T = size
    data_size = num_minibatch*num_trajectories
    H_mean, H_sigma, W_mean, W_sigma = channel_information
    
    ## generate communication data to train the parameterized policy
    X = torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T)).to(device)
    H = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(n_R*data_size, n_T, 2))).to(device)
    W = torch.view_as_complex(torch.normal(W_mean, W_sigma, size=(n_R*data_size, T, 2))).to(device)
    Y = H.matmul(X) + W
    
    ## generate the multilayer peceptron
    logits_net = mlp(sizes=[2*n_R*T]+hidden_sizes+[2*2*n_R*n_T]).to(device)
    optimizer = Adam(logits_net.parameters(), lr=lr)
    lamb, t = torch.tensor([1., 0.]).to(device)

    iter_loss = []
    iterations = []
    ## plot progress bar
    with tqdm(total = num_epochs*num_minibatch) as pbar:
        pbar.set_description('Processing:')
        for i in range(num_epochs):
            for j in range(num_minibatch):
                itr = j+i*num_minibatch+1
                # print("iteration: ", itr)
                tau_h_hat = tau_h = tau_y= torch.zeros(0).to(device)
                ## generate trajectories
                for k in range(num_trajectories):
                    ## extract data from the dataset
                    index = j*num_trajectories + k
                    h = torch.flatten(torch.view_as_real(H[n_R*index:n_R*index+n_R,:]))
                    y = torch.flatten(torch.view_as_real(Y[n_R*index:n_R*index+n_R,:]))
                    ## forward propagation and sample from the distribution
                    logits = logits_net(y)
                    h_hat = Normal(logits[:len(logits)//2], logits[len(logits)//2:]).sample()
                    ## record the data and output
                    tau_y = torch.cat((tau_y, y.unsqueeze(0)), 0)
                    tau_h = torch.cat((tau_h, h.unsqueeze(0)), 0)
                    tau_h_hat = torch.cat((tau_h_hat, h_hat.unsqueeze(0)), 0)
                ## compute loss and update
                optimizer.zero_grad()
                ## 2-norm of h - h_hat square at each realization (num_trajectories by 1) 
                norm = torch.norm(torch.view_as_complex((tau_h - tau_h_hat).reshape(num_trajectories, n_T*n_R ,2)), dim=1)**2###
                ## lambda update
                lamb += lr/itr*(norm.mean() - t)  ###
                ## model parameters update with policy gradient
                batch_loss = lamb*(norm.unsqueeze(1)*Normal(logits_net(tau_y)[:, :logits.size(dim=0)//2], 
                                        logits_net(tau_y)[:, logits.size(dim=0)//2:]).log_prob(tau_h_hat)).mean() ###
                batch_loss.backward()
                optimizer.step()
                ## t update
                t -= lr/itr*(1-lamb)
                ## save the loss to plot
                iter_loss.append(norm.mean()/(n_R*n_T))
                # iter_loss.append(batch_loss.detach().numpy())
                iterations.append(j+i*num_minibatch)
                pbar.update(1)
            ## after 1 epoch, shuffle the dataset, using the same index to shuffle H and Y
            H_shuffle = H.reshape(data_size, n_R, n_T)
            Y_shuffle = Y.reshape(data_size, n_R ,T)
            idx = torch.randperm(Y_shuffle.shape[0])
            Y_shuffle = Y_shuffle[idx,:,:]
            H_shuffle = H_shuffle[idx,:,:]
            Y = Y_shuffle.reshape(n_R*data_size, T) ###
            H = H_shuffle.reshape(n_R*data_size, n_T)

    ## plot the loss
    plt.plot(iterations, iter_loss)
    plt.suptitle("MMSE based PD with PG channel estimator")
    plt.title('epoch:%s, $[n_T,n_R,T]$:[%s,%s,%s], lr:%s, $\\sigma_H$:%s, |D|:%s' 
                 %(num_epochs,n_T,n_R,T,lr,H_sigma,num_trajectories))
    plt.xlabel('iterations')
    plt.ylabel('loss')
    # plt.ylim([0, 2])
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hs', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ep', type=int, default=1)    # 3
    parser.add_argument('--tau', type=int, default=10)  # 10k length of trajectory = size of mini-batch
    parser.add_argument('--nmb', type=int, default=10)  # 10number of mini-batch
    args = parser.parse_args()

    print(torch.cuda.is_available())
    num_iterations = 500
    # num_epochs = 10
    itr_batch_size = num_iterations//200
    # num_trajectories = 1

    hidden_sizes=[64,32]

    n_R, n_T = 1, 1
    T = n_T*1
    H_mean = 0
    W_mean, W_sigma = 0, 0.1

    channel_information = [H_mean, args.hs, W_mean, W_sigma]
    train([n_T, n_R, T], hidden_sizes, args.lr , num_iterations, itr_batch_size, channel_information, args.ep, 10**args.tau, args.nmb)