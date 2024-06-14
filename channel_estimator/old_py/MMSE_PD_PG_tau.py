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

def train(size, hidden_sizes=[64, 32], lr=1e-3, num_iterations=1000, itr_batch_size=10, 
          channel_information = [5, 0.2, 0, 0.01], num_epochs = 100, num_trajectories = 10):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    start_time = time.time()
    n_T, n_R, T = size

    H_mean, H_sigma, W_mean, W_sigma = channel_information
    
        # X = np.tile(np.eye(n_T), int(T/n_T))
    X = torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T)).to(device)
    # print(X)
    
    epoch_loss = []
    for i in range(num_epochs):
        ## generate the multilayer peceptron
        logits_net = mlp(sizes=[2*n_R*T]+hidden_sizes+[2*2*n_R*n_T]).to(device)
        optimizer = Adam(logits_net.parameters(), lr=lr)
        lamb, t = torch.tensor([1., 0.]).to(device)
        iter_loss = []
        iterations = []
        for j in range(num_iterations):
            tau_h_hat = tau_h = tau_y= torch.zeros(0)
            ## generate trajectories
            for k in range(num_trajectories):
                ## generate communication data
                H = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(n_R, n_T, 2))).to(device)
                    # H = np.random.normal(H_mean, H_sigma, [n_R,2*n_T]).view(np.complex128)
                W = torch.view_as_complex(torch.normal(W_mean, W_sigma, size=(n_R, T, 2))).to(device)
                    # W = np.random.normal(W_mean, W_sigma, [n_R,T*2]).view(np.complex128)
                Y = H.matmul(X) + W
                    # Y = np.matmul(H,X) + W

                ## forward propagation
                    # h = H.transpose().reshape(-1).view(np.double)
                h = torch.flatten(torch.view_as_real(H))
                    # y = Y.transpose().reshape(-1).view(np.double)
                y = torch.flatten(torch.view_as_real(Y))
                logits = logits_net(y)
                h_hat = Normal(logits[:len(logits)//2], logits[len(logits)//2:]).sample()
                    # tau.append([h, h_hat])
                tau_y = torch.cat((tau_y, y.unsqueeze(0)), 0)
                tau_h = torch.cat((tau_h, h.unsqueeze(0)), 0)
                tau_h_hat = torch.cat((tau_h_hat, h_hat.unsqueeze(0)), 0)

            ## compute loss and update
            optimizer.zero_grad()
                # tau = [[row[0] for row in tau], [row[1] for row in tau]]  # reshape #####
                # print(np.array(ct[0])-np.array(ct[1]))
                # norm = np.linalg.norm(h-np.double(np.array(h_hat)).view(np.complex128))
            norm = torch.norm(torch.view_as_complex((tau_h - tau_h_hat).reshape(num_trajectories, n_T*n_R ,2)), dim=1)
            lamb += lr*(norm.mean() - t)  ###
            batch_loss = lamb*(norm.unsqueeze(1)*Normal(logits_net(tau_y)[:, :logits.size(dim=0)//2], 
                                     logits_net(tau_y)[:, logits.size(dim=0)//2:]).log_prob(tau_h_hat)).mean() ###
            if (j)%itr_batch_size == 0:
                iter_loss.append(norm.mean())
                iterations.append(j)
            batch_loss.backward()
            optimizer.step()
            t -= lr*(1-lamb)
        epoch_loss.append(iter_loss)

    end_time = time.time()
    print("time:", round((end_time-start_time),2), " sec")
    ## plot the loss
    epoch_loss = np.array(epoch_loss).mean(-2)
    # print(epoch_loss.size())
    plt.plot(iterations, epoch_loss)
    plt.suptitle("MMSE based PD with PG channel estimator")
    plt.title('epoch:%s, $[n_T,n_R,T]$:[%s,%s,%s], lr:%s, $\\sigma_H$:%s, |D|:%s' 
                 %(num_epochs,n_T,n_R,T,lr,H_sigma,num_trajectories))
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.ylim([0, 2])
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hs', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ep', type=int, default=10)
    parser.add_argument('--tau', type=int, default=10)
    args = parser.parse_args()

    print(torch.cuda.is_available())
    num_iterations = 500
    # num_epochs = 10
    itr_batch_size = num_iterations//200
    # num_trajectories = 1

    hidden_sizes=[64,32]
    lr = 1e-4
    
    n_T, n_R= 8, 4
    T = n_T*1
    H_mean = 0
    W_mean, W_sigma = 0, 0.1

    channel_information = [H_mean, args.hs, W_mean, W_sigma]
    train([n_T, n_R, T], hidden_sizes, args.lr , num_iterations, itr_batch_size, channel_information, args.ep, args.tau)