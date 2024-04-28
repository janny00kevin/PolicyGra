import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
from tqdm import tqdm

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Softplus):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(size, hidden_sizes=[64, 32], lr=1e-3, 
          channel_information = [0, 1, 0, 0.1], num_epochs = 3, num_trajectories = 1e4, num_minibatch = 10):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    n_T, n_R, T = size
    data_size = num_minibatch*num_trajectories
    H_mean, H_sigma, W_mean, W_sigma = channel_information
    test_size = 2000
    
    ## generate communication data to train the parameterized policy
    X = torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T)).to(device)
    H = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(n_R*data_size, n_T, 2))).to(device)
    W = torch.view_as_complex(torch.normal(W_mean, W_sigma, size=(n_R*data_size, T, 2))).to(device)
    Y = H.matmul(X) + W

    ## generate testing data
    H_test = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(n_R*test_size, n_T, 2))).to(device)
    W_test = torch.view_as_complex(torch.normal(W_mean, W_sigma, size=(n_R*test_size, T, 2))).to(device)
    Y_test = H_test.matmul(X) + W_test
    
    ## generate the multilayer peceptron
    logits_net = mlp(sizes=[2*n_R*T]+hidden_sizes+[2*2*n_R*n_T]).to(device)
    optimizer = Adam(logits_net.parameters(), lr=lr)
    lamb, t = torch.tensor([1., 0.]).to(device)

    iter_loss = torch.zeros(num_epochs*num_minibatch).to(device)
    testing_loss = torch.zeros(num_epochs*num_minibatch).to(device)
    ## plot progress bar
    with tqdm(total = num_epochs*num_minibatch) as pbar:
        pbar.set_description('Processing:')
        for i in range(num_epochs):
            for j in range(num_minibatch):
                itr = j+i*num_minibatch+1
                tau_h_hat = tau_h = tau_y= torch.zeros(0).to(device)
                ## generate trajectories
                for k in range(num_trajectories):
                    ## extract data from the dataset
                    index = j*num_trajectories + k
                    h = torch.flatten(torch.view_as_real(H[n_R*index:n_R*index+n_R,:]))
                    y = torch.flatten(torch.view_as_real(Y[n_R*index:n_R*index+n_R,:]))
                    ## forward propagation and sample from the distribution
                    logits = logits_net(y)
                    h_hat = Normal(logits[::2], logits[1::2]).sample()
                    ## record the data and output
                    tau_y = torch.cat((tau_y, y.unsqueeze(0)), 0)
                    tau_h = torch.cat((tau_h, h.unsqueeze(0)), 0)
                    tau_h_hat = torch.cat((tau_h_hat, h_hat.unsqueeze(0)), 0)
                ## compute loss and update
                optimizer.zero_grad()
                ## 2-norm of h - h_hat square at each realization (num_trajectories by 1) 
                norm = torch.norm(torch.view_as_complex((tau_h - tau_h_hat).reshape(num_trajectories, n_T*n_R, 2)), dim=1)**2
                ## lambda update
                lamb = max(0, lamb + lr/itr*(norm.mean() - t))  ###
                ## model parameters update with policy gradient
                batch_loss = lamb*(norm.unsqueeze(1)*Normal(logits_net(tau_y)[:, ::2], 
                                        logits_net(tau_y)[:, 1::2]).log_prob(tau_h_hat)).mean() ###
                batch_loss.backward()
                optimizer.step()
                ## t update
                t -= lr/itr*(1-lamb)
                ## save the loss to plot
                iter_loss[itr-1] = norm.mean()/(n_R*n_T)
                ## validation
                for k in range(test_size):
                    h_test = torch.flatten(torch.view_as_real(H_test[n_R*k:n_R*k+n_R,:]))
                    y_test = torch.flatten(torch.view_as_real(Y_test[n_R*k:n_R*k+n_R,:]))
                    logits = logits_net(y_test)
                    h_hat_test = Normal(logits[::2], logits[1::2]).sample()
                    testing_loss[itr-1] += (torch.norm(torch.view_as_complex((h_test - h_hat_test).reshape(n_T*n_R ,2)))**2)\
                                            .mean()/test_size/(n_T*n_R)
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
    iterations = range(1, num_epochs*num_minibatch+1)
    plt.plot(iterations, torch.Tensor(iter_loss).to("cpu"),iterations, torch.Tensor(testing_loss).to("cpu"))
    plt.suptitle("MMSE based PD with PG channel estimator")
    # plt.title('epoch:%s, $[n_R,n_T,T]$:[%s,%s,%s], lr:%s, $\\sigma_H$:%s, |D|:%s' 
    #              %(num_epochs,n_R,n_T,T,lr,H_sigma,num_trajectories))
    plt.title(' $[n_R,n_T,T]$:[%s,%s,%s], lr:%s, size:%s,%s,%s' 
                 %(n_R,n_T,T,lr, n_R*T*2, hidden_sizes, n_R*n_T*2))
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.legend('training loss', 'validation loss')
    # plt.ylim([1.5, 4.5])
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hs', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ep', type=int, default=10)    # 3
    parser.add_argument('--tau', type=int, default=4)  # 10k length of trajectory = size of mini-batch
    parser.add_argument('--nmb', type=int, default=10)  # 10 number of mini-batch
    args = parser.parse_args()

    print("cuda:", torch.cuda.is_available())

    hidden_sizes=[64,32]

    n_R, n_T = 1, 1
    T = n_T*1
    H_mean = 0
    W_mean, W_sigma = 0, 0.1
    channel_information = [H_mean, args.hs, W_mean, W_sigma]

    train([n_T, n_R, T], hidden_sizes, args.lr, channel_information, args.ep, 10**args.tau, args.nmb)