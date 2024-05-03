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

def train(size, hidden_sizes=[64, 32], lr=1e-3, channel_information = [0, 1, 0, 0.1],
           num_epochs = 3, num_trajectories = 1e4, num_minibatch = 10, cuda = 1):
    device = torch.device("cuda:%s"%(cuda) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    n_T, n_R, T = size
    data_size = num_minibatch*num_trajectories
    H_mean, H_sigma, W_mean, W_sigma = channel_information
    test_size = 2000
    
    ## generate communication data to train the parameterized policy
    X = torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T)).to(device)
    H = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(n_R*data_size, n_T, 2))).to(device)
    W = torch.view_as_complex(torch.normal(W_mean, W_sigma, size=(n_R*data_size, T, 2))).to(device)
    Y = H.matmul(X) + W
    h = torch.view_as_real(H).reshape(data_size, n_R*n_T*2)
    y = torch.view_as_real(H).reshape(data_size, n_R*T*2)

    ## generate testing data
    H_test = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(n_R*test_size, n_T, 2))).to(device)
    W_test = torch.view_as_complex(torch.normal(W_mean, W_sigma, size=(n_R*test_size, T, 2))).to(device)
    Y_test = H_test.matmul(X) + W_test
    h_test = torch.view_as_real(H_test).reshape(test_size, n_R*n_T*2)
    y_test = torch.view_as_real(Y_test).reshape(test_size, n_R*T*2)
    h_test_norm = torch.norm((h_test).reshape(test_size, n_T*n_R*2), dim=1)**2
    
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
                ## trajectories training data
                tau_y = y[j*num_trajectories:(j+1)*num_trajectories, :]
                tau_h = h[j*num_trajectories:(j+1)*num_trajectories, :]
                logits = logits_net(tau_y)
                tau_h_hat = Normal(logits[:,::2], logits[:,1::2]).sample()
                
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
                iter_loss[itr-1] = (norm / torch.norm((tau_h).reshape(num_trajectories, n_T*n_R*2), dim=1)**2).mean()
                
                ## validation
                logits = logits_net(y_test)
                h_hat_test = Normal(logits[:,::2], logits[:,1::2]).sample()
                testing_loss[itr-1] = ((torch.norm((h_test - h_hat_test).reshape(test_size, n_T*n_R*2), dim=1)**2)/
                                       h_test_norm).mean()
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
    epochs = range(1, num_epochs+1)
    plt.plot(epochs, iter_loss.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='training loss')
    plt.plot(epochs, testing_loss.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation loss')
    plt.suptitle("MMSE based PD with PG channel estimator")
    # plt.title('epoch:%s, $[n_R,n_T,T]$:[%s,%s,%s], lr:%s, $\\sigma_H$:%s, |D|:%s' 
    #              %(num_epochs,n_R,n_T,T,lr,H_sigma,num_trajectories))
    plt.title(' $[n_R,n_T,T]$:[%s,%s,%s], lr:%s, size:%s' 
                 %(n_R,n_T,T,lr, [2*n_R*T]+hidden_sizes+[2*2*n_R*n_T]))
    plt.xlabel('epochs')
    plt.ylabel('NMSE')
    plt.legend()
    # plt.ylim([2, 4])
    plt.grid(True)
    # plt.show()
    plt.savefig('./simulation/result/lr%s_%s_ep%s.png' %(lr, [2*n_R*T]+hidden_sizes+[2*2*n_R*n_T],num_epochs))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hs', type=float, default=1) # sigma of H
    parser.add_argument('--lr', type=float, default=1e-3) # learning rate
    parser.add_argument('--ep', type=int, default=10)    # num of epochs
    parser.add_argument('--tau', type=int, default=4)  # 10k length of trajectory = size of mini-batch
    parser.add_argument('--nmb', type=int, default=10)  # 10 number of mini-batch
    parser.add_argument('--nR', type=int, default=4)  
    parser.add_argument('--nT', type=int, default=36)  
    parser.add_argument('--hsz', type=int, default=512)  # hidden layer size
    parser.add_argument('--cuda', type=int, default=1)  # cuda
    args = parser.parse_args()

    print("cuda:%s"%(args.cuda), torch.cuda.is_available())

    if args.hsz == 512:
        hidden_sizes=[512, 256, 256]
    else:
        hidden_sizes=[64, 32]

    T = args.nT*1
    H_mean = 0
    W_mean, W_sigma = 0, 0.1
    channel_information = [H_mean, args.hs, W_mean, W_sigma]

    train([args.nT, args.nR, T], hidden_sizes, args.lr, channel_information, args.ep, 10**args.tau, args.nmb, args.cuda)
