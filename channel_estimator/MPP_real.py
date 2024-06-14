import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
from tqdm import tqdm

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
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
    X = torch.eye(n_T).tile(T//n_T).to(device)
    H = torch.normal(H_mean, H_sigma, size=(data_size, n_R, n_T)).to(device)
    W = torch.normal(W_mean, W_sigma, size=(data_size, n_R, T)).to(device)
    Y = H.matmul(X) + W
    h = H.reshape(data_size, n_R*n_T)
    y = Y.reshape(data_size, n_R*T)

    ## generate testing data
    # H_test = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(n_R*test_size, n_T, 2))).to(device)
    # W_test = torch.view_as_complex(torch.normal(W_mean, W_sigma, size=(n_R*test_size, T, 2))).to(device)
    # Y_test = H_test.matmul(X) + W_test
    # h_test = torch.view_as_real(H_test).reshape(test_size, n_R*n_T*2)
    # y_test = torch.view_as_real(Y_test).reshape(test_size, n_R*T*2)
    # h_test_norm = torch.norm((h_test).reshape(test_size, n_T*n_R*2), dim=1)**2
    
    ## generate the multilayer peceptron
    logits_net = mlp(sizes=[n_R*T]+hidden_sizes+[2*n_R*n_T]).to(device)
    optimizer = Adam(logits_net.parameters(), lr=lr)
    lamb, t = torch.tensor([10., 1000.]).to(device)

    iter_loss_M = torch.zeros(num_epochs*num_minibatch).to(device)
    iter_loss_N = torch.zeros(num_epochs*num_minibatch).to(device)
    testing_loss_M = torch.zeros(num_epochs*num_minibatch).to(device)
    testing_loss_N = torch.zeros(num_epochs*num_minibatch).to(device)
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
                tau_h_hat = Normal(logits[:,::2], torch.exp(logits[:,1::2])).sample()
                # if itr == 10:
                    # print(Normal(logits[:,::2], torch.exp(logits[:,1::2])).sample().requires_grad)
                
                ## compute loss and update
                optimizer.zero_grad()
                ## 2-norm of h - h_hat square at each realization (num_trajectories by 1) 
                norm = torch.norm((tau_h - tau_h_hat), dim=1)**2
                # if itr == 10:
                    # print((Normal(logits_net(tau_y)[:, ::2], 
                    #                     torch.exp(logits_net(tau_y)[:, 1::2])).log_prob(tau_h_hat)).requires_grad)
                    # print("tau_h:", tau_h.requires_grad)
                    # print("tau_h_hat:", tau_h_hat.requires_grad)
                ## lambda update
                lamb = max(0, lamb + lr/itr*(norm.mean() - t)) 
                ## model parameters update with policy gradient
                # batch_loss = (lamb*norm.unsqueeze(1)*(Normal(logits_net(tau_y)[:, ::2], 
                #                         torch.exp(logits_net(tau_y)[:, 1::2])).log_prob(tau_h_hat))).mean() ###
                batch_loss = lamb*(norm.unsqueeze(1)*(Normal(logits[:, ::2], torch.exp(logits[:, 1::2])).log_prob(tau_h_hat))).mean()
                # if itr == 10:
                #     print((Normal(logits[:, ::2], torch.exp(logits[:, 1::2])).log_prob(tau_h_hat)).size())
                #     print(norm.size())
                #     print(norm.unsqueeze(1).size())

                batch_loss.backward()
                optimizer.step()
                ## t update
                t -= lr/itr*(1-lamb)
                ## save the loss to plot
                iter_loss_M[itr-1] = norm.mean() / (n_T*n_R)  ### nRnT
                iter_loss_N[itr-1] = (norm / torch.norm((tau_h).reshape(num_trajectories, n_T*n_R), dim=1)**2).mean()
                # if itr == 10:
                #     print(torch.norm((tau_h).reshape(n_T*n_R, num_trajectories), dim=0) == 
                #           torch.norm((tau_h).reshape(num_trajectories, n_T*n_R), dim=1
                #                      ))
                
                ## validation
                # logits = logits_net(y_test)
                # h_hat_test = Normal(logits[:,::2], torch.exp(logits[:,1::2])).sample()
                # testing_loss_M[itr-1] = (torch.norm((h_test - h_hat_test).reshape(test_size, n_T*n_R*2), dim=1)**2).mean() / (n_R*n_T)  ### nRnT
                # testing_loss_N[itr-1] = torch.norm(h_test - h_hat_test)**2 / torch.norm(h_test)**2
                pbar.update(1)
            ## after 1 epoch, shuffle the dataset, using the same index to shuffle H and Y
            # H_shuffle = H.reshape(data_size, n_R, n_T)
            # Y_shuffle = Y.reshape(data_size, n_R ,T)
            # idx = torch.randperm(Y_shuffle.shape[0])
            # Y_shuffle = Y_shuffle[idx,:,:]
            # H_shuffle = H_shuffle[idx,:,:]
            # Y = Y_shuffle.reshape(n_R*data_size, T) 
            # H = H_shuffle.reshape(n_R*data_size, n_T)

    ## plot the loss
    # print(lamb, t)
    epochs = range(1, num_epochs+1)
    plt.plot(epochs, iter_loss_N.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='training loss')
    # plt.plot(epochs, testing_loss_N.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation loss')
    plt.suptitle("MMSE based PD with PG channel estimator")
    plt.title(' $R[n_R,n_T,T]$:[%s,%s,%s], lr:%s, size:%s, $\\mu_H$:%s' 
                 %(n_R,n_T,T,lr, [n_R*T]+hidden_sizes+[2*n_R*n_T], H_mean))
    plt.xlabel('epochs')
    plt.ylabel('NMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig('./simulation/result/NR_lr%s_%s_ep%s_mu%s.png' %(lr, [n_R*T]+hidden_sizes+[2*n_R*n_T],num_epochs,H_mean))
    plt.close()

    plt.plot(epochs, iter_loss_M.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='training loss')
    # plt.plot(epochs, testing_loss_M.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation loss')
    plt.suptitle("MMSE based PD with PG channel estimator")
    plt.title(' $R[n_R,n_T,T]$:[%s,%s,%s], lr:%s, size:%s, $\\mu_H$:%s'
                 %(n_R,n_T,T,lr, [n_R*T]+hidden_sizes+[2*n_R*n_T], H_mean))
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig('./simulation/result/MR_lr%s_%s_ep%s_mu%s.png' %(lr, [n_R*T]+hidden_sizes+[2*n_R*n_T],num_epochs,H_mean))

    ## save the model
    torch.save(logits_net, './simulation/result/Rlr%s_%s_ep%s_mu%s.pt' %(lr, [n_R*T]+hidden_sizes+[2*n_R*n_T],num_epochs,H_mean))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hs', type=float, default=1) # sigma of H
    parser.add_argument('--hm', type=float, default=0) # mean of H
    parser.add_argument('--lr', type=float, default=1e-4) # learning rate
    parser.add_argument('--ep', type=int, default=10)    # num of epochs
    parser.add_argument('--tau', type=int, default=4)  # 10k length of trajectory = size of mini-batch
    parser.add_argument('--nmb', type=int, default=100)  # 10 number of mini-batch
    parser.add_argument('--nR', type=int, default=4)  
    parser.add_argument('--nT', type=int, default=36)  
    parser.add_argument('--hsz', type=int, default=1)  # hidden layer size
    parser.add_argument('--cuda', type=int, default=1)  # cuda
    args = parser.parse_args()

    print("cuda:%s"%(args.cuda), torch.cuda.is_available())

    if args.hsz == 1:
        hidden_sizes=[64, 32]
    else:
        hidden_sizes=[2048, 1024]

    T = args.nT*1
    # H_mean = 0
    W_mean, W_sigma = 0, 0.1
    channel_information = [args.hm, args.hs, W_mean, W_sigma]

    train([args.nT, args.nR, T], hidden_sizes, args.lr, channel_information, args.ep, 10**args.tau, args.nmb, args.cuda)
