import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
from tqdm import tqdm

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        # Convolution 1 , input_shape=(:,3,12,12)
        self.cnn1 = nn.Conv2d(3,256,3) #output_shape=(:,256,10,10)
        self.cnn2 = nn.Conv2d(256,256,3) #output_shape=(:,256,8,8)
        self.cnn3 = nn.Conv2d(256,256,3) #output_shape=(:,256,6,6)
        # Fully connected 1 ,#input_shape=(32*4*4)
        self.fc1 = nn.Linear(256*6*6, 1024, nn.Linear)
        self.fc2 = nn.Linear(1024, 2048, nn.Linear)
        self.fc3 = nn.Linear(2048, 4*36*2*2, nn.Linear)
        self.dp = nn.Dropout(p=0.5)
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.cnn2(out)
        out = self.cnn3(out)
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.fc1(out)
        out = self.dp(out)
        out = self.fc2(out)
        out = self.dp(out)
        out = self.fc3(out)
        return out

def train(size, hidden_sizes=[64, 32], lr=1e-3, channel_information = [0, 1, 0, 0.1],
           num_epochs = 3, num_trajectories = 1e4, num_minibatch = 10, cuda = 1):
    device = torch.device("cuda:%s"%(cuda) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    n_T, n_R, T = size
    data_size = num_minibatch*num_trajectories
    H_mean, H_sigma, W_mean, W_sigma = channel_information
    test_size = 2000
    
    sqrt = 2**0.5
    ## generate communication data to train the parameterized policy
    X = torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T)).to(device)
    H = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(data_size, n_R, n_T, 2))/sqrt).to(device)
    W = torch.view_as_complex(torch.normal(W_mean, W_sigma, size=(data_size, n_R, T, 2))).to(device)
    Y = H.matmul(X) + W
    # h = torch.view_as_real(H).reshape(data_size, n_R*n_T*2)
    # y = torch.view_as_real(Y).reshape(data_size, n_R*T*2)

    ## generate testing data
    H_test = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(test_size, n_R, n_T, 2))/sqrt).to(device)
    W_test = torch.view_as_complex(torch.normal(W_mean, W_sigma, size=(test_size, n_R, T, 2))).to(device)
    Y_test = H_test.matmul(X) + W_test
    h_test = torch.view_as_real(H_test).reshape(test_size, n_R*n_T*2)
    # y_test = torch.view_as_real(Y_test).reshape(test_size, n_R*T*2)
    # h_test_norm = torch.norm((h_test).reshape(test_size, n_T*n_R*2), dim=1)**2
    
    ## generate the multilayer peceptron
    # logits_net = mlp(sizes=[2*n_R*T]+hidden_sizes+[2*2*n_R*n_T]).to(device)
    logits_net = cnn().to(device)
    # logits = logits_net(torch.randn(2,3,12,12).to(device))           ######################
    # print(logits.size())
    optimizer = Adam(logits_net.parameters(), lr=lr)
    lamb, t = torch.tensor([1., 0.]).to(device)

    iter_loss_M = torch.zeros(num_epochs*num_minibatch).to(device)
    iter_loss_N = torch.zeros(num_epochs*num_minibatch).to(device)
    testing_loss_M = torch.zeros(num_epochs*num_minibatch).to(device)
    testing_loss_N = torch.zeros(num_epochs*num_minibatch).to(device)
    ## plot progress bar
    pbar = tqdm(total = num_epochs*num_minibatch)
    for i in range(num_epochs):
        for j in range(num_minibatch):
            itr = j+i*num_minibatch+1
            ## trajectories training data
            tau_Y = Y[j*num_trajectories:(j+1)*num_trajectories, :]
            tau_h = torch.view_as_real(H[j*num_trajectories:(j+1)*num_trajectories, :]).reshape(num_trajectories, n_T*n_R*2)
            if itr == 10:
                print(torch.stack([tau_Y.real,tau_Y.imag,tau_Y.abs()],dim=1).shape)
            logits = logits_net(torch.stack([tau_Y.real,tau_Y.imag,tau_Y.abs()],dim=1).reshape(num_trajectories,3,12,12))
            tau_h_hat = Normal(logits[:,::2], torch.exp(logits[:,1::2])).sample()
            
            ## compute loss and update
            optimizer.zero_grad()
            ## 2-norm of h - h_hat square at each realization (num_trajectories by 1) 
            norm = torch.norm(torch.view_as_complex((tau_h - tau_h_hat).reshape(num_trajectories, n_T*n_R, 2)), dim=1)**2
            ## lambda update
            lamb = max(0, lamb + lr/itr*(norm.mean() - t)) 
            ## model parameters update with policy gradient
            batch_loss = lamb*(norm.unsqueeze(1)*Normal(logits[:, ::2], torch.exp(logits[:, 1::2])).log_prob(tau_h_hat)).mean() ###
            batch_loss.backward()
            optimizer.step()
            ## t update
            t -= lr/itr*(1-lamb)
            ## save the loss to plot
            iter_loss_M[itr-1] = norm.mean() / (n_T*n_R)  ### nRnT
            iter_loss_N[itr-1] = (norm / torch.norm((tau_h).reshape(num_trajectories, n_T*n_R*2), dim=1)**2).mean()
            
            ## validation
            logits = logits_net(torch.stack([Y_test.real,Y_test.imag,Y_test.abs()],dim=1).reshape(test_size,3,12,12))
            h_hat_test = Normal(logits[:,::2], torch.exp(logits[:,1::2])).sample()
            testing_loss_M[itr-1] = (torch.norm((h_test - h_hat_test).reshape(test_size, n_T*n_R*2), dim=1)**2).mean() / (n_R*n_T)  ### nRnT
            testing_loss_N[itr-1] = torch.norm(h_test - h_hat_test)**2 / torch.norm(h_test)**2
            pbar.set_description('NMSE:%s' %(format(float(iter_loss_N[itr-1]), '.3f')))
            pbar.update(1)
        # after 1 epoch, shuffle the dataset, using the same index to shuffle H and Y
        idx = torch.randperm(Y.shape[0])
        Y = Y[idx,:,:]
        H = H[idx,:,:]

    ## plot the loss
    epochs = range(1, num_epochs+1)
    plt.plot(epochs, iter_loss_N.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='training loss')
    plt.plot(epochs, testing_loss_N.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation loss')
    plt.suptitle("MPP estimator with 3 256 CLs")   ###
    plt.title(' $[n_R,n_T,T]$:[%s,%s,%s], lr:%s, size:%s, $\\mu_H$:%s' 
                 %(n_R,n_T,T,lr, hidden_sizes, H_mean))  ###
    plt.xlabel('epochs')
    plt.ylabel('NMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig('./simulation/result/N_lr%s_%s_ep%s_mu%s.png' %(lr, hidden_sizes,num_epochs,H_mean))
    plt.close()

    plt.plot(epochs, iter_loss_M.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='training loss')
    plt.plot(epochs, testing_loss_M.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation loss')
    plt.suptitle("MPP estimator with 3 256 CLs")   ###
    plt.title(' $[n_R,n_T,T]$:[%s,%s,%s], lr:%s, size:%s, $\\mu_H$:%s' 
                 %(n_R,n_T,T,lr, hidden_sizes, H_mean))   ###
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig('./simulation/result/M_lr%s_%s_ep%s_mu%s_tau%s.png' %(lr, hidden_sizes,num_epochs,H_mean,num_trajectories))   ###

    ## save the model
    torch.save(logits_net, './simulation/result/lr%s_%s_ep%s_mu%s_tau%s.pt' %(lr, hidden_sizes,num_epochs,H_mean,num_trajectories))   ###

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hs', type=float, default=1) # sigma of H
    parser.add_argument('--hm', type=float, default=0) # mean of H
    parser.add_argument('--lr', type=float, default=1e-5) # learning rate
    parser.add_argument('--ep', type=int, default=10)    # num of epochs
    parser.add_argument('--tau', type=int, default=3000)  # 10k length of trajectory = size of mini-batch
    parser.add_argument('--nmb', type=int, default=100)  # 100 number of mini-batch
    parser.add_argument('--nR', type=int, default=4)  
    parser.add_argument('--nT', type=int, default=36)  
    parser.add_argument('--hsz', type=int, default=1, nargs='+')  # hidden layer size
    parser.add_argument('--cuda', type=int, default=0)  # cuda
    args = parser.parse_args()

    print("cuda:%s"%(args.cuda), torch.cuda.is_available())
    # print(args.hsz)

    if args.hsz == 1:
        hidden_sizes = [64, 32]
    else:
        hidden_sizes = args.hsz
    
    hidden_sizes = [1024,2048]

    T = args.nT*1
    # H_mean = 0
    W_mean, W_sigma = 0, 0.1
    channel_information = [args.hm, args.hs, W_mean, W_sigma]

    if args.tau < 10:
        tau = 10**args.tau
    else:
        tau = args.tau

    train([args.nT, args.nR, T], hidden_sizes, args.lr, channel_information, args.ep, int(tau), args.nmb, args.cuda)
