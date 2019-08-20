
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torchcontrib
import pickle
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn
import numpy as np
import pandas as pd
import torch.nn.functional as F

LR = 0.005
BATCH_SIZE = 100
EPOCH = 500
N_FEATURES = 7
N_HIDDEN = 3
elu = torch.nn.ELU()
relu = torch.nn.RELU()
sig = torch.nn.Sigmoid()
dropout = torch.nn.Dropout(0.3)
x = pd.read_pickle("trainx.pkl")
y = pd.read_pickle("trainy.pkl")
xnp = x.to_numpy()
xnp = xnp.astype(np.float64)
ynp = y.to_numpy()
ynp = ynp.astype(np.float64)
## The taget here is the parameters u
## The inputs are the Y
inputs = torch.from_numpy(ynp)
targets = torch.from_numpy(xnp)


## Since there's no absolute difference in pytorch
## we have to define one ourself

class madloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.abs((x - y)))


def mad_loss(x, y):
    return torch.mean(torch.abs((x - y)))



class LinReg(nn.Module):
    def __init__(self, n_input, n_hidden, n_output,loss_fun):
        super(LinReg, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self._loss=loss_fun

        linear1 = torch.nn.Linear(n_input, n_hidden)
        torch.nn.init.xavier_uniform(linear1.weight)

        linear2 = torch.nn.Linear(n_hidden, 7)
        torch.nn.init.xavier_uniform(linear2.weight)

        self.classifier = torch.nn.Sequential(
            linear1, dropout, self._loss,
            linear2,
        )

    def forward(self, x):      
        varSize = x.data.shape[0]
        x = x.contiguous()
        x = self.classifier(x)
        return x




net_SGD = LinReg(N_FEATURES, N_HIDDEN, 7,elu)
net_Momentum = LinReg(N_FEATURES, N_HIDDEN, 7,elu)
net_RMSprop = LinReg(N_FEATURES, N_HIDDEN, 7,elu)
net_Adam = LinReg(N_FEATURES, N_HIDDEN, 7,elu)
netSWA = LinReg(N_FEATURES, N_HIDDEN, 7,elu)


net_overfitting = torch.nn.Sequential(
    torch.nn.Linear(N_FEATURES, N_HIDDEN),
    torch.nn.ELU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.ELU(),
    torch.nn.Linear(N_HIDDEN, 30),
    torch.nn.ELU(),
    torch.nn.Linear(30, 7)
)

net_dropped = torch.nn.Sequential(
    torch.nn.Linear(N_FEATURES, N_HIDDEN),
    nn.BatchNorm1d(N_HIDDEN),
    torch.nn.Dropout(0.3),
    torch.nn.ELU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.Dropout(0.3),
    torch.nn.ELU(),
    torch.nn.Linear(N_HIDDEN, 30),
    torch.nn.Dropout(0.3),
    torch.nn.ELU(),
    torch.nn.Linear(30, 7),
)

dataset = Data.TensorDataset(inputs.float(), targets.float())
loader = Data.DataLoader(
    dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.9)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
base_opt = torch.optim.SGD(netSWA.parameters(), lr=0.1)
SWAopt = torchcontrib.optim.SWA(
    base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)


loss_func = madloss()


loss_SGD = []
loss_Momentum = []
loss_RMSprop = []
loss_Adam = []
loss_SWA = []

losses = [loss_SGD, loss_Momentum, loss_RMSprop, loss_Adam, loss_SWA]
#losses = [loss_SGD, loss_Adam,loss_RMSprop]
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam, netSWA]
#nets = [net_SGD, net_Adam,net_RMSprop]
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam, SWAopt]
#optimizers = [opt_SGD, opt_Adam,opt_RMSprop]



## begin training
for epoch in range(0, EPOCH + 1):
    print('Training Epoch= {}/{} '.format(epoch, EPOCH))
    for step, (batch_x, batch_y) in enumerate(loader):
        var_x = Variable(batch_x)
        var_y = Variable(batch_y)
        for net, optimizer, loss_history in zip(nets, optimizers, losses):
            #print ('Model:' + type(net).__name__)
            #print ('Opt:' + type(optimizer).__name__)
            prediction = net(var_x)
            loss = mad_loss(prediction, var_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.data.item())

#     if epoch % 5  == 0:
    loss_run = loss.data.item()
    print(step, loss_run)
    print('Training MSELoss=%.4f' % loss_run)


## load the test set
x = pd.read_pickle("testx.pkl")
y = pd.read_pickle("testy.pkl")
xnp = x.to_numpy()
xnp = xnp.astype(np.float64)
ynp = y.to_numpy()
ynp = ynp.astype(np.float64)
## The taget here is the parameters u
## The inputs are the Y
x_test = torch.from_numpy(ynp)
y_test = torch.from_numpy(xnp)


## plot learners learning rate
labels = ['SGD', 'Momentum', 'RMSprop', 'Adam', 'SWA']
f, ax = plt.subplots(figsize=(12, 9))
for i, loss_history in enumerate(losses):
    plt.plot(loss_history, label=labels[i])

plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()


## make prediction using test set
plist = []
for net in nets:
  pred = net(x_test.float())
  plist.append(pred.detach().numpy())
  ls = mad_loss(pred, y_test.float())
