# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: deep RRR
#     language: python
#     name: deep_rrr
# ---

# %% [markdown]
# # Running Whittington&Bogacz predictive-coding network on MNIST

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pydove as dv

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from tqdm.notebook import tqdm

from pcn import PCNetwork


# %% [markdown]
# ## Load dataset

# %%
def make_onehot(Y):
    y_oh = torch.FloatTensor(Y.shape[0], Y.max().item() + 1)
    y_oh.zero_()
    y_oh.scatter_(1, Y.reshape(-1, 1), 1)
    return y_oh


# %%
dataset = {}

trainset = torchvision.datasets.MNIST("data/", train=True, download=True)
testset = torchvision.datasets.MNIST("data/", train=False, download=True)

# split into train/validation/test and normalize
mu = 33.3184
sigma = 78.5675
dataset["train"] = [(trainset.data[:50000] - mu) / sigma, trainset.targets[:50000]]
dataset["validation"] = [(trainset.data[50000:] - mu) / sigma, trainset.targets[50000:]]

dataset["test"] = [(testset.data - mu) / sigma, testset.targets]

# make it one_hot
for key, item in dataset.items():
    dataset[key] = [item[0], make_onehot(item[1])]

# flatten images
for key, item in dataset.items():
    dataset[key] = [item[0].reshape(item[0].shape[0], -1), item[1]]

# center Y
# mean = dataset["train"][1].mean(0, keepdims=True)
# for key, item in dataset.items():
#     dataset[key] = [item[0], item[1] - mean]

# %%
torch.manual_seed(123)

batch_size = 32

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(*dataset["train"]),
    batch_size=batch_size,
    shuffle=True,
)

# %%
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(*dataset["test"]), batch_size=batch_size
)


# %%
def evaluate(net: PCNetwork, loader) -> tuple:
    """ Returns RMSE and prediction accuracy. """
    n = 0
    
    n_correct = 0
    n_all = 0
    
    sum_sq = 0.0
    for x, y in loader:
        y_pred = net.forward(x)
        sum_sq += torch.sum((y - y_pred) ** 2).item()
        n += len(y)
        
        # count correct predictions
        pred_digit = y_pred.argmax(dim=1)
        n_correct += y[range(len(y)), pred_digit].sum()
        n_all += len(y)
    
    return np.sqrt(sum_sq / n), n_correct / n_all


# %%
n_epochs = 50
test_every = 1

dims = [784, 600, 600, 10]
it_inference = 20
lr_inference = 0.1

torch.manual_seed(123)

net = PCNetwork(
    dims,
    activation=torch.sigmoid,
    lr_inference=lr_inference,
    it_inference=it_inference,
    variances=1.0,
)

# %%
rms_errors = np.zeros(n_epochs)
accuracies = np.zeros(n_epochs)
train_losses = np.zeros((n_epochs, len(train_loader)))

optimizer = torch.optim.Adam(net.slow_parameters(), lr=0.001)

for epoch in tqdm(range(n_epochs), desc="epoch"):
    # train
    for i, (x, y) in enumerate(tqdm(train_loader, desc="train")):
        net.forward_constrained(x, y)

        optimizer.zero_grad()
        loss = net.loss()
        loss.backward()
        optimizer.step()
        
        train_losses[epoch, i] = loss.item()
    
    # check performance
    rms_errors[epoch], accuracies[epoch] = evaluate(net, test_loader)

# %%
with dv.FigureManager() as (_, ax):
    ax.semilogy(train_losses.ravel(), lw=0.5)
    ax.set_xlabel("batch")
    ax.set_ylabel("PC loss")

# %%
with dv.FigureManager(1, 2) as (_, (ax1, ax2)):
    ax1.plot(rms_errors)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Test RMSE")
    
    ax2.plot(100 * accuracies)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy (%)")

# %%
