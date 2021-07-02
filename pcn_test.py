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
# # Implementing the predictive-coding network (PCN) from Whittington and Bogacz

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
# ## Solving XOR problem from PCN paper

# %%
data = {
    "x": torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]]),
    "y": torch.Tensor([1, 0, 0, 1]),
}


# %%
def test(net: PCNetwork, data: dict) -> float:
    n = 0
    sum_sq = 0.0
    for x, y in zip(data["x"], data["y"]):
        y_pred = net.forward(x)
        sum_sq += ((y - y_pred) ** 2).item()
        n += 1
    
    return np.sqrt(sum_sq / n)


# %%
net = PCNetwork(
    [2, 5, 1],
    activation="tanh",
    lr=0.2,
    lr_inference=0.2,
    it_inference=100,
    weight_decay=0.0,
    variances=[1.0, 10.0],
)

# %%
n_epochs = 500
n_runs = 4

test_every = 50

rms_errors = np.zeros((n_runs, n_epochs // test_every))
for run in tqdm(range(n_runs), desc="run"):
    net.reset()
    
    for epoch in tqdm(range(n_epochs), desc="epoch"):
        # check performance
        if epoch % test_every == 0:
            rms_errors[run, epoch // test_every] = test(net, data)
        
        # train
        for x, y in zip(data["x"], data["y"]):
            net.learn(x, y)

# %%
with dv.FigureManager() as (_, ax):
    for run in range(n_runs):
        ax.plot(np.arange(0, n_epochs, test_every), rms_errors[run], label=f"run {run + 1}")
        
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.legend()    


# %% [markdown]
# ## SCRATCH: MNIST dataset

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

# Splitting into train/validation/test and normalizing
mu = 33.3184
sigma = 78.5675
dataset["train"] = [(trainset.data[:50000] - mu) / sigma, trainset.targets[:50000]]
dataset["validation"] = [(trainset.data[50000:] - mu) / sigma, trainset.targets[50000:]]

dataset["test"] = [(testset.data - mu) / sigma, testset.targets]

# Making it one_hot
for key, item in dataset.items():
    dataset[key] = [item[0], make_onehot(item[1])]

# Flattening
for key, item in dataset.items():
    dataset[key] = [item[0].reshape(item[0].shape[0], -1), item[1]]

# Centering Y
# mean = dataset["train"][1].mean(0, keepdims=True)
# for key, item in dataset.items():
#     dataset[key] = [item[0], item[1] - mean]

# %%
torch.manual_seed(123)

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
n_correct = 0
n_all = 0
for data, target in tqdm(test_loader):
    # run the inference step
    zeta = data @ W1 + b1
    for i in range(num_zsteps):
        eps_zeta = zeta - data @ W1 - b1
        z = f(zeta)
        y = z @ W2 + b2
        
        eps_y = y - z @ W2 - b2
        zeta += z_lr * (-(1 / alpha) * eps_zeta + fder(zeta) * (eps_y @ W2.T))

    # predict output
    z = f(zeta)
    y = z @ W2 + b2

    # count correct predictions
    pred = y.argmax(dim=1)
    n_correct += target[range(len(target)), pred].sum()
    n_all += len(target)

print(f"Prediction accuracy: {100 * n_correct / n_all:.2f}%.")
