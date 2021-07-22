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
from pcn_ref import PCNetworkRef

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
n_epochs = 500
n_runs = 4

test_every = 50

rms_errors = np.zeros((n_runs, n_epochs // test_every))
for run in tqdm(range(n_runs), desc="run"):
    net = PCNetwork(
        [2, 5, 1],
        activation=torch.tanh,
        lr_inference=0.2,
        it_inference=100,
        variances=[1.0, 10.0],
    )
        
    # Whittington&Bogacz's code inexplicably multiplies the learning rate by the
    # output-layer variance... the reference implementation, PCNetworkRef, does this
    # PCNetwork does not, so we have to put it in by hand
    optimizer = torch.optim.SGD(net.slow_parameters(), lr=0.2 * 10, weight_decay=0)
    
    for epoch in tqdm(range(n_epochs), desc="epoch"):
        # check performance
        if epoch % test_every == 0:
            rms_errors[run, epoch // test_every] = test(net, data)
        
        # train
        for x, y in zip(data["x"], data["y"]):
            net.forward_constrained(x, y)
            
            optimizer.zero_grad()
            net.loss().backward()
            optimizer.step()

# %%
with dv.FigureManager() as (_, ax):
    for run in range(n_runs):
        ax.plot(np.arange(0, n_epochs, test_every), rms_errors[run], label=f"run {run + 1}")
        
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.legend()

# %%
net_ref = PCNetworkRef(
    [2, 5, 1],
    lr=0.2,
    activation="tanh",
    lr_inference=0.2,
    it_inference=100,
    variances=[1.0, 10.0],
)

rms_errors_ref = np.zeros((n_runs, n_epochs // test_every))
for run in tqdm(range(n_runs), desc="run"):
    net_ref.reset()
    for epoch in tqdm(range(n_epochs), desc="epoch"):
        # check performance
        if epoch % test_every == 0:
            rms_errors_ref[run, epoch // test_every] = test(net_ref, data)
        
        # train
        for x, y in zip(data["x"], data["y"]):
            net_ref.learn(x, y)

# %%
with dv.FigureManager() as (_, ax):
    for run in range(n_runs):
        ax.plot(np.arange(0, n_epochs, test_every), rms_errors_ref[run], label=f"run {run + 1}")
        
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.legend()
