# Predictive-coding network

[![Python 3.8](https://img.shields.io/badge/python-3.8-yellow.svg)](https://www.python.org/downloads/release/python-360/)
[![PyTorch 1.8.1](https://img.shields.io/badge/pytorch-1.8.1-yellow.svg)](https://pytorch.org/)

This is a PyTorch implementation of the Whittington & Bogacz supervised [Predictive Coding Network](https://www.mitpressjournals.org/doi/pdf/10.1162/NECO_a_00949).

The implementation was tested using Python 3.8.2 and PyTorch 1.8.1, but it would probably work with older versions, as well.

## Setup

After downloading the code, `cd` to the directory and install pre-requisites (preferably in a fresh virtual environment) using

    pip install -r requirements.txt

Install the code as a package using

    pip install .

in the same folder. For development, create an editable install using

    pip install -e .

Test that everything works:

    pytest test/

## Getting started

The basic class is `PCNetwork`, defined in `pcn.py`, *e.g.,*

    from pcn import PCNetwork
    net = PCNetwork(
        [784, 600, 600, 10],
        activation=torch.sigmoid,
        lr_inference=0.1,
        it_inference=20,
        variances=1.0,
    )

Training is slightly different from a usual `nn.Module` because of the non-trivial inference stage:

    optimizer = torch.optim.Adam(net.slow_parameters(), lr=0.001)

    for epoch in range(10):
        for x, y in train_loader:
            net.forward_constrained(x, y)

            optimizer.zero_grad()
            loss = net.loss()
            loss.backward()
            optimizer.step()

where `train_loader` can be any iterable that returns pairs of (batches of) input and output samples. `forward_constrained` currently uses an `SGD` optimizer internally.

Complete examples can be found in `pcn_on_mnist.py` and `pcn_on_xor.py`. These are actually Jupyter notebooks converted to Python scripts using [`jupytext`](https://github.com/mwouts/jupytext). (`jupytext` should already be installed if you ran `pip install -r requirements.txt`.)

To create actual `ipynb` notebook files, you can run

    jupytext --sync pcn_on_mnist.py
    jupytext --sync pcn_on_xor.py

Alternatively, follow the instructions [here](https://github.com/mwouts/jupytext) to open the scripts as notebooks in [Jupyter](https://jupyter.org/) or [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/).

Another option is to open the files in VSCode, which will automatically recognize the cells and allow you to run them in a Jupyter instance.
