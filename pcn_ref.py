""" A reference implementation of the predictive-coding network. """

from typing import Sequence, Union

import torch
import torch.nn as nn

import numpy as np


class PCNetworkRef(object):
    """ A reference implementation of the predictive coding network from
    Whittington&Bogacz.

    This focuses on the `tanh` nonlinearity, has no weight decay, and sets all variances
    to 1.
    """

    def __init__(
        self,
        dims: Sequence,
        it_inference: int = 100,
        lr: float = 0.2,
        lr_inference: float = 0.2,
    ):
        """ Initialize the network.

        :param dims: number of units in each layer
        :param it_inference: number of iterations per inference step
        :param lr: learning rate for weights
        :param lr_inference: learning rate for inference step
        """
        self.dims = np.copy(dims)

        self.it_inference = it_inference
        self.lr = lr
        self.lr_inference = lr_inference

        # create fields that will be populated later
        self.W = []
        self.b = []
        self.x = []
        self.fp = []
        self.eps = []

    def reset(self):
        """ Initialize all tensors. """
        W = []
        b = []
        x = []
        fp = []
        eps = []
        for i in range(len(self.dims)):
            if i + 1 < len(self.dims):
                # weights
                W.append(torch.Tensor(self.dims[i + 1], self.dims[i]))
                nn.init.xavier_uniform_(W[-1])

                # biases
                # noinspection PyArgumentList
                b.append(torch.zeros(self.dims[i + 1]))

                # error nodes
                # no error nodes for the input sample!
                eps.append(torch.zeros(self.dims[i + 1]))

                # derivative of activation functions applied to x
                fp.append(torch.zeros(self.dims[i + 1]))

            # activation vectors
            x.append(torch.zeros(self.dims[i]))

        self.W = W
        self.b = b
        self.x = x
        self.fp = fp
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Do a forward pass through the non-error nodes.

        This works in pure inference mode.

        :param x: input sample
        :returns: activation at output (last) layer
        """
        self.x[0] = x
        for i in range(len(self.dims) - 1):
            x = self.W[i] @ torch.tanh(x) + self.b[i]

            self.x[i + 1] = x

        return x

    def calculate_errors(self):
        """ Update error nodes.

        This assumes that the input and output sample values were set.

        This also updates the derivatives of the activation functions.
        """
        x = self.x[0]
        for i in range(len(self.dims) - 1):
            x_pred = self.W[i] @ torch.tanh(x) + self.b[i]

            x = self.x[i + 1]
            self.eps[i] = x - x_pred

            self.fp[i] = 1 / torch.cosh(x) ** 2

    def update_variables(self):
        """ Update variable nodes.

        This assumes that the input and output sample values were set in
        `self.x`, and that `calculate_errors` was run in order to calculate
        `self.eps` and `self.fp`.
        """
        for i in range(1, len(self.dims) - 1):
            g = (self.W[i].T @ self.eps[i]) * self.fp[i - 1]
            self.x[i] += self.lr_inference * (g - self.eps[i - 1])

    def update_weights(self):
        """ Update weights and biases. """
        for i in range(len(self.dims) - 1):
            grad_W = torch.outer(self.eps[i], torch.tanh(self.x[i]))
            grad_b = self.eps[i]

            self.W[i] += self.lr * grad_W
            self.b[i] += self.lr * grad_b

    def infer(self, y: torch.Tensor):
        """ Perform inference in supervised mode.

        This assumes that the input sample value was set in `self.x`.

        :param y: output sample
        """
        self.x[-1] = y
        for i in range(self.it_inference):
            self.calculate_errors()
            self.update_variables()

        # ensure the errors are up-to-date
        self.calculate_errors()

    def learn(self, x: torch.Tensor, y: torch.Tensor):
        """ Perform inference and weight updating in supervised mode.

        :param x: input sample
        :param y: output sample
        """
        self.forward(x)
        self.infer(y)
        self.update_weights()
