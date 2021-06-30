""" Implement the predictive-coding network from Whittington & Bogacz. """

from typing import Sequence, Union

import torch
import torch.nn as nn

import numpy as np


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


class PCNetwork(object):
    """ An implementation of the predictive coding network from Whittington&Bogacz. """

    def __init__(
        self,
        dims: Sequence,
        activation: Union[Sequence, str] = "tanh",
        it_inference: int = 100,
        lr: float = 0.2,
        lr_inference: float = 0.2,
        weight_decay: float = 0.0,
        variances: Union[Sequence, float] = 1.0,
    ):
        """ Initialize the network.

        :param dims: number of units in each layer
        :param activation: activation function(s) to use for each layer
            Can be "tanh", "relu", "sigmoid".
        :param it_inference: number of iterations per inference step
        :param lr: learning rate for weights
        :param lr_inference: learning rate for inference step
        :param weight_decay: weight decay parameter
        :param variances: variance(s) to use for each layer after the first
        """
        self.dims = np.copy(dims)
        self.activation = (
            (len(self.dims) - 1) * [activation]
            if isinstance(activation, str)
            else list(activation)
        )

        assert len(self.activation) == len(self.dims) - 1

        self.it_inference = it_inference
        self.lr = lr
        self.lr_inference = lr_inference
        self.weight_decay = weight_decay
        self.variances = torch.from_numpy(
            np.copy(variances)
            if np.size(variances) > 1
            else np.repeat(variances, len(self.dims) - 1)
        )

        assert len(self.variances) == len(self.dims) - 1

        self.activation_map = {
            "linear": identity,
            "relu": torch.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        self.der_activation_map = {
            "linear": lambda x: torch.ones_like(x),
            "relu": lambda x: torch.heaviside(x, torch.tensor([0.5])),
            "tanh": lambda x: 1 / torch.cosh(x) ** 2,
            "sigmoid": lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x)),
        }

        # gain factors for Xavier initialization
        self.gain_map = {
            "tanh": 1,
            "relu": 1 / np.sqrt(6),
            "sigmoid": 4,
            "linear": 1 / np.sqrt(6),
        }
        self.maxbias_map = {"tanh": 0, "relu": 0.1, "sigmoid": 0, "linear": 0}

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
                nn.init.xavier_uniform_(W[-1], gain=self.gain_map[self.activation[i]])

                # biases
                # noinspection PyArgumentList
                b.append(
                    torch.Tensor(self.dims[i + 1]).uniform_(
                        0, self.maxbias_map[self.activation[i]]
                    )
                )

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
            f = self.activation_map[self.activation[i]]
            x = self.W[i] @ f(x) + self.b[i]

            self.x[i + 1] = x

        return x

    def calculate_errors(self):
        """ Update error nodes.

        This assumes that the input and output sample values were set.

        This also updates the derivatives of the activation functions.
        """
        x = self.x[0]
        for i in range(len(self.dims) - 1):
            f = self.activation_map[self.activation[i]]
            fp = self.der_activation_map[self.activation[i]]

            x_pred = self.W[i] @ f(x) + self.b[i]

            x = self.x[i + 1]
            self.eps[i] = (x - x_pred) / self.variances[i]

            fp_value = fp(x)
            self.fp[i] = fp_value

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
        v_out = self.variances[-1]
        for i in range(len(self.dims) - 1):
            f = self.activation_map[self.activation[i]]
            grad_W = (
                v_out * torch.outer(self.eps[i], f(self.x[i]))
                - self.weight_decay * self.W[i]
            )
            grad_b = v_out * self.eps[i]

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
