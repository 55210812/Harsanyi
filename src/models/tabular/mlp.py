import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["mlp"]

class MLP(nn.Module):

    def __init__(self,
                 n_layer,
                 in_dim,
                 out_dim,
                 width=1024,
                 bias=True,
                 activation="relu",
                 beta=1):
        super(MLP, self).__init__()
        assert activation in ["relu", "softplus", "sigmoid", "tanh"]

        self.n_layer = n_layer
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.bias = bias
        self.activation = activation
        self.beta = beta # only useful when activation is 'softplus'

        if n_layer < 2:
            raise Exception(f"Invalid #layer: {n_layer}.")

        self.all_layers = self._make_layers(in_dim, width, out_dim, n_layer)

    def _get_activation(self):
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "softplus":
            return nn.Softplus(beta=self.beta)
        elif self.activation == "sigmoid":
            return nn.Sigmoid()
        elif self.activation == "tanh":
            return nn.Tanh()
        else:
            raise NotImplementedError("activation function not implemented")

    def _make_layers(self, in_dim, hidd_dim, out_dim, n_layer):
        layers = [nn.Linear(in_dim, hidd_dim, bias=self.bias)]
        layers.append(self._get_activation())

        for _ in range(n_layer - 2):
            layers.append(nn.Linear(hidd_dim, hidd_dim, bias=self.bias))
            layers.append(self._get_activation())

        layers.append(nn.Linear(hidd_dim, out_dim, bias=self.bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.all_layers(x)


# wrapper function for mlp
def mlp(n_layer, in_dim, out_dim, width, bias=True, activation="relu", beta=1):
    return MLP(n_layer, in_dim, out_dim, width, bias, activation, beta)
