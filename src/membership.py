"""

Class based Membership functions that uses the lower
level functions.

"""

__author__ = "Jay Morgan"

# external imports
import torch
import torch.nn as nn

# custom imports
from src.functional.membership import *


class GenericFunction(nn.Module):
    def __init__(self):
        super(GenericFunction, self).__init__()

    def _initialize(self, param):
        return nn.Parameter(torch.FloatTensor([param]))

    def forward(self, x):
        raise NotImplementedError


class Gaussian(GenericFunction):
    def __init__(self, mu, sigma):
        self.register_parameter("mu", self._initialize(mu))
        self.register_parameter("sigma", self._initialize(sigma))

    def forward(self, x):
        return gaussian(x, mu=self.mu, sigma=self.sigma)


class Trapezoid(GenericFunction):
    def __init__(self, a, b, c, d):
        super().__init__()
        self.register_parameter("a", self._initialize(a))
        self.register_parameter("b", self._initialize(b))
        self.register_parameter("c", self._initialize(c))
        self.register_parameter("d", self._initialize(d))

    def forward(self, x):
        return trapezoid(x, self.a, self.b, self.c, self.d)
