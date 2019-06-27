"""

Membership functions from a crisp to linguistic variable.

"""
__author__ = "Jay Morgan"


# external imports
import torch

# custom imports
from src.functional.utils import _tt 


def triangle(x, a, b, c):
    """Triangular Membership Function
    
    :param x: input value
    :param a: start point where membership is 0
    :param b: center point where membership is 1
    :param c: end point where membership is 0
    """
    assert a < b < c
    a, b, c = _tt(a), _tt(b), _tt(c)
    x1 = (x-a) / (b-a)
    x2 = (c-x) / (c-b)
    membership = torch.min(x1, x2)
    return torch.max(membership, _tt(0))


def trapezoid(x, a, b, c, d):
    """Trapezoidal Membership Function

    :param x: input value
    :param a: bottom left point where membership is 0
    :param b: top left point where membership is 1
    :param c: top right point where membership is 1
    :param d: bottom right point where membership is 0
    """
    a, b, c, d = _tt(a), _tt(b), _tt(c), _tt(d)
    x1 = (x-a) / (b-a)
    x2 = (d-x) / (d-c)
    membership = torch.min(torch.min(x1, _tt(1)), x2)
    return torch.max(membership, _tt(0))


def gaussian(x, a, b):
    """Gaussian Membership Function

    :param x: input value
    :param a: The mean of the Gaussian Distribution
    :param b: The standard deviation of the Distribution

    Usage: gaussian(40, a=50, b=20)
           gaussian(torch.Tensor([[20],[30]]), a=50, b=20)
    """
    a, b = _tt(a), _tt(b)
    return torch.exp(-(1/2 * (((x-a)/b)**2)))


def bell(x, a, b, c):
    """General Bell Curve Membership Function

    :param x: input value
    :param a: width of bell curve.
    :param b: slop of the curve, lower values = curvier
    :param c: centre of the curve.
    """
    a, b, c = _tt(a), _tt(b), _tt(c)
    return 1 / (1 + (torch.abs((x-c) / a) ** (2*b)))


def sigmoid(x, a, b):
    """Sigmoidal Membership Function

    :param x: input value
    :param a: amount of curvature, higher values = unit step
    :param b: 0.5 centre posistion
    """
    return 1 / (1 + torch.exp(-(a * (x-b))))


def lr(x, a, b, c):
    """ Left-Right (LR) Membership Function

    :param x: input value
    :param a: centre point of change
    :param b: rate of decay after change
    :param c: length of decay
    """
    _x = x.clone()
    _x[x <= a] = fl((a-x[x <= a]) / b)
    _x[x >= a] = fr((x[x >= a]-a) / c)
    return _x


# monotonically decreasing functions
def fl(x): return torch.max(_tt(0), torch.sqrt(1-(x**2)));
def fr(x): return torch.exp(-(torch.abs(x)**3));