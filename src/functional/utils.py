"""

Simple utility functions

"""
__author__ = "Jay Morgan"

import torch


def _to_tensor(el):
    """ Ensure parameters are rank-1 single element tensors

    :param el: Element to convert to rank-1 tensor.
    :return: el as a torch.Tensor type.
    """
    if isinstance(el, torch.Tensor):
        return el
    else:
        return torch.autograd.Variable(torch.FloatTensor([el]), requires_grad=True)


# alias for functions
_tt = _to_tensor