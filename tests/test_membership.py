# internal imports
import unittest

# external imports
import torch

# custom imports
from src.functional.membership import *


class TestMembership(unittest.TestCase):
    def setUp(self):
        self.x = torch.linspace(0, 100, 100)
        return super().setUp()

    def test_triangle(self):
        out = triangle(self.x, 20, 60, 80)
    
    def test_trapezoid(self):
        out = trapezoid(self.x, 10, 20, 60, 95)

    def test_gaussian(self):
        out = gaussian(x, 50, 20)

    def test_bell(self):
        out = bell(self.x, 20, 4, 50)

    def test_sigmoid(self):
        out = sigmoid(self.x, 0.5, 50)

    def test_lr(self):
        out = lr(self.x, 65, 60, 10)
        out = lr(self.x, 25, 10, 40)



if __name__ == "__main__":
    unittest.main()