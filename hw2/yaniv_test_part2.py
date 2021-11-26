import unittest

import os
import numpy as np
import matplotlib.pyplot as plt
import unittest
import torch
import torchvision
import torchvision.transforms as tvtf

import hw2.optimizers as optimizers


class MyTestCase(unittest.TestCase):
    seed = 42

    def setup(self):
        plt.rcParams.update({'font.size': 12})


    def test_vanilla_sgd(self):
        test = self
        self.setup()

        # Test VanillaSGD
        torch.manual_seed(42)
        p = torch.randn(500, 10)
        dp = torch.randn(*p.shape) * 2
        params = [(p, dp)]

        vsgd = optimizers.VanillaSGD(params, learn_rate=0.5, reg=0.1)
        vsgd.step()

        expected_p = torch.load('tests/assets/expected_vsgd.pt')
        diff = torch.norm(p - expected_p).item()
        print(f'diff={diff}')
        test.assertLess(diff, 1e-3)


if __name__ == '__main__':
    unittest.main()
