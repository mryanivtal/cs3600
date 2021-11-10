import unittest
import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import hw1.transforms as hw1tf



class MyTestCase(unittest.TestCase):
    def setup(self):
        plt.rcParams.update({'font.size': 12})
        torch.random.manual_seed(1904)
        test = unittest.TestCase()
        #----------------------------------------------


    def test_bias_trick(self):
        test = self

        tf_btrick = hw1tf.BiasTrick()

        test_cases = [
            torch.randn(64, 512),
            torch.randn(2, 3, 4, 5, 6, 7),
            torch.randint(low=0, high=10, size=(1, 12)),
            torch.tensor([10, 11, 12])
        ]

        for x_test in test_cases:
            xb = tf_btrick(x_test)
            print('shape =', xb.shape)
            test.assertEqual(x_test.dtype, xb.dtype, "Wrong dtype")
            test.assertTrue(torch.all(xb[..., 1:] == x_test), "Original features destroyed")
            test.assertTrue(torch.all(xb[..., [0]] == torch.ones(*xb.shape[:-1], 1)), "First feature is not equal to 1")


if __name__ == '__main__':
    unittest.main()
