import torch
import unittest
import hw2.layers as layers
from hw2.grad_compare import compare_layer_to_torch



class MyTestCase(unittest.TestCase):
    def setup(self):
        pass


    def test_block_grad(self, block: layers.Layer, x, y=None, delta=1e-3):
        # ----------------- pycharm test block -------------
        test = self
        # --------------------------------------------------

        diffs = compare_layer_to_torch(block, x, y)
        # Assert diff values
        for diff in diffs:
            test.assertLess(diff, delta)


    def test_layer(self):
        # ----------------- pycharm test block -------------
        test = self
        # --------------------------------------------------

        N = 100
        in_features = 200
        num_classes = 10
        eps = 1e-6

        # Test LeakyReLU
        alpha = 0.1
        lrelu = layers.LeakyReLU(alpha=alpha)
        x_test = torch.randn(N, in_features)

        # Test forward pass
        z = lrelu(x_test)
        test.assertSequenceEqual(z.shape, x_test.shape)
        test.assertTrue(torch.allclose(z, torch.nn.LeakyReLU(alpha)(x_test), atol=eps))

        # Test backward pass
        self.test_block_grad(lrelu, x_test)
        # assert(True)





if __name__ == '__main__':
    unittest.main()
