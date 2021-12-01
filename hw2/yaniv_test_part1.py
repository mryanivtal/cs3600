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


    def test_lrelu(self):
        # ----------------- pycharm test block -------------
        test = self
        test_block_grad = self.test_block_grad
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
        test_block_grad(lrelu, x_test)
        # assert(True)

    def test_relu(self):
        # ----------------- pycharm test block -------------
        test = self
        test_block_grad = self.test_block_grad

        N = 100
        in_features = 200
        num_classes = 10
        eps = 1e-6
        # --------------------------------------------------

        # Test ReLU
        relu = layers.ReLU()
        x_test = torch.randn(N, in_features)

        # Test forward pass
        z = relu(x_test)
        test.assertSequenceEqual(z.shape, x_test.shape)
        test.assertTrue(torch.allclose(z, torch.relu(x_test), atol=eps))

        # Test backward pass
        test_block_grad(relu, x_test)


    def test_sigmoid(self):
        # ----------------- pycharm test block -------------
        test = self
        test_block_grad = self.test_block_grad

        N = 100
        in_features = 200
        num_classes = 10
        eps = 1e-6
        # --------------------------------------------------

        # Test Sigmoid
        sigmoid = layers.Sigmoid()
        x_test = torch.randn(N, in_features, in_features)  # 3D input should work

        # Test forward pass
        z = sigmoid(x_test)
        test.assertSequenceEqual(z.shape, x_test.shape)
        test.assertTrue(torch.allclose(z, torch.sigmoid(x_test), atol=eps))

        # Test backward pass
        test_block_grad(sigmoid, x_test)

    def test_tanh(self):
        # ----------------- pycharm test block -------------
        test = self
        test_block_grad = self.test_block_grad

        N = 100
        in_features = 200
        num_classes = 10
        eps = 1e-6
        # --------------------------------------------------

        # Test TanH
        tanh = layers.TanH()
        x_test = torch.randn(N, in_features, in_features)  # 3D input should work

        # Test forward pass
        z = tanh(x_test)
        test.assertSequenceEqual(z.shape, x_test.shape)
        test.assertTrue(torch.allclose(z, torch.tanh(x_test), atol=eps))

        # Test backward pass
        test_block_grad(tanh, x_test)

    def test_linear(self):
        # ----------------- pycharm test block -------------
        test = self
        test_block_grad = self.test_block_grad

        N = 100
        in_features = 200
        num_classes = 10
        eps = 1e-6
        # --------------------------------------------------

        # Test Linear
        out_features = 1000
        fc = layers.Linear(in_features, out_features)
        x_test = torch.randn(N, in_features)

        # Test forward pass
        z = fc(x_test)
        test.assertSequenceEqual(z.shape, [N, out_features])
        torch_fc = torch.nn.Linear(in_features, out_features, bias=True)
        torch_fc.weight = torch.nn.Parameter(fc.w)
        torch_fc.bias = torch.nn.Parameter(fc.b)
        test.assertTrue(torch.allclose(torch_fc(x_test), z, atol=eps))

        # Test backward pass
        test_block_grad(fc, x_test)

        # Test second backward pass
        x_test = torch.randn(N, in_features)
        z = fc(x_test)
        z = fc(x_test)
        test_block_grad(fc, x_test)


    def test_cross_entropy(self):
        # ----------------- pycharm test block -------------
        test = self
        test_block_grad = self.test_block_grad

        N = 100
        in_features = 200
        num_classes = 10
        eps = 1e-6
        # --------------------------------------------------

        # Test CrossEntropy
        cross_entropy = layers.CrossEntropyLoss()
        scores = torch.randn(N, num_classes)
        labels = torch.randint(low=0, high=num_classes, size=(N,), dtype=torch.long)

        # Test forward pass
        loss = cross_entropy(scores, labels)
        expected_loss = torch.nn.functional.cross_entropy(scores, labels)
        test.assertLess(torch.abs(expected_loss - loss).item(), 1e-5)
        print('loss=', loss.item())

        # Test backward pass
        test_block_grad(cross_entropy, scores, y=labels)


    def test_build_model(self):
        # ----------------- pycharm test block -------------
        test = self
        test_block_grad = self.test_block_grad

        N = 100
        in_features = 200
        num_classes = 10
        eps = 1e-6
        # --------------------------------------------------

        # Test Sequential
        # Let's create a long sequence of layers and see
        # whether we can compute end-to-end gradients of the whole thing.

        seq = layers.Sequential(
            layers.Linear(in_features, 100),
            layers.Linear(100, 200),
            layers.Linear(200, 100),
            layers.ReLU(),
            layers.Linear(100, 500),
            layers.LeakyReLU(alpha=0.01),
            layers.Linear(500, 200),
            layers.ReLU(),
            layers.Linear(200, 500),
            layers.LeakyReLU(alpha=0.1),
            layers.Linear(500, 1),
            layers.Sigmoid(),
        )
        x_test = torch.randn(N, in_features)

        # Test forward pass
        z = seq(x_test)
        test.assertSequenceEqual(z.shape, [N, 1])

        # Test backward pass
        test_block_grad(seq, x_test)

if __name__ == '__main__':
    unittest.main()
