import unittest
import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import hw1.knn_classifier as hw1knn

class MyTestCase(unittest.TestCase):
    def setup(self):
        plt.rcParams.update({'font.size': 12})
        torch.random.manual_seed(1904)
        test = self

        # Prepare data for kNN Classifier
        import torchvision.transforms as tvtf

        import cs3600.dataloader_utils as dataloader_utils
        import hw1.datasets as hw1datasets
        import hw1.transforms as hw1tf

        # Define the transforms that should be applied to each CIFAR-10 image before returning it
        tf_ds = tvtf.Compose([
            tvtf.ToTensor(),  # Convert PIL image to pytorch Tensor
            hw1tf.TensorView(-1),  # Reshape to 1D Tensor
        ])

        # Define how much data to load (only use a subset for speed)
        num_train = 10000
        num_test = 1000
        batch_size = 1024

        # Training dataset & loader
        data_root = os.path.expanduser('~/.pytorch-datasets')
        ds_train = hw1datasets.SubsetDataset(
            torchvision.datasets.MNIST(root=data_root, download=True, train=True, transform=tf_ds), num_train)
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size)

        # Test dataset & loader
        ds_test = hw1datasets.SubsetDataset(
            torchvision.datasets.MNIST(root=data_root, download=True, train=False, transform=tf_ds), num_test)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size)

        # Get all test data
        x_test, y_test = dataloader_utils.flatten(dl_test)
        self.assertEqual(True, True)


    def test_l2_dist(self):
        self.setup()
        test = self

        def l2_dist_naive(x1, x2):
            """
            Naive distance calculation, just for testing.
            Super slow, don't use!
            """
            dists = torch.empty(x1.shape[0], x2.shape[0], dtype=torch.float)
            for i, j in it.product(range(x1.shape[0]), range(x2.shape[0])):
                dists[i, j] = torch.sum((x1[i] - x2[j]) ** 2).item()
            return torch.sqrt(dists)

        # Test distance calculation
        x1 = torch.randn(12, 34)
        x2 = torch.randn(45, 34)

        dists = hw1knn.l2_dist(x1, x2)
        dists_naive = l2_dist_naive(x1, x2)

        test.assertTrue(torch.allclose(dists, dists_naive), msg="Wrong distances")


    def test_accuracy(self):
        self.setup()
        test = self

        y1 = torch.tensor([0, 1, 2, 3])
        y2 = torch.tensor([2, 2, 2, 2])
        test.assertEqual(hw1knn.accuracy(y1, y2), 0.25)


if __name__ == '__main__':
    unittest.main()
