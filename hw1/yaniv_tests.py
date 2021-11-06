import unittest

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import unittest

import hw1.datasets as hw1datasets
import cs3600.plot as plot

import os
import torchvision
import torchvision.transforms as tvtf




class MyTestCase(unittest.TestCase):

    def setup(self):
        plt.rcParams.update({'font.size': 12})
        torch.random.manual_seed(42)
        test = unittest.TestCase()

        self.cfar10_labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        data_root = os.path.expanduser('~/.pytorch-datasets')

        self.cifar10_train_ds = torchvision.datasets.CIFAR10(
            root=data_root, download=True, train=True,
            transform=tvtf.ToTensor()
        )

        print('Number of samples:', len(self.cifar10_train_ds))


    def test_setup(self):
        self.setup()
        cifar10_train_ds = self.cifar10_train_ds

        # Create a simple DataLoader that partitions the data into batches
        # of size N=8 in random order, using two background proceses

        cifar10_train_dl = torch.utils.data.DataLoader(
            cifar10_train_ds, batch_size=8, shuffle=True, num_workers=2
        )

        # Iterate over batches sampled with our DataLoader
        num_batches_to_show = 5
        for idx, (images, classes) in enumerate(cifar10_train_dl):
            # The DataLoader returns a tuple of:
            # images: Tensor of size NxCxWxH
            # classes: Tensor of size N
            fig, axes = plot.tensors_as_images(images, figsize=(8, 1))
            fig.suptitle(f'Batch #{idx + 1}:', x=0, y=0.6)
            if idx >= num_batches_to_show - 1:
                break

        self.assertEqual(True, True)  # add assertion here


    def test_sampler(self):
        self.setup()
        cifar10_train_ds = self.cifar10_train_ds

        import hw1.dataloaders as hw1dataloaders

        # Test sampler with odd number of elements
        sampler = hw1dataloaders.FirstLastSampler(list(range(5)))
        self.assertEqual(list(sampler), [0, 4, 1, 3, 2, ])
        print('------')  # DEBUG

        # Test sampler with evennumber of elements
        sampler = hw1dataloaders.FirstLastSampler(list(range(6)))
        self.assertEqual(list(sampler), [0, 5, 1, 4, 2, 3])

        print('------')  # DEBUG

        # Create a DataLoader that partitions the data into batches
        # of size N=2 in an order determined by our custom sampler
        cifar10_train_dl = torch.utils.data.DataLoader(
            cifar10_train_ds, batch_size=2, num_workers=0,
            sampler=hw1dataloaders.FirstLastSampler(cifar10_train_ds),
        )

        # Iterate over batches sampled with our DataLoader
        num_batches_to_show = 3
        for idx, (images, classes) in enumerate(cifar10_train_dl):
            fig, axes = plot.tensors_as_images(images, figsize=(8, 1))
            fig.suptitle(f'Batch #{idx + 1}:', x=0, y=0.6)
            plt.show()
            if idx >= num_batches_to_show - 1:
                break


if __name__ == '__main__':
    unittest.main()
