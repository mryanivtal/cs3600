import unittest

import os
import numpy as np
import matplotlib.pyplot as plt
import unittest
import torch
import torchvision
import torchvision.transforms as tvtf

import hw2.optimizers as optimizers
from hw2 import training


class MyTestCase(unittest.TestCase):
    seed = 42

    def setup(self):
        plt.rcParams.update({'font.size': 12})
        data_dir = os.path.expanduser('~/.pytorch-datasets')
        ds_train = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=True, transform=tvtf.ToTensor())
        ds_test = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=False, transform=tvtf.ToTensor())

        print(f'Train: {len(ds_train)} samples')
        print(f'Test: {len(ds_test)} samples')

        self.data_dir = data_dir
        self.ds_train = ds_train
        self.ds_test = ds_test


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


    def test_batch_train(self):
        # ------- pycarm block- -------
        self.setup()
        test = self
        seed = self.seed
        data_dir = self.data_dir
        ds_train = self.ds_train
        ds_test = self.ds_test
        # ----------------------------

        import hw2.layers as layers
        import hw2.answers as answers
        from torch.utils.data import DataLoader

        # Overfit to a very small dataset of 20 samples
        batch_size = 20
        max_batches = 2
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)

        # Get hyperparameters
        hp = answers.part2_overfit_hp()

        torch.manual_seed(seed)

        # Build a model and loss using our custom MLP and CE implementations
        model = layers.MLP(3 * 32 * 32, num_classes=10, hidden_features=[128] * 3, wstd=hp['wstd'])
        loss_fn = layers.CrossEntropyLoss()

        # Use our custom optimizer
        optimizer = optimizers.VanillaSGD(model.params(), learn_rate=hp['lr'], reg=hp['reg'])

        # Run training over small dataset multiple times
        trainer = training.LayerTrainer(model, loss_fn, optimizer)
        best_acc = 0
        for i in range(20):
            res = trainer.train_epoch(dl_train, max_batches=max_batches)
            best_acc = res.accuracy if res.accuracy > best_acc else best_acc

        test.assertGreaterEqual(best_acc, 98)


    def test_search_batch_train(self):
        # ------- pycarm block- -------
        self.setup()
        test = self
        seed = self.seed
        data_dir = self.data_dir
        ds_train = self.ds_train
        ds_test = self.ds_test
        # ----------------------------

        import hw2.layers as layers
        import hw2.answers as answers
        from torch.utils.data import DataLoader

        # Overfit to a very small dataset of 20 samples
        batch_size = 20
        max_batches = 2
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)

        # Get hyperparameters
        for wstd in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]:
            for lr in [0.0005, 0.001, 0.005, 0.008, 0.1, 0.15, 0.2]:

                hp = dict(wstd=wstd, lr=lr, reg=0)

                torch.manual_seed(seed)

                # Build a model and loss using our custom MLP and CE implementations
                model = layers.MLP(3 * 32 * 32, num_classes=10, hidden_features=[128] * 3, wstd=hp['wstd'])
                loss_fn = layers.CrossEntropyLoss()

                # Use our custom optimizer
                optimizer = optimizers.VanillaSGD(model.params(), learn_rate=hp['lr'], reg=hp['reg'])

                # Run training over small dataset multiple times
                trainer = training.LayerTrainer(model, loss_fn, optimizer)
                best_acc = 0
                for i in range(20):
                    res = trainer.train_epoch(dl_train, max_batches=max_batches)
                    best_acc = res.accuracy if res.accuracy > best_acc else best_acc

        test.assertGreaterEqual(best_acc, 98)



if __name__ == '__main__':
    unittest.main()
