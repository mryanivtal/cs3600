import pickle
import unittest

import os
import numpy as np
import matplotlib.pyplot as plt
import unittest

import pandas as pd
import torch
import torchvision
import torchvision.transforms as tvtf

import hw2.optimizers as optimizers
from hw2 import training
import hw2.layers as layers
import hw2.answers as answers
from torch.utils.data import DataLoader

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
        batch_size = 10
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
        batch_size = 10
        max_batches = 2
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)

        # Get hyperparameters
        for wstd in [0.01, 0.05, 0.1]:
            for lr in [0.001, 0.005, 0.01, 0.05, 0.1]:
                for reg in [0.00, 0.001, 0.01, 0.1]:
                    print(wstd, lr, reg)

                    hp = dict(wstd=wstd, lr=lr, reg=reg)

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


    def test_fit(self):
        # ------- pycarm block- -------
        self.setup()
        test = self
        seed = self.seed
        data_dir = self.data_dir
        ds_train = self.ds_train
        ds_test = self.ds_test
        from cs3600.plot import plot_fit
        # ----------------------------

        # Define a larger part of the CIFAR-10 dataset (still not the whole thing)
        batch_size = 50
        max_batches = 100
        in_features = 3 * 32 * 32
        num_classes = 10
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size // 2, shuffle=False)

        # Define a function to train a model with our Trainer and various optimizers
        def train_with_optimizer(opt_name, opt_class, fig):
            torch.manual_seed(seed)

            # Get hyperparameters
            hp = answers.part2_optim_hp()
            hidden_features = [128] * 5
            num_epochs = 10

            # Create model, loss and optimizer instances
            model = layers.MLP(in_features, num_classes, hidden_features, wstd=hp['wstd'])
            loss_fn = layers.CrossEntropyLoss()
            optimizer = opt_class(model.params(), learn_rate=hp[f'lr_{opt_name}'], reg=hp['reg'])

            # Train with the Trainer
            trainer = training.LayerTrainer(model, loss_fn, optimizer)
            fit_res = trainer.fit(dl_train, dl_test, num_epochs, max_batches=max_batches)

            fig, axes = plot_fit(fit_res, fig=fig, legend=opt_name)
            return fig


        fig_optim = None
        fig_optim = train_with_optimizer('vanilla', optimizers.VanillaSGD, fig_optim)


    def test_momentum(self):
        # ------- pycarm block- -------
        self.setup()
        test = self
        seed = self.seed
        data_dir = self.data_dir
        ds_train = self.ds_train
        ds_test = self.ds_test
        from cs3600.plot import plot_fit
        # ----------------------------

        # Define a larger part of the CIFAR-10 dataset (still not the whole thing)
        batch_size = 50
        max_batches = 100
        in_features = 3 * 32 * 32
        num_classes = 10
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size // 2, shuffle=False)

        # Define a function to train a model with our Trainer and various optimizers
        def train_with_optimizer(opt_name, opt_class, fig):
            torch.manual_seed(seed)

            # Get hyperparameters
            hp = answers.part2_optim_hp()
            hidden_features = [128] * 5
            num_epochs = 10

            # Create model, loss and optimizer instances
            model = layers.MLP(in_features, num_classes, hidden_features, wstd=hp['wstd'])
            loss_fn = layers.CrossEntropyLoss()
            optimizer = opt_class(model.params(), learn_rate=hp[f'lr_{opt_name}'], reg=hp['reg'])

            # Train with the Trainer
            trainer = training.LayerTrainer(model, loss_fn, optimizer)
            fit_res = trainer.fit(dl_train, dl_test, num_epochs, max_batches=max_batches)

            fig, axes = plot_fit(fit_res, fig=fig, legend=opt_name)
            return fig

        fig_optim = None
        fig_optim = train_with_optimizer('momentum', optimizers.MomentumSGD, fig_optim)
        fig_optim

    def test_search_vanilla(self):
        # ------- pycarm block- -------
        self.setup()
        test = self
        seed = self.seed
        data_dir = self.data_dir
        ds_train = self.ds_train
        ds_test = self.ds_test
        from cs3600.plot import plot_fit
        # ----------------------------
        import pandas as pd
        import pickle
        ##======================todo:yaniv:delete!=============================================
        ##======================todo:yaniv:delete!=============================================
        ##======================todo:yaniv:delete!=============================================

        # Define a larger part of the CIFAR-10 dataset (still not the whole thing)
        batch_size = 50
        max_batches = 100
        in_features = 3 * 32 * 32
        num_classes = 10
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size // 2, shuffle=False)

        # Define a function to train a model with our Trainer and various optimizers
        def train_with_optimizer(opt_name, opt_class, hp):
            torch.manual_seed(seed)

            # Get hyperparameters
            hidden_features = [128] * 5
            num_epochs = 10

            # Create model, loss and optimizer instances
            model = layers.MLP(in_features, num_classes, hidden_features, wstd=hp['wstd'])
            loss_fn = layers.CrossEntropyLoss()
            optimizer = opt_class(model.params(), learn_rate=hp[f'lr_{opt_name}'], reg=hp['reg'])

            # Train with the Trainer
            trainer = training.LayerTrainer(model, loss_fn, optimizer)
            fit_res = trainer.fit(dl_train, dl_test, num_epochs, max_batches=max_batches)

            return fit_res

        best_results = pd.DataFrame(columns=['wstd', 'lr_vanilla', 'lr_momentum', 'lr_rmsprop', 'reg', 'accuracy'])

        for wstd in [0.01, 0.05, 0.1]:
            for reg in np.linspace(1e-5, 0.00015, 15):
                for lr_vanilla in np.linspace(1e-5, 0.00015, 15):
                    hp = dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=None, lr_rmsprop=None, reg=reg)
                    res = train_with_optimizer('vanilla', optimizers.VanillaSGD, hp)
                    best_results = best_results.append(
                        {'wstd': hp['wstd'], 'lr_vanilla': hp['lr_vanilla'], 'lr_momentum': hp['lr_momentum'],
                         'lr_rmsprop': hp['lr_rmsprop'], 'reg': hp['reg'], 'accuracy': max(res.test_acc)},
                        ignore_index=True)
                    print(best_results)
                    with open('.\search_outputs.pickle', mode='wb') as handle:
                        pickle.dump(best_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('********************************************Summary*************************************')
        print(best_results)
        print('****************************************************************************************')


if __name__ == '__main__':
    unittest.main()
