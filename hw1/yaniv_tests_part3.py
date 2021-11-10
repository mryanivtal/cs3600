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


    def test_linear_classifier(self):
        test = self

        import torchvision.transforms as tvtf

        # Define the transforms that should be applied to each image in the dataset before returning it
        tf_ds = tvtf.Compose([
            # Convert PIL image to pytorch Tensor
            tvtf.ToTensor(),
            # Normalize each chanel with precomputed mean and std of the train set
            tvtf.Normalize(mean=(0.1307,), std=(0.3081,)),
            # Reshape to 1D Tensor
            hw1tf.TensorView(-1),
            # Apply the bias trick (add bias element to features)
            hw1tf.BiasTrick(),
        ])
        #---------------------------------------------------------
        import hw1.datasets as hw1datasets
        import hw1.dataloaders as hw1dataloaders

        # Define how much data to load
        num_train = 10000
        num_test = 1000
        batch_size = 1000

        # Training dataset
        data_root = os.path.expanduser('~/.pytorch-datasets')
        ds_train = hw1datasets.SubsetDataset(
            torchvision.datasets.MNIST(root=data_root, download=True, train=True, transform=tf_ds),
            num_train)

        # Create training & validation sets
        dl_train, dl_valid = hw1dataloaders.create_train_validation_loaders(
            ds_train, validation_ratio=0.2, batch_size=batch_size
        )

        # Test dataset & loader
        ds_test = hw1datasets.SubsetDataset(
            torchvision.datasets.MNIST(root=data_root, download=True, train=False, transform=tf_ds),
            num_test)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size)

        x0, y0 = ds_train[0]
        n_features = torch.numel(x0)
        n_classes = 10

        # Make sure samples have bias term added
        test.assertEqual(n_features, 28 * 28 * 1 + 1, "Incorrect sample dimension")
        #----------------------------------------------------------------
        import hw1.linear_classifier as hw1linear

        # Create a classifier
        lin_cls = hw1linear.LinearClassifier(n_features, n_classes)

        # Evaluate accuracy on test set
        mean_acc = 0
        for (x, y) in dl_test:
            y_pred, _ = lin_cls.predict(x)
            mean_acc += lin_cls.evaluate_accuracy(y, y_pred)
        mean_acc /= len(dl_test)

        print(f"Accuracy: {mean_acc:.1f}%")

if __name__ == '__main__':
    unittest.main()
