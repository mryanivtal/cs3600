import numpy as np
import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss

class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None

        # ====== YOUR CODE: ======
        self.weights = torch.randn((n_features, n_classes)) * weight_std
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = x @ self.weights
        y_pred = torch.argmax(class_scores, axis=-1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        acc = (y == y_pred).sum().float() / len(y)
        # ========================

        return acc * 100

    def train(
            self,
            dl_train: DataLoader,
            dl_valid: DataLoader,
            loss_fn: ClassifierLoss,
            learn_rate=0.1,
            weight_decay=0.001,
            max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======

            # Train one full epoch:
            for batch_idx, (x, y) in enumerate(dl_train):
                y_pred, class_scores = self.predict(x)                          # predict using existing weights

                _ = loss_fn.loss(x, y, class_scores, y_pred)                    # Calc loss gradient
                d_reg_d_w = 2 * weight_decay * self.weights                     # regularization diff

                batch_grad = loss_fn.grad() + d_reg_d_w                         # Calc gradient
                self.weights -= learn_rate * batch_grad                         # Update weights

            reg = weight_decay * np.multiply(self.weights, self.weights).sum()  # Weight decay regularization for new w matrix

            # Iterate on all batches in epoch, store accuracy and loss values (Train)
            epoch_train_acc, epoch_train_loss = self.evaluate_epoch_accuracy_and_loss(dl_train, loss_fn, reg)
            train_res.accuracy.append(epoch_train_acc)
            train_res.loss.append(epoch_train_loss)

            # Iterate on all batches in epoch, store accuracy and loss values (Validation)
            epoch_valid_acc, epoch_valid_loss = self.evaluate_epoch_accuracy_and_loss(dl_valid, loss_fn, reg)
            valid_res.accuracy.append(epoch_valid_acc)
            valid_res.loss.append(epoch_valid_loss)

            # Test stop conditions

            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res


    def evaluate_epoch_accuracy_and_loss(self, dl_data: DataLoader, loss_fn:ClassifierLoss, regularization: float) -> (float, float):
        batch_train_accuracy = []
        batch_num_samples = []
        batch_loss = []

        for batch_idx, (x, y) in enumerate(dl_data):
            y_pred, class_scores = self.predict(x)
            batch_train_accuracy.append(self.evaluate_accuracy(y, y_pred))
            batch_num_samples.append(len(y_pred))
            batch_loss.append(loss_fn.loss(x, y, class_scores, y_pred) + regularization)

        batch_train_accuracy = np.array(batch_train_accuracy)
        batch_num_samples = np.array(batch_num_samples)
        batch_loss = np.array(batch_loss)

        epoch_train_accuracy = (batch_train_accuracy * batch_num_samples).sum() / batch_num_samples.sum()
        epoch_avg_loss = batch_loss.sum() / batch_num_samples.sum()

        return epoch_train_accuracy, epoch_avg_loss


    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        pass
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp['weight_std'] = 0.001
    hp['learn_rate'] = 0.01
    hp['weight_decay'] = 0.001
    # ========================

    return hp
