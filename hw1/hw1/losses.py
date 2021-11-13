import abc
import torch
import numpy as np

class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        #-----------------------------vecrtorized solution
        num_samples = x_scores.shape[0]

        marginal_loss = x_scores + self.delta
        marginal_loss = marginal_loss - torch.unsqueeze(x_scores[np.arange(num_samples), y], dim=1) * torch.ones(10)
        marginal_loss[np.arange(num_samples), y] = 0
        marginal_loss[marginal_loss < 0] = 0

        per_sample_loss = marginal_loss.sum(axis=1)
        loss = per_sample_loss.mean()

        #-----------------------------Non-vecrtorized solution
        # num_categories = x_scores.shape[1]
        #
        # loss = 0
        # for i, y_sample in enumerate(y):
        #     loss_sample = 0
        #     x_score_sample = x_scores[i, :]
        #     y_score_sample = x_score_sample[y_sample]
        #     for j in range(num_categories):
        #         if y_sample != j:
        #             iter_loss = max(0, self.delta + x_score_sample[j] - y_score_sample)
        #             loss_sample += iter_loss
        #
        #     loss += loss_sample
        # loss = loss / num_samples

        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx['marginal_loss'] = marginal_loss
        self.grad_ctx['x'] = x
        self.grad_ctx['y'] = y
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        non_zero_marginal_loss = self.grad_ctx['marginal_loss']
        non_zero_marginal_loss[non_zero_marginal_loss > 0] = 1

        y = self.grad_ctx['y']
        x = self.grad_ctx['x']
        num_samples = x.shape[0]

        row_sum = torch.sum(non_zero_marginal_loss, axis=1)
        non_zero_marginal_loss[np.arange(num_samples), y] = -row_sum.T
        grad = (x.T @ non_zero_marginal_loss) / num_samples
        # ========================

        return grad
