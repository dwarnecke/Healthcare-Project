import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot


class ConfusionMatrix:
    """
    Class to calculate the confusion matrix for a binary classification problem.
    """

    def __init__(self, classes):
        """
        Create the confusion matrix.
        :param classes: The number of classes in the matrix
        """

        self.classes = classes
        self.matrix = torch.zeros((classes, classes), device=torch.device('cuda'))

    def reset(self):
        """
        Reset the confusion matrix.
        """

        self.matrix *= 0

    def add(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Add the given predictions to the confusion matrix.
        :param y_pred: The model predictions
        :param y_true: The true labels
        """

        # Pivot the predictions and labels to one-hot vectors
        y_pred = torch.argmax(y_pred, dim=-1)
        y_pred = one_hot(y_pred, num_classes=self.classes).to(dtype=torch.float32)
        y_true = one_hot(y_true, num_classes=self.classes).to(dtype=torch.float32)

        self.matrix += torch.matmul(y_pred.t(), y_true)  

    def calculate(self, weigh_scores=True):
        """
        Calculate the confusion matrix accuracy, recall, precision, and F1 score.
        :return: The confusion matrix accuracy, recall, precision, and F1 score
        """

        matrix = self.matrix

        # Calculate the model accuracy
        correct = int(matrix.diagonal().sum())
        total = int(matrix.sum().item())
        accuracy = correct / total

        # Calculate the model F1 score
        epsilon = 1e-5
        if self.classes == 2:
            recall = float(matrix[1, 1] / (matrix[1, 1] + matrix[0, 1] + epsilon))
            precision = float(matrix[1, 1] / (matrix[1, 1] + matrix[1, 0] + epsilon))
            f1 = 2 * recall * precision / (recall + precision + epsilon)
        else:
            recall = matrix.diagonal() / (matrix.sum(dim=0) + epsilon)
            precision = matrix.diagonal() / (matrix.sum(dim=1) + epsilon)
            f1 = 2 * recall * precision / (recall + precision + epsilon)
            if weigh_scores: 
                f1 = float(torch.sum(f1 * matrix.sum(dim=0)) / total)

        return accuracy, recall, precision, f1






















        