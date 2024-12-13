import torch
import torch.nn as nn


class Transformer(nn.Module):
    """
    Transformer module that uses self attention.
    """

    def __init__(self, dimension, heads: int = 8):
        """
        Create a transformer module that uses self attention.
        :param dimension: The dimension of the model encodings
        :param heads: The number of heads in the multi-head attention
        """

        super().__init__()

        self.attention_norm = nn.LayerNorm(dimension)
        self.self_attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.linear_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.linear = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the transformer module.
        :param x: The inputs to the transformer module
        :return: The outputs of the transformer module
        """

        # Propagate through the attention module
        y = self.attention_norm(x)
        y, _ = self.self_attention(y, y, y)
        y = y + x

        # Propagate through the linear module
        y = self.linear_norm(y)
        y = self.linear(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = y + x

        return y



