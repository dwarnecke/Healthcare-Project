import torch
import torch.nn as nn
from project.models.embedder import Embedder
from project.models.tokenizer import Tokenizer
from project.models.transformer import Transformer


class SuicideClassifier(torch.nn.Module):
    """
    Transformer based language classifier model to classify suicidal medical notes.
    """

    def __init__(
            self,
            classes: int,
            dim: int,
            transformers: int = 2,
            linears: int = 1,
            heads: int = 2,
            tokenizer = None,
            embedder = None):
        """
        Create the model and initialize the layers.
        :param classes: The number of classes to predict
        :param dim: The dimension of the model
        :param heads: The number of heads to use in the transformers
        :param transformers: The number of transformer layers to use
        :param forwards: The number of linear layers to use
        :param tokenizer: The tokenizer to use
        :param embedder: The embedding model to use
        """

        super().__init__()

        # Save the necessary model architecture parameters
        self.classes = classes
        self.dimension = dim

        self.tokenizer = Tokenizer() if tokenizer is None else tokenizer
        self.embedder = Embedder() if embedder is None else embedder
        self.initial_layer = nn.Linear(self.embedder.embedding_dim, dim, bias=False)
        self.adaptive_layer = nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten())
        self.final_layer = nn.Linear(dim, classes)

        # Define and initialize the transformer layers 
        self.transformers = nn.Sequential()
        for i in range(transformers):
            module = Transformer(dim, heads=heads)
            self.transformers.add_module(str(i), module)

        # Define and initialize the linear layers
        self.linear_layers = nn.Sequential()
        for i in range(linears):
            module = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
            self.linear_layers.add_module(str(i), module)

    def forward(self, x):
        """
        Forward pass through the classifier model.
        :param x: The input sentences to the model
        :return: The outputs of the model
        """

        # Tokenize and embed the input sentences
        with torch.no_grad():
            y = self.tokenizer(x)
            y = self.embedder(y)

        # Propagate the embeddings through the model
        y = self.initial_layer(y)
        y = self.transformers(y)
        y = torch.permute(y, (0, 2, 1))
        y = self.adaptive_layer(y)
        y = self.linear_layers(y)
        y = self.final_layer(y)

        return y



        