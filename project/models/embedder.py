import numpy as np
import torch
import torch.nn as nn


class Embedder(nn.Module):
    """
    Module to embed the input tokens.
    """

    def __init__(self, path: str = 'project/datasets/glove.6B.200d.txt'):
        """
        Create the embedding module.
        :param path: The path to the embeddings file
        :param unk_token: The token to use for unknown words
        """

        super().__init__()

        # Load file embeddings into local memory
        embeddings = {}
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        embeddings['<UNK>'] = np.zeros((1, len(vector)), dtype='float32')

        # Convert the embeddings into a torch module
        self.embedding_index = {word: i for i, word in enumerate(embeddings.keys())}
        self.num_embeddings = len(embeddings)
        self.embedding_dim = len(vector)
        embedding_mat = np.empty((self.num_embeddings, self.embedding_dim), dtype='float32')
        for word, idx in self.embedding_index.items():
            embedding_mat[idx] = embeddings[word]
        self.embedding_layer = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding_layer.weight = nn.Parameter(torch.tensor(embedding_mat))

        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.tensor):
        """
        Forward pass through the embedding module.
        :param x: The token inputs to the embedding module
        :return: The outputs of the embedding module
        """

        # Index the token embedding indices
        num_sequences = len(x)
        num_tokens = len(x[0])
        y = torch.empty((num_sequences, num_tokens), dtype=torch.int)
        for i in range(num_sequences):
            for j in range(num_tokens):
                y[i][j] = self.embedding_index.get(x[i][j], self.num_embeddings-1)

        # Embed the tokens indices to their vectors
        y = y.to(self.embedding_layer.weight.device)
        y = self.embedding_layer(y)
        y = self.dropout(y)

        return y



