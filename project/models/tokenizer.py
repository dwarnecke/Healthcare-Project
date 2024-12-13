import re
import torch
import torch.nn as nn


class Tokenizer(nn.Module):
    """
    Class to tokenize the input text.
    """

    def __init__(self, pad_token: str = '<UNK>', truncate_limit: int = 256):
        """
        Create the tokenizer.
        :param unk_token: The token to use for unknown words
        :param truncate_limit: The maximum length of the tokenized text
        """

        super().__init__()

        self.pad_token = pad_token
        self.truncate_limit = truncate_limit

    def forward(self, x: list):
        """
        Tokenize the given texts.
        :param x: The sentences to tokenize
        :return: The tokenized texts
        """

        y = []

        # Tokenize the sentences to words
        for text in x:
            text = text.lower()
            text = text.replace('\n', ' ')
            text = text.replace('\t', ' ')
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s]', '', text)

            tokens = text.split()
            y.append(tokens)

        # Resize the sequences to the same length
        max_tokens = min(max(len(tokens) for tokens in y), self.truncate_limit)
        for tokens in y:
            if len(tokens) >= max_tokens:
                tokens[:] = tokens[:max_tokens]
            else:
                tokens.extend([self.pad_token] * (max_tokens - len(tokens)))

        return y




