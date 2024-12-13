import pandas as pd
from torch.utils.data import Dataset


class SuicidalDataset(Dataset):
    """
    Dataset class for the suicide classifier.
    """

    def __init__(self, path):
        """
        Create the dataset.
        :param path: The path to the dataset
        """

        self.data = pd.read_csv(path)

    def __len__(self):
        """
        Get the length of the dataset.
        :return: The length of the dataset
        """

        length = self.data.shape[0]

        return length

    def __getitem__(self, idx):
        """
        Get the item at the given index.
        :param idx: The index to get
        :return: The item at the given index
        """

        text = self.data['text'][idx]
        label = self.data['suicidal'][idx]

        return text, label


class CodeDataset(Dataset):
    """
    Dataset class for the code classifier.
    """

    def __init__(self, path):
        """
        Create the dataset.
        :param path: The path to the dataset
        """

        # Use only the suicidal data
        self.data = pd.read_csv(path)
        self.data = self.data[self.data['suicidal'] == 1]
        self.data = self.data.reset_index()

        # Define the ICD integer code map for training
        self.code_map = {
            'T36-T50': 0,
            'X71-X83': 1,
            'R45.851': 2,
            'T71': 3,
            'T14.91': 4,
            'T51-T65': 5,
            'unsure': 6}

    def __len__(self):
        """
        Get the length of the dataset.
        :return: The length of the dataset
        """

        length = self.data.shape[0]

        return length

    def __getitem__(self, idx):
        """
        Get the item at the given index.
        :param idx: The index to get
        :return: The item at the given index
        """

        text = self.data['text'][idx]
        code = self.data['code'][idx]
        code = self.code_map[code]

        return text, code
        