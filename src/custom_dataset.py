from numba import njit
import pandas as pd
import torch as tc
import numpy as np
import torch.utils.data as tcud
from directory_manager import PathManager

path_manager = PathManager()
logger = path_manager.get_logger()


class DMDataset(tcud.Dataset):
    """
    Custom dataset class for handling dark matter simulation data.

    This class prepares input features and target values for training a machine learning model.
    It supports both pandas DataFrame and NumPy array formats.

    Attributes:
        features (torch.FloatTensor): Tensor representation of the input features.
        targets (torch.FloatTensor): Tensor representation of the target values.
    """

    def __init__(self, features, targets):
        """
        Initializes the dataset with input features and target values.

        Args:
            features (pd.DataFrame or np.ndarray): Input feature data.
            targets (pd.DataFrame or np.ndarray): Target labels corresponding to the features.
        """
        # Convert features to tensor
        if isinstance(features, pd.DataFrame):
            self.features = tc.FloatTensor(features.to_numpy())
        elif isinstance(features, np.ndarray):
            self.features = tc.FloatTensor(features)
        else:
            logger.error("Features must be either a pandas DataFrame or a numpy array.")
            raise ValueError("Features must be either a pandas DataFrame or a numpy array.")

        # Convert targets to tensor
        if isinstance(targets, pd.DataFrame):
            self.targets = tc.FloatTensor(targets.values.reshape(-1, 1))
        elif isinstance(targets, np.ndarray):
            self.targets = tc.FloatTensor(targets.reshape(-1, 1))
        else:
            logger.error("Targets must be either a pandas DataFrame or a numpy array.")
            raise ValueError("Targets must be either a pandas DataFrame or a numpy array.")


    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The total number of data points.
        """
        return len(self.features)

    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset.


        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the feature tensor and target tensor for the given index.
        """
        features = self.features[index].float()
        target = self.targets[index].float()
        return features, target


    def data_loader(self, batch_size=32, shuffle=True, num_workers=0, pin_memory=True, prefetch_factor=2, persistent_workers=True):
        """
        Creates a DataLoader for the dataset.

        Args:
            batch_size (int, optional): Number of samples per batch (default is 32).
            shuffle (bool, optional): Whether to shuffle the dataset before each epoch (default is True).
            num_workers (int, optional): Number of subprocesses to use for data loading (default is 0).
            pin_memory (bool, optional): Whether to copy tensors into CUDA pinned memory (default is True).
            prefetch_factor (int, optional): Number of batches to prefetch (default is 2).
            
        Returns:
            torch.utils.data.DataLoader: DataLoader for the dataset.
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            logger.debug("Batch_size should be a positive integer value.")

        return tcud.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            drop_last=True  # Avoid small last batch that slows down training
        )

