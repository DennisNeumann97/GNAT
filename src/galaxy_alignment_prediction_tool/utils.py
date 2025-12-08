import os
import sys
import numpy as np
import h5py
import logging

class ioUtils:
    def __init__(
        self,
        logger=logging.getLogger(__name__),
    ):
        
        self.logger = logger

    def _h5_to_dict(
        self, 
        obj
    ):
        """
        Recursively convert an h5py Group or Dataset into a Python dict or value.
        """
        if isinstance(obj, h5py.Dataset):
            return obj[()]                     # Base case: dataset → numpy array/scalar
        elif isinstance(obj, h5py.Group):
            return {key: self._h5_to_dict(obj[key])  # Recursive case: group → dict
                    for key in obj.keys()}
        else:
            return None  # Shouldn't occur, but keeps it safe.

    def load_h5_recursive(self, file_path):
        with h5py.File(file_path, 'r') as h5file:
            return {key: self._h5_to_dict(h5file[key]) for key in h5file.keys()}
        
    def is_in_snapshot_dict(
        self,
        snapshot_dict: dict,
        key: str
    ) -> bool:
        """
        Check if a key is in the snapshot dictionary.

        Args:
            snapshot_dict (dict): Dictionary containing snapshot data.
            key (str): Key to check for.
        Returns:
            bool: True if key is in the dictionary, False otherwise.
        """

        return key in snapshot_dict