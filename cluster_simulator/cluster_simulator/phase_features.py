import simpy
from loguru import logger
import numpy as np
import pandas as pd
import math
from enum import Enum


class Pattern(str, Enum):
    SEQ = "seq"
    RAND = "rand"
    STRIDE = "stride"


class Operation(str, Enum):
    READ = "read"
    WRITE = "write"


class PhaseFeatures:
    """
    Class to encapsulate the features of an I/O phase. This class is used to generalize the parameters of an I/O phase and make it easier to manage and manipulate these parameters.
    """

    def __init__(self, cores=1, operation=None, volume=None,
                 read_volume=None, write_volume=None,
                 pattern=None,
                 read_io_pattern=Pattern.SEQ, write_io_pattern=Pattern.SEQ,
                 read_io_size=4e3, write_io_size=4e3,
                 data=None, appname=None, bw=None):
        """
        Initialize the PhaseFeatures class with the given parameters.

        Args:
            cores (int): Number of cores used for the I/O phase.
            operation (str): Type of operation (read, write, or both).
            pattern (Pattern): I/O pattern (sequential or random).
            read_volume (float): Volume of read operations.
            write_volume (float): Volume of write operations.
            read_io_pattern (Pattern): Pattern of read operations.
            write_io_pattern (Pattern): Pattern of write operations.
            read_io_size (float): Size of read I/O operations.
            write_io_size (float): Size of write I/O operations.
        """
        self.cores = cores
        self.operation = operation
        self.volume = volume
        self.read_volume = read_volume
        self.write_volume = write_volume
        self.pattern = pattern
        self.read_io_pattern = read_io_pattern
        self.write_io_pattern = write_io_pattern
        self.read_io_size = read_io_size
        self.write_io_size = write_io_size
        self.bw = bw*1e6 if bw else bw

        # Determine the operation type based on the read and write volumes
        if operation is None :
            if self.read_volume and not self.write_volume:
                self.operation = Operation.READ
            elif self.write_volume and not self.read_volume:
                self.operation = Operation.WRITE
            elif self.read_volume == 0 and self.write_volume == 0:
                self.operation = None

        if operation == Operation.READ:
            # take read_volume from volume
            self.read_volume = self.volume if self.volume else self.read_volume
            self.write_volume = 0

            # if self.volume is None and self.read_volume is None:
            #     self.read_volume = 1e9
            if pattern is not None:
                if isinstance(pattern, str):
                    self.read_io_pattern = pattern
                if isinstance(pattern, (float, int)):
                    self.read_io_pattern = "seq" if pattern > 0.5 else "rand"

            if self.read_volume is None:
                self.read_volume = 1e9

        if operation == Operation.WRITE:
            self.write_volume = self.volume if self.volume else self.write_volume
            self.read_volume = 0
            if pattern is not None:
                if isinstance(pattern, str):
                    self.write_io_pattern = pattern
                if isinstance(pattern, (float, int)):
                    self.write_io_pattern = "seq" if pattern > 0.5 else "rand"

            if self.write_volume is None:
                self.write_volume = 1e9

    def get_attributes(self):
        """
        Returns a dictionary of selected attributes of the PhaseFeatures class.

        Returns:
            dict: A dictionary of attribute names and values.

        """
        attribute_list = ['read_io_size', 'write_io_size',
                          'read_volume', 'write_volume',
                          'read_io_pattern', 'write_io_pattern']
        # nodes instead of cores
        attribute_dict = {"nodes": [getattr(self, "cores")]}
        for attribute_name in attribute_list:
            attribute_dict[attribute_name] = [getattr(self, attribute_name)]
        return attribute_dict

    def get_dataframe(self):
        """
        Returns a pandas DataFrame with the attributes of the PhaseFeatures class.

        Returns:
            pd.DataFrame: A DataFrame with specific columns.
        """
        # Define the data for the DataFrame
        data = {
            'nodes': [self.cores],
            'read_volume': [self.read_volume],
            'write_volume': [self.write_volume],
            'read_io_size': [self.read_io_size],
            'write_io_size': [self.write_io_size],
            'read_io_pattern': [self.read_io_pattern],
            'write_io_pattern': [self.write_io_pattern]
        }

        # Create and return the DataFrame
        return pd.DataFrame(data)

