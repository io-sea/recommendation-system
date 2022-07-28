#!/usr/bin/env python
"""
This module proposes classes and methods to gather various signals from instrumentation to combine I/O activites into phases distinct from compute activities.
"""


__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import os
import pandas as pd
import random
from app_decomposer.signal_decomposer import KmeansSignalDecomposer

class JobDecomposer:
    """This class takes separate read and write dataflow timeseries in order to extract separated phases for each type: compute, read and write phases."""
    def __init__(self, job_id=None, signal_decomposer=KmeansSignalDecomposer):
        """Initiates JobDecomposer class by fetching job related data.

        Args:
            job_id (int): slurm job id for which data will be retrieved.
            signal_decomposer(SignalDecomposer): class that decomposes each signal into phases.
        """
        self.job_id = job_id
        self.signal_decomposer = signal_decomposer
        self.timestamps, self.read_signal, self.write_signal = self.get_job_timeseries()


    @staticmethod
    def list_jobs(dataset_path):
        """list all present jobs in the dataset folder and return list of files, ids and dataset names.

        Args:
            dataset (os.path): path to the dataset folder.

        Returns:
            job_files, job_ids, dataset_name (tuple): list of files, ids and dataset names.
        """
        job_files = []
        job_ids = []
        dataset_names = []
        for root, dirs, files in os.walk(dataset_path):
            for csv_file in files:
                if csv_file.endswith(".csv"):
                    job_files.append(os.path.join(root, csv_file))
                    job_ids.append(csv_file.split("_")[-1].split(".csv")[0])
                    dataset_names.append(os.path.split(root)[-1])
        return job_files, job_ids, dataset_names

    def get_job_timeseries(self):
        """Method to extract read and write timeseries from a job.
        TODO: connect this method directly to the IOI database.
        For the moment, data will be mocked by a csv file containing the timeseries.

        Returns:
            timestamps, read_signal, write_signal (numpy array): various arrays of the job timeseries.
        """
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        dataset_path = os.path.join(parent_dir, "dataset_generation", "dataset_generation")

        # Get list of jobs
        job_files, job_ids, dataset_names = self.list_jobs(dataset_path=dataset_path)
        if self.job_id is None:
            csv_file = random.choice(job_files)
        else:
            csv_file = job_files[job_ids.index(str(self.job_id))]

        df = pd.read_csv(csv_file, index_col=0)
        return df[["timestamp"]].to_numpy(), df[["bytesRead"]].to_numpy(), df[["bytesWritten"]].to_numpy()

    def get_phases(self):
        """Get phases from each timeserie of the job."""
        read_decomposer = self.signal_decomposer(self.read_signal)
        read_breakpoints, read_labels = read_decomposer.decompose()
        return read_breakpoints, read_labels












