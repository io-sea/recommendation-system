import unittest
from unittest.mock import patch
import time
import numpy as np
import simpy
from loguru import logger
import sklearn
import os
import pandas as pd
import random

from app_decomposer.job_decomposer import JobDecomposer

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

def get_job_timeseries_from_file(job_id=None):
    """Method to extract read and write timeseries from a job.
    TODO: connect this method directly to the IOI database.
    For the moment, data will be mocked by a csv file containing the timeseries.

    Returns:
        timestamps, read_signal, write_signal (numpy array): various arrays of the job timeseries.
    """
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    dataset_path = os.path.join(parent_dir, "dataset_generation", "dataset_generation")

    # Get list of jobs
    job_files, job_ids, dataset_names = list_jobs(dataset_path=dataset_path)
    if job_id is None:
        csv_file = random.choice(job_files)
    else:
        csv_file = job_files[job_ids.index(str(job_id))]

    df = pd.read_csv(csv_file, index_col=0)
    return df[["timestamp"]].to_numpy(), df[["bytesRead"]].to_numpy(), df[["bytesWritten"]].to_numpy()

class TestJobDecomposer(unittest.TestCase):
    """Test that the app decomposer follows some pattern."""

    def setUp(self):
        """Set up test fixtures, if any."""

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_init_job_dec(self, mock_get_timeseries):
        """Test if JobDecomposer initializes well from dumped files containing job timeseries."""
        # mock the method to return some dataset file content
        mock_get_timeseries.return_value = get_job_timeseries_from_file(job_id=457344)
        # init the job decomposer
        jd = JobDecomposer()
        self.assertEqual(len(jd.timestamps), len(jd.read_signal))

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_init_job_dec_no_jobid(self, mock_get_timeseries):
        """Test if JobDecomposer initializes well with no jobid for mock."""
        # mock the method to return some dataset file content
        mock_get_timeseries.return_value = get_job_timeseries_from_file()
        # init the job decomposer
        jd = JobDecomposer()
        self.assertEqual(len(jd.timestamps), len(jd.read_signal))

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_job_dec_and_output(self, mock_get_timeseries):
        """Test if JobDecomposer gets the expected output."""
        # mock the method to return some dataset file content
        mock_get_timeseries.return_value = get_job_timeseries_from_file(job_id=457344)
        # init the job decomposer
        jd = JobDecomposer()
        bkps, labels, _, _ = jd.get_phases()
        self.assertIsInstance(bkps, list)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(jd.timestamps), len(labels))


    # @patch.object(JobDecomposer, 'get_job_timeseries')
    # def test_job_dec_and_combine(self, mock_get_timeseries):
    #     """Test combining phases from read and write signals."""
    #     timestamps = np.arange(10)
    #     read_signal = np.array([0, 0, 5, 7, 0, 0, 0, 0, 0, 0])
    #     write_signal = np.array([0, 0, 0, 0, 0, 0, 0, 25, 20, 0])
    #     mock_get_timeseries.return_value = timestamps, read_signal, write_signal
    #     # init the job decomposer
    #     jd = JobDecomposer()
    #     #r_breakpoints, r_labels, w_breakpoints, w_labels = jd.get_phases()
    #     results = jd.get_phases()
    #     for result in results:
    #         print(result)

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_job_dec_and_combine(self, mock_get_timeseries):
        """Test combining phases from read and write signals."""
        timestamps = np.arange(10)
        read_signal = np.array([60, 30, 7, 0, 0, 0, 0, 0, 7, 10])
        write_signal = np.array([0, 0, 0, 0, 0, 0, 0, 25, 20, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        #r_breakpoints, r_labels, w_breakpoints, w_labels = jd.get_phases()
        #results = jd.get_phases()
        read_breakpoints, read_labels, _, _ = jd.get_phases()
        print(f"breakpoints = {read_breakpoints}")
        print(f"labels = {read_labels}")
        compute, read, bandwidth = jd.reduce_phase_merge_labels(timestamps, read_signal, read_labels)
        print(f"compute = {compute}")
        print(f"read = {read}")
        print(f"bandwidth = {bandwidth}")


