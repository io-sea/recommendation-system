import unittest
from unittest.mock import patch
import time
import numpy as np
import simpy
from loguru import logger
import sklearn
import os
from os.path import dirname, abspath
import pandas as pd
import random

from app_decomposer.job_decomposer import JobDecomposer, get_events_indexes, get_signal_representation, get_phase_volume, phases_to_representation, get_events_indexes, get_events_indexes_no_merge, get_phase_volume

from app_decomposer.signal_decomposer import KmeansSignalDecomposer, get_lowest_cluster

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
    # get out tests, and package folder
    current_dir = dirname(dirname(dirname(abspath(__file__))))
    dataset_path = os.path.join(current_dir, "dataset_generation", "dataset_generation")
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

    def test_get_volume(self):
        """Tests if method that computes phase volume from signal works well."""
        self.assertEqual(get_phase_volume(np.array([1])), 1)
        self.assertEqual(get_phase_volume(np.arange(10),
                                          method="sum",
                                          start_index=2,
                                          end_index=5), 9)
        self.assertEqual(get_phase_volume(np.arange(10), method="sum"), 45)
        self.assertEqual(get_phase_volume(np.arange(10), method="trapz"), 40.5)
        self.assertEqual(get_phase_volume(np.arange(10), method="simps"), 40.5)


    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_get_events_indexes(self, mock_get_timeseries):
        """Test if get_events_indexes and variant with merge works well."""
        timestamps = np.arange(10)
        read_signal = np.array([60, 30, 7, 0, 0, 0, 0, 0, 7, 10])
        write_signal = np.array([0, 0, 0, 0, 0, 0, 0, 25, 20, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        _, read_labels, _, write_labels = jd.get_phases()
        start_points, end_points = get_events_indexes(read_labels, read_signal)
        w_start_points, w_end_points = get_events_indexes(write_labels, write_signal)
        self.assertListEqual(start_points, [0, 8])
        self.assertListEqual(end_points, [3, 10])
        self.assertListEqual(w_start_points, [7])
        self.assertListEqual(w_end_points, [9])

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_get_events_indexes_no_merge(self, mock_get_timeseries):
        """Test if get_events_indexes and variant with no_merge works well."""
        timestamps = np.arange(10)
        read_signal = np.array([60, 30, 7, 0, 0, 0, 0, 0, 7, 10])
        write_signal = np.array([0, 0, 0, 0, 0, 0, 0, 25, 20, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        _, read_labels, _, write_labels = jd.get_phases()
        start_points, end_points = get_events_indexes_no_merge(read_labels, read_signal)
        w_start_points, w_end_points = get_events_indexes_no_merge(write_labels, write_signal)
        self.assertListEqual(start_points, [0, 1, 2, 8])
        self.assertListEqual(end_points, [1, 2, 3, 10])
        self.assertListEqual(w_start_points, [7])
        self.assertListEqual(w_end_points, [9])

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_get_events_indexes_no_merge_1(self, mock_get_timeseries):
        """Test if get_events_indexes and variant merge works well."""
        timestamps = np.arange(5)
        read_signal = np.array([0, 1, 2, 2, 0])
        write_signal = np.array([0, 1, 2, 2, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        _, read_labels, _, write_labels = jd.get_phases()
        print(f"read labels = {read_labels}")
        start_points, end_points = get_events_indexes_no_merge(read_labels, read_signal)
        compute, data, bw = phases_to_representation(start_points, end_points, read_signal, dx=1)
        print(f"compute={compute}, data={data}, bandwidth={bw}")

        self.assertListEqual(start_points, [1, 2])
        self.assertListEqual(end_points, [2, 4])
        self.assertListEqual(compute, [0, 1, 2, 3])
        self.assertListEqual(data, [0, 1, 4, 0])
        self.assertListEqual(bw, [0, 1, 2, 0])

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_get_events_indexes_no_merge_2(self, mock_get_timeseries):
        """Test if get_events_indexes and variant merge works well."""
        timestamps = np.arange(5)
        read_signal = np.array([0, 2, 1, 1, 0])
        write_signal = np.array([0, 1, 2, 2, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        _, read_labels, _, write_labels = jd.get_phases()
        print(f"read labels = {read_labels}")
        start_points, end_points = get_events_indexes_no_merge(read_labels, read_signal)
        compute, data, bw = phases_to_representation(start_points, end_points, read_signal, dx=1)
        print(f"compute={compute}, data={data}, bandwidth={bw}")
        self.assertListEqual(compute, [0, 1, 2])
        self.assertListEqual(data, [0,  4, 0])
        self.assertListEqual(bw, [0, 4/3, 0])

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_get_events_indexes_no_merge_3(self, mock_get_timeseries):
        """Test if get_events_indexes and variant merge works well."""
        timestamps = np.arange(5)
        read_signal = np.array([1, 1, 1, 0])
        write_signal = np.array([1, 1, 1, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        _, read_labels, _, write_labels = jd.get_phases()
        print(f"read labels = {read_labels}")
        start_points, end_points = get_events_indexes_no_merge(read_labels, read_signal)
        compute, data, bw = phases_to_representation(start_points, end_points, read_signal, dx=1)
        print(f"compute={compute}, data={data}, bandwidth={bw}")
        self.assertListEqual(start_points, [0])
        self.assertListEqual(end_points, [3])
        self.assertListEqual(compute, [0, 1])
        self.assertListEqual(data, [3, 0])
        self.assertListEqual(bw, [1, 0])

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_get_events_indexes_no_merge_4(self, mock_get_timeseries):
        """Test if get_events_indexes and variant merge works well."""
        timestamps = np.arange(5)
        read_signal = np.array([1, 3, 3, 1, 0])
        write_signal = np.array([1, 3, 3, 1, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        _, read_labels, _, write_labels = jd.get_phases()
        print(f"read labels = {read_labels}")
        start_points, end_points = get_events_indexes_no_merge(read_labels, read_signal)
        compute, data, bw = phases_to_representation(start_points, end_points, read_signal, dx=1)
        print(f"compute={compute}, data={data}, bandwidth={bw}")
        self.assertListEqual(start_points, [0, 1, 3])
        self.assertListEqual(end_points, [1, 3, 4])
        self.assertListEqual(compute, [0, 1, 2, 3])
        self.assertListEqual(data, [1, 6, 1, 0])
        self.assertListEqual(bw, [1, 3, 1, 0])

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_read_signal_decomposer_pattern_1(self, mock_get_timeseries):
        """Test extracting representation from read signal having different type of patterns. Pattern here has IO at the begining and end of the job."""
        timestamps = np.arange(10)
        read_signal = np.array([60, 30, 7, 0, 0, 0, 0, 0, 7, 10])
        write_signal = np.array([0, 0, 0, 0, 0, 0, 0, 25, 20, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        #r_breakpoints, r_labels, w_breakpoints, w_labels = jd.get_phases()
        #results = jd.get_phases()
        read_breakpoints, read_labels, _, _ = jd.get_phases()
        print(f"signal = {read_signal}")
        print(f"labels = {read_labels}")
        compute, read, bandwidth = get_signal_representation(timestamps, read_signal, read_labels, merge_clusters=True)
        print(f"compute={compute}, read={read}, bandwidth={bandwidth}")
        self.assertEqual(len(compute), 2) # should be compute = [0, 6]
        self.assertListEqual(compute, [0, 6])
        self.assertListEqual(read, [97, 17])

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_read_signal_decomposer_pattern_1_no_merge(self, mock_get_timeseries):
        """Test extracting representation from read signal having different type of patterns. Pattern here has IO at the begining and end of the job."""
        timestamps = np.arange(10)
        read_signal = np.array([60, 30, 7, 0, 0, 0, 0, 0, 7, 10])
        write_signal = np.array([0, 0, 0, 0, 0, 0, 0, 25, 20, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        #r_breakpoints, r_labels, w_breakpoints, w_labels = jd.get_phases()
        #results = jd.get_phases()
        read_breakpoints, read_labels, _, _ = jd.get_phases()
        print(f"signal = {read_signal}")
        print(f"labels = {read_labels}")
        compute, read, bandwidth = get_signal_representation(timestamps, read_signal, read_labels, merge_clusters=False)
        print(f"compute={compute}, read={read}, bandwidth={bandwidth}")
        self.assertListEqual(compute, [0, 1, 2, 8])
        self.assertListEqual(read, [60, 30, 7, 17])
        self.assertListEqual(bandwidth, [60.0, 30.0, 7.0, 8.5])

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_read_signal_decomposer_pattern_2(self, mock_get_timeseries):
        """Test extracting representation from read signal having different type of patterns. Pattern here has an IO in the middle of the job."""
        timestamps = np.arange(6)
        read_signal = np.array([0, 0, 10, 8, 0, 0])
        write_signal = np.array([0, 0, 0, 0, 0, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        #r_breakpoints, r_labels, w_breakpoints, w_labels = jd.get_phases()
        #results = jd.get_phases()
        read_breakpoints, read_labels, _, _ = jd.get_phases()
        print(f"signal = {read_signal}")
        print(f"labels = {read_labels}")
        compute, read, bandwidth = get_signal_representation(timestamps, read_signal, read_labels, merge_clusters=True)
        print(f"compute={compute}, read={read}, bandwidth={bandwidth}")
        self.assertEqual(len(compute), 3) # should be compute = [0, 2, 4]
        self.assertListEqual(compute, [0, 2, 4])
        self.assertListEqual(read, [0, 18, 0])

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_read_signal_decomposer_pattern_2_no_merge(self, mock_get_timeseries):
        """Test extracting representation from read signal having different type of patterns. Pattern here has an IO in the middle of the job."""
        timestamps = np.arange(6)
        read_signal = np.array([0, 0, 10, 8, 0, 0])
        write_signal = np.array([0, 0, 0, 0, 0, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        #r_breakpoints, r_labels, w_breakpoints, w_labels = jd.get_phases()
        #results = jd.get_phases()
        read_breakpoints, read_labels, _, _ = jd.get_phases()
        print(f"signal = {read_signal}")
        print(f"labels = {read_labels}")
        compute, read, bandwidth = get_signal_representation(timestamps, read_signal, read_labels, merge_clusters=True)
        self.assertListEqual(compute, [0, 2, 4])
        self.assertListEqual(read, [0, 18, 0])
        self.assertListEqual(bandwidth, [0, 9, 0])


    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_read_signal_decomposer_pattern_3(self, mock_get_timeseries):
        """Test extracting representation from read signal having different type of patterns. Pattern here has complex IO pattern."""
        timestamps = np.arange(6)
        read_signal = np.array([12, 0, 10, 8, 0, 19, 0])
        write_signal = np.array([0, 0, 0, 0, 0, 0, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        #r_breakpoints, r_labels, w_breakpoints, w_labels = jd.get_phases()
        #results = jd.get_phases()
        read_breakpoints, read_labels, _, _ = jd.get_phases()
        print(f"signal = {read_signal}")
        print(f"labels = {read_labels}")
        compute, read, bandwidth = get_signal_representation(timestamps, read_signal, read_labels, merge_clusters=True)
        print(f"compute={compute}, read={read}, bandwidth={bandwidth}")
        self.assertEqual(len(compute), 4) # should be compute = [0, 2, 4, 5]
        self.assertListEqual(compute, [0, 2, 4, 5])
        self.assertListEqual(read, [12, 18, 19, 0])

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_read_signal_decomposer_pattern_3_no_merge(self, mock_get_timeseries):
        """Test extracting representation from read signal having different type of patterns. Pattern here has complex IO pattern."""
        timestamps = np.arange(6)
        read_signal = np.array([12, 0, 10, 8, 0, 19, 0])
        write_signal = np.array([0, 0, 0, 0, 0, 0, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        #r_breakpoints, r_labels, w_breakpoints, w_labels = jd.get_phases()
        #results = jd.get_phases()
        read_breakpoints, read_labels, _, _ = jd.get_phases()
        print(f"signal = {read_signal}")
        print(f"labels = {read_labels}")
        compute, read, bandwidth = get_signal_representation(timestamps, read_signal, read_labels, merge_clusters=False)
        print(f"compute={compute}, read={read}, bandwidth={bandwidth}")
        self.assertListEqual(compute, [0, 2, 4, 5])
        self.assertListEqual(read, [12, 18, 19, 0])
        self.assertListEqual(bandwidth, [12.0, 9.0, 19.0, 0.0])


    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_read_signal_decomposer_pattern_4(self, mock_get_timeseries):
        """Test getting phases from read and write signals. Pattern here has complex IO pattern."""
        timestamps = np.arange(6)
        read_signal = np.array([0, 0, 10, 8, 0, 19, 20])
        write_signal = np.array([0, 0, 0, 0, 0, 0, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        #r_breakpoints, r_labels, w_breakpoints, w_labels = jd.get_phases()
        #results = jd.get_phases()
        read_breakpoints, read_labels, _, _ = jd.get_phases()
        print(f"signal = {read_signal}")
        print(f"labels = {read_labels}")
        compute, read, bandwidth = get_signal_representation(timestamps, read_signal, read_labels, merge_clusters=True)
        print(f"compute={compute}, read={read}, bandwidth={bandwidth}")
        self.assertEqual(len(compute), 3) # should be compute = [0, 2, 4]
        self.assertListEqual(compute, [0, 2, 4])
        self.assertListEqual(read, [0, 18, 39])

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_read_signal_decomposer_pattern_4_no_merge(self, mock_get_timeseries):
        """Test getting phases from read and write signals. Pattern here has complex IO pattern."""
        timestamps = np.arange(6)
        read_signal = np.array([0, 0, 10, 8, 0, 19, 20])
        write_signal = np.array([0, 0, 0, 0, 0, 0, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        #r_breakpoints, r_labels, w_breakpoints, w_labels = jd.get_phases()
        #results = jd.get_phases()
        read_breakpoints, read_labels, _, _ = jd.get_phases()
        print(f"signal = {read_signal}")
        print(f"labels = {read_labels}")
        compute, read, bandwidth = get_signal_representation(timestamps, read_signal, read_labels, merge_clusters=False)
        print(f"compute={compute}, read={read}, bandwidth={bandwidth}")
        self.assertListEqual(compute, [0, 2, 4])
        self.assertListEqual(read, [0, 18, 39])
        self.assertListEqual(bandwidth, [0, 9.0, 19.5])



    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_read_signal_decomposer_pattern_5_no_merge(self, mock_get_timeseries):
        """Test extracting representation from read signal having different type of patterns. Pattern here has complex IO pattern."""
        timestamps = np.arange(5)
        read_signal = np.array([0, 1, 1, 0, 0])
        write_signal = np.array([0, 0, 0, 0, 1])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        #r_breakpoints, r_labels, w_breakpoints, w_labels = jd.get_phases()
        #results = jd.get_phases()
        read_breakpoints, read_labels, _, write_labels = jd.get_phases()
        print(f"signal = {read_signal}")
        print(f"labels = {read_labels}")
        compute, read, bandwidth = get_signal_representation(timestamps, read_signal, read_labels)
        compute_, write, write_bandwidth = get_signal_representation(timestamps, write_signal, write_labels, merge_clusters=False)
        print(f"compute={compute}, read={read}, bandwidth={bandwidth}")
        print(f"compute={compute_}, write={write}, write_bandwidth={write_bandwidth}")
        self.assertListEqual(compute, [0, 1, 3])
        self.assertListEqual(read, [0, 2, 0])
        self.assertListEqual(bandwidth, [0, 1, 0])


    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_job_decomposer_pattern_1(self, mock_get_timeseries):
        """Test combining phases from read and write signals to get a representation."""
        timestamps = np.arange(5)
        read_signal = np.array([0, 1, 1, 0, 0, 0, 0])
        write_signal = np.array([0, 0, 0, 0, 1, 1, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        events, reads, writes, _, _ = jd.get_job_representation(merge_clusters=True)
        print(f"compute={events}, read={reads}, writes={writes}")
        self.assertListEqual(events, [0, 1, 3, 4])
        self.assertListEqual(reads, [0, 2, 0, 0])
        self.assertListEqual(writes, [0, 0, 2, 0])

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_job_decomposer_pattern_1_no_merge(self, mock_get_timeseries):
        """Test combining phases from read and write signals to get a representation."""
        timestamps = np.arange(5)
        read_signal = np.array([0, 1, 1, 0, 0, 0, 0])
        write_signal = np.array([0, 0, 0, 0, 1, 1, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        events, reads, writes, _, _ = jd.get_job_representation(merge_clusters=False)
        print(f"compute={events}, read={reads}, writes={writes}")
        self.assertListEqual(events, [0, 1, 3, 4])
        self.assertListEqual(reads, [0, 2, 0, 0])
        self.assertListEqual(writes, [0, 0, 2, 0])

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_job_decomposer_pattern_2(self, mock_get_timeseries):
        """Test combining phases from read and write signals to get a representation."""
        timestamps = np.arange(5)
        read_signal = np.array([0, 1, 1, 0, 0, 0, 0])
        write_signal = np.array([0, 0, 0, 0, 0, 1, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        events, reads, writes, _, _ = jd.get_job_representation(merge_clusters=True)
        print(f"compute={events}, read={reads}, writes={writes}")
        self.assertListEqual(events, [0, 1, 4, 5])
        self.assertListEqual(reads, [0, 2, 0, 0])
        self.assertListEqual(writes, [0, 0, 1, 0])

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_job_decomposer_pattern_2_no_merge(self, mock_get_timeseries):
        """Test combining phases from read and write signals to get a representation."""
        timestamps = np.arange(5)
        read_signal = np.array([0, 1, 1, 0, 0, 0, 0])
        write_signal = np.array([0, 0, 0, 0, 0, 1, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()

        read_bkps, read_labels, write_bkps, write_labels = jd.get_phases()
        compute, read, bandwidth = get_signal_representation(timestamps, read_signal, read_labels, merge_clusters=False)
        print(f"read signal : compute={compute}, read={read}, bw={bandwidth}")
        compute, write, bandwidth = get_signal_representation(timestamps, write_signal, write_labels, merge_clusters=False)
        print(f"write signal : compute={compute}, write={write}, bw={bandwidth}")
        events, reads, writes, _, _ = jd.get_job_representation(merge_clusters=False)

        print(f"compute={events}, read={reads}, writes={writes}")
        self.assertListEqual(events, [0, 1, 4, 5])
        self.assertListEqual(reads, [0, 2, 0, 0])
        self.assertListEqual(writes, [0, 0, 1, 0])

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_job_decomposer_pattern_3(self, mock_get_timeseries):
        """Test combining phases from read and write signals to get a representation."""
        timestamps = np.arange(5)
        read_signal = np.array([1, 1, 1, 0, 0, 0])
        write_signal = np.array([0, 0, 0, 1, 1, 1])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        events, reads, writes, _, _ = jd.get_job_representation(merge_clusters=True)
        print(f"compute={events}, read={reads}, writes={writes}")
        self.assertListEqual(events, [0, 1])
        self.assertListEqual(reads, [3, 0])
        self.assertListEqual(writes, [0, 3])

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_job_decomposer_pattern_3_no_merge(self, mock_get_timeseries):
        """Test combining phases from read and write signals to get a representation."""
        timestamps = np.arange(5)
        read_signal = np.array([1, 1, 1, 0, 0, 0])
        write_signal = np.array([0, 0, 0, 1, 1, 1])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        events, reads, writes, _, _ = jd.get_job_representation(merge_clusters=False)
        print(f"compute={events}, read={reads}, writes={writes}")
        self.assertListEqual(events, [0, 1])
        self.assertListEqual(reads, [3, 0])
        self.assertListEqual(writes, [0, 3])

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_job_decomposer_pattern_simulataneous_read_write(self, mock_get_timeseries):
        """Test combining phases from read and write signals to get a representation."""
        timestamps = np.arange(5)
        read_signal = np.array([1, 1, 1, 0, 0, 0])
        write_signal = np.array([1, 1, 1, 0, 0, 1])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        events, reads, writes, _, _ = jd.get_job_representation(merge_clusters=True)
        print(f"compute={events}, read={reads}, writes={writes}")
        self.assertListEqual(events, [0, 1])
        self.assertListEqual(reads, [3, 0])
        self.assertListEqual(writes, [3, 1])

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_job_decomposer_pattern_bw_1(self, mock_get_timeseries):
        """Test combining phases from read and write signals to get a representation with bandwidths."""
        timestamps = np.arange(5)
        read_signal = np.array([0, 10, 12, 0, 0, 0, 0])
        write_signal = np.array([0, 0, 0, 0, 10, 30, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        events, reads, writes, read_bw, write_bw = jd.get_job_representation(merge_clusters=True)
        print(f"compute={events}, read_bw={read_bw}, write_bw={write_bw}")
        self.assertListEqual(events, [0, 1, 3, 4])
        self.assertListEqual(reads, [0, 22, 0, 0])
        self.assertListEqual(writes, [0, 0, 40, 0])
        self.assertListEqual(read_bw, [0, 11, 0, 0])
        self.assertListEqual(write_bw, [0, 0, 20, 0])

