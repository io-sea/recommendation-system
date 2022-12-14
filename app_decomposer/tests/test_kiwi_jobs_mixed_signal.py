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
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from cluster_simulator.cluster import Cluster, Tier, EphemeralTier, bandwidth_share_model, compute_share_model, get_tier, convert_size
from cluster_simulator.phase import DelayPhase, ComputePhase, IOPhase
from cluster_simulator.application import Application
from cluster_simulator.analytics import display_run, display_run_with_signal


from app_decomposer.job_decomposer import JobDecomposer, ComplexDecomposer, get_events_indexes, get_signal_representation, get_phase_volume, phases_to_representation, complex_to_representation, get_events_indexes, get_events_indexes_no_merge, get_phase_volume, is_interval_in

from app_decomposer.config_parser import Configuration
from app_decomposer.signal_decomposer import KmeansSignalDecomposer, get_lowest_cluster

from cluster_simulator.analytics import get_execution_signal, get_execution_signal_2, display_original_sim_signals

from app_decomposer import API_DICT_TS

SHOW_FIGURE = False

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
    df_clean = df.drop_duplicates(subset=['timestamp'])
    # timeseries = {}
    # timeseries["volume"] = {}
    # timeseries["volume"]["timestamp"] = df_clean[["timestamp"]].to_numpy().flatten()
    # timeseries["volume"]["bytesRead"] = df_clean[["bytesRead"]].to_numpy().flatten()
    # timeseries["volume"]["bytesWritten"] = df_clean[["bytesWritten"]].to_numpy().flatten()
    # timeseries["operationsCount"] = {}
    # timeseries["operationsCount"]["operationRead"] = df_clean[["operationRead"]].to_numpy().flatten()
    # timeseries["operationsCount"]["operationWrite"] = df_clean[["operationWrite"]].to_numpy().flatten()
    # timeseries["accessPattern"]
    timeseries = {}
    for ts_type in ["volume", "operationsCount", "accessPattern"]:
        timeseries[ts_type] = {}
        ts_list = API_DICT_TS[ts_type]
        ts_list.append("timestamp")
        for ts in ts_list:
            timeseries[ts_type][ts] = df_clean[[ts]].to_numpy().flatten()
    return timeseries
    #return , df_clean[["bytesRead"]].to_numpy(), df_clean[["bytesWritten"]].to_numpy()

def plot_job_signal(jobid=None):
    x, read_signal, write_signal = get_job_timeseries_from_file(job_id=jobid)
    # plt.rcParams["figure.figsize"] = (20, 5)
    # plt.rcParams['figure.facecolor'] = 'gray'

    plt.plot(x, read_signal, label="read signal")
    plt.plot(x, write_signal, label="write signal")
    plt.grid(True)
    plt.legend()
    plt.title(f"timeserie for jobid = {jobid}")
    plt.show()

class TestJobDecomposerFeatures(unittest.TestCase):
    """Examine and qualify JobDecomposer output phases features and signal representation mixes read and write values within the same phase."""


    def test_get_phases_features_symetric(self):
        """Test if JobDecomposer issues phases features in suitable format."""
        # mock the representation issued by the job decomposer.
        representation = {
                'node_count': 1,
                'events': [0, 1],
                'read_volumes': [0, 50],
                'read_bw': [0, 10.0],
                'write_volumes': [0, 50],
                'write_bw': [0, 5],
                'read_pattern': ['Uncl', 'Str'],
                'write_pattern': ['Uncl', 'Str'],
                'read_operations': [0, 2],
                'write_operations': [0, 1]
                }
        phases_features = ComplexDecomposer.get_phases_features(representation)
        expected_phases_features = [{'job_id': 'unknown', 'nodes': 1, 'read_volume': 0,
                                        'write_volume': 0, 'read_io_pattern': 'uncl', 'write_io_pattern': 'uncl', 'read_io_size': 0, 'write_io_size': 0, 'ioi_bw': 0},
                                    {'job_id': 'unknown', 'nodes': 1, 'read_volume': 50, 'write_volume': 50, 'read_io_pattern': 'str', 'write_io_pattern': 'str', 'read_io_size': 25.0, 'write_io_size': 50.0, 'ioi_bw': 3.0}]
        self.assertCountEqual(phases_features, expected_phases_features)

    @patch.object(ComplexDecomposer, 'get_job_node_count')
    @patch.object(Configuration, 'get_kc_token')
    @patch.object(ComplexDecomposer, 'get_job_timeseries')
    def test_job_phases_read_write_uncorrelated(self, mock_get_timeseries, mock_get_kc_token, mock_get_node_count):
        """Test if JobDecomposer initializes well from dumped files containing job timeseries."""
        # mock the method to return some dataset file content
        timeseries = {
            'volume': {
                'timestamp': np.array([0, 1, 2, 3], dtype=int),
                'bytesRead': np.array([0, 10, 0, 0], dtype=int),
                'bytesWritten': np.array([ 0, 0, 40, 0], dtype=int)},
            'operationsCount': {
                'timestamp': np.array([0, 1, 2, 3], dtype=int),
                'operationRead': np.array([0, 2], dtype=int),
                'operationWrite': np.array([0, 1], dtype=int)},
            'accessPattern': {
                'timestamp': np.array([0, 1, 2, 3], dtype=int),
                'accessRandRead': np.array([0, 12, 0, 0], dtype=int),
                'accessSeqRead': np.array([5, 13, 0, 0], dtype=int),
                'accessStrRead': np.array([0, 0, 0, 0], dtype=int),
                'accessUnclRead': np.array([0, 1, 0, 0], dtype=int),
                'accessRandWrite': np.array([0, 0, 0, 0], dtype=int),
                'accessSeqWrite': np.array([0, 0, 0, 0], dtype=int),
                'accessStrWrite': np.array([0, 0, 10, 0], dtype=int),
                'accessUnclWrite': np.array([0, 1, 0, 0], dtype=int)
            }
        }
        mock_get_timeseries.return_value = timeseries
        mock_get_kc_token.return_value = 'token'
        mock_get_node_count.return_value = 1
        # init the job decomposer
        cd = ComplexDecomposer()

        representation = cd.get_job_representation()
        phases_features = ComplexDecomposer.get_phases_features(representation)
        expected_phases_features = [{'job_id': 'unknown', 'nodes': 1, 'read_volume': 0, 'write_volume': 0, 'read_io_pattern': 'uncl', 'write_io_pattern': 'uncl', 'read_io_size': 0, 'write_io_size': 0, 'ioi_bw': 0}, {'job_id': 'unknown', 'nodes': 1, 'read_volume': 10, 'write_volume': 0, 'read_io_pattern': 'seq', 'write_io_pattern': 'uncl', 'read_io_size': 5.0, 'write_io_size': 0, 'ioi_bw': 2.0}, {'job_id': 'unknown', 'nodes': 1, 'read_volume': 0, 'write_volume': 40, 'read_io_pattern': 'uncl', 'write_io_pattern': 'str', 'read_io_size': 0, 'write_io_size': 0, 'ioi_bw': 8.0}, {'job_id': 'unknown', 'nodes': 1, 'read_volume': 0, 'write_volume': 0, 'read_io_pattern': 'uncl', 'write_io_pattern': 'uncl', 'read_io_size': 0, 'write_io_size': 0, 'ioi_bw': 0}]
        self.assertCountEqual(phases_features, expected_phases_features)


    @patch.object(ComplexDecomposer, 'get_job_node_count')
    @patch.object(Configuration, 'get_kc_token')
    @patch.object(ComplexDecomposer, 'get_job_timeseries')
    def test_job_phases_read_write_correlated(self, mock_get_timeseries, mock_get_kc_token, mock_get_node_count):
        """Test if JobDecomposer initializes well from dumped files containing job timeseries."""
        # mock the method to return some dataset file content
        timeseries = {
            'volume': {
                'timestamp': np.array([0, 1, 2, 3], dtype=int),
                'bytesRead': np.array([0, 10, 50, 0], dtype=int),
                'bytesWritten': np.array([ 0, 0, 40, 0], dtype=int)},
            'operationsCount': {
                'timestamp': np.array([0, 1, 2, 3], dtype=int),
                'operationRead': np.array([0, 2], dtype=int),
                'operationWrite': np.array([0, 1], dtype=int)},
            'accessPattern': {
                'timestamp': np.array([0, 1, 2, 3], dtype=int),
                'accessRandRead': np.array([0, 12, 0, 0], dtype=int),
                'accessSeqRead': np.array([5, 13, 0, 0], dtype=int),
                'accessStrRead': np.array([0, 0, 0, 0], dtype=int),
                'accessUnclRead': np.array([0, 1, 0, 0], dtype=int),
                'accessRandWrite': np.array([0, 0, 0, 0], dtype=int),
                'accessSeqWrite': np.array([0, 0, 0, 0], dtype=int),
                'accessStrWrite': np.array([0, 0, 10, 0], dtype=int),
                'accessUnclWrite': np.array([0, 1, 0, 0], dtype=int)
            }
        }
        mock_get_timeseries.return_value = timeseries
        mock_get_kc_token.return_value = 'token'
        mock_get_node_count.return_value = 1
        # init the job decomposer
        cd = ComplexDecomposer()

        representation = cd.get_job_representation(merge_clusters=True)
        phases_features = ComplexDecomposer.get_phases_features(representation)
        print(phases_features)
        expected_phases_features = [{'job_id': 'unknown', 'nodes': 1, 'read_volume': 0,
                                     'write_volume': 0, 'read_io_pattern': 'uncl', 'write_io_pattern': 'uncl', 'read_io_size': 0, 'write_io_size': 0, 'ioi_bw': 0.0}, {'job_id': 'unknown', 'nodes': 1, 'read_volume': 60, 'write_volume': 40, 'read_io_pattern': 'seq', 'write_io_pattern': 'str', 'read_io_size': 30.0, 'write_io_size': 0, 'ioi_bw': 14.0}, {'job_id': 'unknown', 'nodes': 1, 'read_volume': 0, 'write_volume': 0, 'read_io_pattern': 'uncl', 'write_io_pattern': 'uncl', 'read_io_size': 0, 'write_io_size': 0, 'ioi_bw': 0.0}]
        self.assertCountEqual(phases_features, expected_phases_features)
