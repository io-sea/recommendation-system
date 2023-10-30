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

from cluster_simulator.analytics import get_execution_signal, get_execution_signal_2, display_original_sim_signals, get_execution_signal_3, plot_simple_signal

from app_decomposer.utils import convert_size
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
    # return , df_clean[["bytesRead"]].to_numpy(), df_clean[["bytesWritten"]].to_numpy()


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


def plot_signal(x, read_signal, write_signal):
    min_shape = min(x.size, read_signal.size, write_signal.size)
    plt.plot(x[:min_shape], read_signal[:min_shape], label="read signal")
    plt.plot(x[:min_shape], write_signal[:min_shape], label="write signal")
    plt.grid(True)
    plt.legend()
    plt.title("timeseries for job signals")
    return plt


class TestJobDecomposerFeatures(unittest.TestCase):
    """Examine and qualify JobDecomposer output phases features and signal representation mixes read and write values within the same phase."""
    """Examine and qualify JobDecomposer on 1D signals."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.env = simpy.Environment()
        nvram_bandwidth = {'read':  {'seq': 780, 'rand': 760},
                           'write': {'seq': 515, 'rand': 505}}
        ssd_bandwidth = {'read':  {'seq': 1, 'rand': 1},
                         'write': {'seq': 1, 'rand': 1}}

        self.ssd_tier = Tier(self.env, 'SSD', max_bandwidth=ssd_bandwidth, capacity=200e9)
        self.nvram_tier = Tier(self.env, 'NVRAM', max_bandwidth=nvram_bandwidth, capacity=80e9)

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
                'bytesWritten': np.array([0, 0, 40, 0], dtype=int)},
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
        expected_phases_features = [{'job_id': 'unknown', 'nodes': 1, 'read_volume': 0, 'write_volume': 0, 'read_io_pattern': 'uncl', 'write_io_pattern': 'uncl', 'read_io_size': 0, 'write_io_size': 0, 'ioi_bw': 0}, {'job_id': 'unknown', 'nodes': 1, 'read_volume': 10, 'write_volume': 0, 'read_io_pattern': 'seq', 'write_io_pattern': 'uncl', 'read_io_size': 5.0, 'write_io_size': 0, 'ioi_bw': 2.0}, {
            'job_id': 'unknown', 'nodes': 1, 'read_volume': 0, 'write_volume': 40, 'read_io_pattern': 'uncl', 'write_io_pattern': 'str', 'read_io_size': 0, 'write_io_size': 0, 'ioi_bw': 8.0}, {'job_id': 'unknown', 'nodes': 1, 'read_volume': 0, 'write_volume': 0, 'read_io_pattern': 'uncl', 'write_io_pattern': 'uncl', 'read_io_size': 0, 'write_io_size': 0, 'ioi_bw': 0}]
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
                'bytesWritten': np.array([0, 0, 40, 0], dtype=int)},
            'operationsCount': {
                'timestamp': np.array([0, 1, 2, 3], dtype=int),
                'operationRead': np.array([0, 2, 5, 0], dtype=int),
                'operationWrite': np.array([0, 0, 1, 0], dtype=int)},
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
                                     'write_volume': 0, 'read_io_pattern': 'uncl', 'write_io_pattern': 'uncl', 'read_io_size': 0, 'write_io_size': 0, 'ioi_bw': 0.0},
                                    {'job_id': 'unknown', 'nodes': 1, 'read_volume': 60, 'write_volume': 40, 'read_io_pattern': 'seq',
                                        'write_io_pattern': 'str', 'read_io_size': 60/7, 'write_io_size': 40.0, 'ioi_bw': 14.0},
                                    {'job_id': 'unknown', 'nodes': 1, 'read_volume': 0, 'write_volume': 0, 'read_io_pattern': 'uncl', 'write_io_pattern': 'uncl', 'read_io_size': 0, 'write_io_size': 0, 'ioi_bw': 0.0}]
        self.assertCountEqual(phases_features, expected_phases_features)

    @patch.object(ComplexDecomposer, 'get_job_node_count')
    @patch.object(Configuration, 'get_kc_token')
    @patch.object(ComplexDecomposer, 'get_job_timeseries')
    def test_job_phases_read_write_mixed(self, mock_get_timeseries, mock_get_kc_token, mock_get_node_count):
        """Test if JobDecomposer initializes well from dumped files containing job timeseries."""
        # mock the method to return some dataset file content
        jobid = 3918
        timeseries = get_job_timeseries_from_file(job_id=jobid)
        mock_get_timeseries.return_value = timeseries
        mock_get_kc_token.return_value = 'token'
        mock_get_node_count.return_value = 1
        # init the job decomposer
        cd = ComplexDecomposer()

        representation = cd.get_job_representation(merge_clusters=True)
        compute, reads, read_bw, writes, write_bw = representation["events"], representation["read_volumes"], representation["read_bw"], representation["write_volumes"], representation["write_bw"]
        # This is the app encoding representation for Execution Simulator
        print(f"compute={compute}, reads={list(map(convert_size, reads))}, read_bw={list(map(convert_size, read_bw))}")
        print(f"compute={compute}, writes={list(map(convert_size, writes))}, write_bw={list(map(convert_size, write_bw))}")

        print(representation)

        # Original signal
        timestamps = (cd.timestamps.flatten() - cd.timestamps.flatten()[0])/5
        read_signal = cd.read_signal.flatten()/1e6
        write_signal = cd.write_signal.flatten()/1e6

        plt.plot(timestamps, read_signal, label="read signal")
        plt.plot(timestamps, write_signal, label="write signal")
        plt.grid(True)
        plt.legend()
        plt.title(f"original timeseries for jobid = {jobid}")
        plt.show()

        # Simulated signal

        read_bw_scaled = list(map(lambda x: x/1e6, read_bw))
        write_bw_scaled = list(map(lambda x: x/1e6, write_bw))

        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app = Application(self.env, name=f"job#{jobid}",
                          compute=compute,
                          read=reads,
                          write=writes,
                          data=data,
                          read_bw=read_bw_scaled,
                          write_bw=write_bw_scaled)
        self.env.process(app.run(cluster, placement=[0]*(10*len(compute))))
        self.env.run()
        # Extract app execution signals
        output = get_execution_signal_3(data)
        time = output[app.name]["time"]
        sim_read_bw = output[app.name]["read_bw"]
        sim_write_bw = output[app.name]["write_bw"]
        min_length = min(len(timestamps), len(sim_read_bw), len(sim_write_bw))
        timestamps = timestamps[:min_length]
        sim_read_bw = sim_read_bw[:min_length]
        sim_write_bw = sim_write_bw[:min_length]

        plt.plot(timestamps, sim_read_bw, label="read signal")
        plt.plot(timestamps, sim_write_bw, label="write signal")
        plt.grid(True)
        plt.legend()
        plt.title(f"Simulated timeseries for jobid = {jobid}")
        plt.show()

    @patch.object(ComplexDecomposer, 'get_job_node_count')
    @patch.object(Configuration, 'get_kc_token')
    @patch.object(ComplexDecomposer, 'get_job_timeseries')
    def test_job_phases_read_write_mixed_2(self, mock_get_timeseries, mock_get_kc_token, mock_get_node_count):
        """Test if JobDecomposer initializes well from dumped files containing job timeseries."""
        # mock the method to return some dataset file content
        jobid = 3917
        timeseries = get_job_timeseries_from_file(job_id=jobid)
        mock_get_timeseries.return_value = timeseries
        mock_get_kc_token.return_value = 'token'
        mock_get_node_count.return_value = 1
        # init the job decomposer
        cd = ComplexDecomposer()

        representation = cd.get_job_representation(merge_clusters=True)
        compute, reads, read_bw, writes, write_bw = representation["events"], representation["read_volumes"], representation["read_bw"], representation["write_volumes"], representation["write_bw"]
        # This is the app encoding representation for Execution Simulator
        print(f"compute={compute}, reads={reads}, read_bw={read_bw}")
        print(f"compute={compute}, writes={writes}, write_bw={write_bw}")

        print(representation)

        # Original signal
        timestamps = (cd.timestamps.flatten() - cd.timestamps.flatten()[0])/5
        read_signal = cd.read_signal.flatten()/1e6
        write_signal = cd.write_signal.flatten()/1e6

        plt.plot(timestamps, read_signal, label="read signal")
        plt.plot(timestamps, write_signal, label="write signal")
        plt.grid(True)
        plt.legend()
        plt.title(f"original timeseries for jobid = {jobid}")
        plt.show()

        # Simulated signal

        read_bw_scaled = list(map(lambda x: x/1e6, read_bw))
        write_bw_scaled = list(map(lambda x: x/1e6, write_bw))

        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app = Application(self.env, name=f"job#{jobid}",
                          compute=compute,
                          read=reads,
                          write=writes,
                          data=data,
                          read_bw=read_bw_scaled,
                          write_bw=write_bw_scaled)
        self.env.process(app.run(cluster, placement=[0]*(10*len(compute))))
        self.env.run()
        # Extract app execution signals
        output = get_execution_signal_3(data)
        time = output[app.name]["time"]
        sim_read_bw = output[app.name]["read_bw"]
        sim_write_bw = output[app.name]["write_bw"]
        min_length = min(len(timestamps), len(sim_read_bw), len(sim_write_bw))
        timestamps = timestamps[:min_length]
        sim_read_bw = sim_read_bw[:min_length]
        sim_write_bw = sim_write_bw[:min_length]

        plt.plot(timestamps, sim_read_bw, label="read signal")
        plt.plot(timestamps, sim_write_bw, label="write signal")
        plt.grid(True)
        plt.legend()
        plt.title(f"Simulated timeseries for jobid = {jobid}")
        plt.show()
