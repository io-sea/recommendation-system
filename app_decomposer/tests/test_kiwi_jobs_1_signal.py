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

from cluster_simulator.cluster import Cluster, Tier, EphemeralTier, bandwidth_share_model, compute_share_model, get_tier, convert_size
from cluster_simulator.phase import DelayPhase, ComputePhase, IOPhase
from cluster_simulator.application import Application
from cluster_simulator.analytics import display_run, display_run_with_signal


from app_decomposer.job_decomposer import JobDecomposer, ComplexDecomposer, get_events_indexes, get_signal_representation, get_phase_volume, phases_to_representation, complex_to_representation, get_events_indexes, get_events_indexes_no_merge, get_phase_volume, is_interval_in

from app_decomposer.config_parser import Configuration
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


class QualifyJobDecomposer1Signal(unittest.TestCase):
    """Examine and qualify JobDecomposer on 1D signals."""
    def setUp(self):
        """Set up test fixtures, if any."""
        self.env = simpy.Environment()
        nvram_bandwidth = {'read':  {'seq': 780, 'rand': 760},
                           'write': {'seq': 515, 'rand': 505}}
        ssd_bandwidth = {'read':  {'seq': 1, 'rand': 1},
                         'write': {'seq': 1, 'rand': 1}}

        self.ssd_tier = Tier(self.env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
        self.nvram_tier = Tier(self.env, 'NVRAM', bandwidth=nvram_bandwidth, capacity=80e9)

    @patch.object(Configuration, 'get_kc_token')
    @patch.object(ComplexDecomposer, 'get_job_timeseries')
    def test_job_3912(self, mock_get_timeseries, mock_get_kc_token):
        """Test if JobDecomposer initializes well from dumped files containing job timeseries."""
        # mock the method to return some dataset file content
        mock_get_timeseries.return_value = get_job_timeseries_from_file(job_id=3912)
        mock_get_kc_token.return_value = 'token'
        # init the job decomposer
        cd = ComplexDecomposer()
        compute, read_volumes, read_bw, write_volumes, write_bw = cd.get_job_representation()
        print(compute)
        print(read_volumes)
        print(read_bw)
        print(write_volumes)
        print(write_bw)
        print("--------")
        print(np.sum(cd.read_signal.flatten()))
        print(sum(read_volumes))
        print(cd.read_signal.flatten().tolist())
        # print(np.sum(cd.write_signal.flatten()))
        # print(sum(write_volumes))
        # print(cd.write_signal.flatten().tolist())
        compute, reads, writes, read_bw, write_bw = cd.get_job_representation()
        # This is the app encoding representation for Execution Simulator
        print(f"compute={compute}, reads={reads}, read_bw={read_bw}")
        print(f"compute={compute}, writes={writes}, write_bw={write_bw}")

        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app = Application(self.env, name="job#3912",
                          compute=compute,
                           read=reads,
                           write=writes,
                           bw=read_bw,
                           data=data)
        self.env.process(app.run(cluster, placement=[0]*(10*len(compute))))
        self.env.run()
        #fig = display_run_with_signal(data, cluster, app_signal=app_signal, width=800, height=900)
        fig = display_run(data, cluster, width=800, height=900)
        fig.show()

        # data = simpy.Store(self.env)
        # cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
        #                   tiers=[self.ssd_tier, self.nvram_tier])
        # app1 = Application(self.env, name="#read2G->Comp2s", compute=[0, 2],
        #                    read=[1e9, 0], write=[0, 0], bw=[50, 50], data=data)
        # app2 = Application(self.env, name="#comp1s->write2G", compute=[0, 1],
        #                    read=[0, 0], write=[0, 2e9], bw=[50, 50], data=data)
        # self.env.process(app2.run(cluster, placement=[0, 0]))
        # self.env.process(app1.run(cluster, placement=[0, 0]))
