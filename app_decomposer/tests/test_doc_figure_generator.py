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

import unittest
import time
import numpy as np
import simpy

from cluster_simulator.cluster import Cluster, Tier, EphemeralTier, bandwidth_share_model, compute_share_model, get_tier, convert_size
from cluster_simulator.phase import DelayPhase, ComputePhase, IOPhase
from cluster_simulator.application import Application
from cluster_simulator.analytics import display_run, display_run_with_signal



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

class TestFigGenerator(unittest.TestCase):
    """Test that the app decomposer follows some pattern."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.env = simpy.Environment()
        nvram_bandwidth = {'read':  {'seq': 780, 'rand': 760},
                           'write': {'seq': 515, 'rand': 505}}
        ssd_bandwidth = {'read':  {'seq': 1, 'rand': 1},
                         'write': {'seq': 1, 'rand': 1}}

        self.ssd_tier = Tier(self.env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
        self.nvram_tier = Tier(self.env, 'NVRAM', bandwidth=nvram_bandwidth, capacity=80e9)

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_generate_simple_compare_app(self, mock_get_timeseries):
        """Test if JobDecomposer initializes well from dumped files containing job timeseries."""
        # mock the method to return some dataset file content
        # mock_get_timeseries.return_value = get_job_timeseries_from_file(job_id=457344)
        timestamps = np.arange(6)
        read_signal = np.array([1, 1, 0, 0, 0, 0])
        write_signal = np.array([0, 0, 0, 0, 1, 1])
        app_signal = timestamps, read_signal, write_signal
        # mock the method to return previous arrays
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        compute, reads, writes, read_bw, write_bw = jd.get_job_representation(merge_clusters=True)
        # This is the app encoding representation for Execution Simulator
        print(f"compute={compute}, reads={reads}, read_bw={read_bw}")
        print(f"compute={compute}, writes={writes}, write_bw={write_bw}")

        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app = Application(self.env, name="#read-compute-write", compute=compute,
                           read=list(map(lambda x:x*1e6, reads)),
                           write=list(map(lambda x:x*1e6, writes)), data=data)
        self.env.process(app.run(cluster, placement=[0, 0, 0, 0, 0, 0]))

        self.env.run()
        #fig = display_run_with_signal(data, cluster, app_signal=app_signal, width=800, height=900)
        fig = display_run(data, cluster, width=800, height=900)
        fig.show()
        current_dir = dirname(dirname(dirname(abspath(__file__))))
        file_path = os.path.join(current_dir, "app_decomposer", "docs", "figure_synthetic_signal.html")
        fig.write_html(file_path)


    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_generate_simple_compare_app_0(self, mock_get_timeseries):
        """Test if JobDecomposer initializes well from dumped files containing job timeseries."""
        # mock the method to return some dataset file content
        # mock_get_timeseries.return_value = get_job_timeseries_from_file(job_id=457344)
        timestamps = np.arange(7)
        read_signal = np.array([0, 1, 1, 0, 0, 0, 0])
        write_signal = np.array([0, 0, 0, 0, 1, 1, 0])
        app_signal = timestamps, read_signal, write_signal
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        compute, reads, writes, read_bw, write_bw = jd.get_job_representation(merge_clusters=True)
        # This is the app encoding representation for Execution Simulator
        print(f"compute={compute}, reads={reads}, read_bw={read_bw}")
        print(f"compute={compute}, writes={writes}, write_bw={write_bw}")

        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app = Application(self.env, name="#read-compute-write", compute=compute,
                           read=list(map(lambda x:x*1e6, reads)),
                           write=list(map(lambda x:x*1e6, writes)), data=data)
        self.env.process(app.run(cluster, placement=[0, 0, 0, 0, 0, 0]))

        self.env.run()
        # fig = display_run_with_signal(data, cluster, app_signal=app_signal, width=800, height=900)
        # fig.show()

        #print(f"compute={events}, read={reads}, writes={writes}")

    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_generate_simple_compare_app_1(self, mock_get_timeseries):
        """Test if JobDecomposer initializes well from dumped files containing job timeseries."""
        # mock the method to return some dataset file content
        # mock_get_timeseries.return_value = get_job_timeseries_from_file(job_id=457344)
        timestamps = np.linspace(0, 100, 101)
        read_signal = np.where(abs(timestamps-10)<=5, 1, 0)
        write_signal = np.where(abs(timestamps-60)<=5, 1, 0)
        app_signal = timestamps, read_signal, write_signal
        # mock the method to return previous arrays
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        # init the job decomposer
        jd = JobDecomposer()
        compute, reads, writes, read_bw, write_bw = jd.get_job_representation(merge_clusters=True)
        # This is the app encoding representation for Execution Simulator
        print(f"read signal={read_signal}")
        print(f"write signal={write_signal}")
        print(f"compute={compute}, reads={reads}, read_bw={read_bw}")
        print(f"compute={compute}, writes={writes}, write_bw={write_bw}")

        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app = Application(self.env, name="#read-compute-write", compute=compute,
                           read=list(map(lambda x:x*1e6, reads)),
                           write=list(map(lambda x:x*1e6, writes)), data=data)
        self.env.process(app.run(cluster, placement=[0, 0, 0, 0, 0, 0]))

        self.env.run()
        # fig = display_run_with_signal(data, cluster, app_signal=app_signal, width=800, height=900)
        # fig.show()


    def test_generate_simple_compare_app_2(self):
        """Test if JobDecomposer initializes well from dumped files containing job timeseries."""
        # readlabels=[1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0]
        # [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1  0 0 0 0 0 0]
        # writlabels=[0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 0]

        # read:
        #     events: [0, 15, 17, 32, 42, 48]
        #     volume: [3,  2,  3,  4,  2,  0]

        # write:
        #     events: [0, 1, 7, 10, 13, 14]
        #     volume: [0,12, 17,11,  7,  0]

        # compute =[0, 1 , 18, 19, 21, 39, 53, 59]
        # read    =[3, 0 , 2,   0,  3,  4, 2 , 0 ]
        # write   =[0, 12, 0,  17,  0, 11, 7,  0 ]

        # results:
        compute = [0, 1, 8, 12, 16, 17, 18, 34, 45, 51]
        read = [3, 0, 0, 0, 2, 0, 3, 4, 2, 0]
        write = [0, 12, 17, 11, 7, 0, 0, 0, 0, 0]

        # mock_get_timeseries.return_value = get_job_timeseries_from_file(job_id=2537) #457344

        # # init the job decomposer
        # jd = JobDecomposer()
        # app_signal = jd.timestamps.flatten(), jd.read_signal.flatten(), jd.write_signal.flatten()
        # compute, reads, writes, read_bw, write_bw = jd.get_job_representation(merge_clusters=True)
        # This is the app encoding representation for Execution Simulator
        # for sig in app_signal:
        #     print(f"min = {min(sig)}")
        #     print(f"max = {max(sig)}")
        #     print(f"mean = {np.mean(sig)}")
        # print(f"compute={compute}, reads={reads}, read_bw={read_bw}")
        # print(f"compute={compute}, writes={writes}, write_bw={write_bw}")

        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app = Application(self.env, name="#read-compute-write", compute=compute,
                           read=list(map(lambda x:x*1e6, read)),
                           write=list(map(lambda x:x*1e6, write)), data=data)
        self.env.process(app.run(cluster, placement=[0]*len(read)))

        self.env.run()
        # fig = display_run(data, cluster, width=800, height=900)
        # fig.show()


