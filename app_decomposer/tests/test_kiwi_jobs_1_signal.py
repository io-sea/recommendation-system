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
    return df_clean[["timestamp"]].to_numpy(), df_clean[["bytesRead"]].to_numpy(), df_clean[["bytesWritten"]].to_numpy()

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

class QualifyComplexDecomposerOnSyntheticSignals(unittest.TestCase):
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
    def test_job_read_1(self, mock_get_timeseries, mock_get_kc_token):
        """Test if JobDecomposer initializes well from dumped files containing job timeseries."""
        # mock the method to return some manual content
        timestamps = np.arange(4)
        read_signal = np.array([1e6, 0, 0, 0])
        write_signal = np.array([0, 0, 0, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        mock_get_kc_token.return_value = 'token'
        # init the job decomposer
        cd = ComplexDecomposer()
        compute, reads, read_bw, writes, write_bw = cd.get_job_representation()

        print(f"representation for read_bw={read_bw}")
        input_bw = list(map(lambda x: x/1e6, read_bw))
        print(f"to be injected in Sim read_bw={input_bw}")
        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app = Application(self.env, name="job#3912",
                          compute=compute,
                           read=reads,
                           write=writes,
                           bw=input_bw,
                           data=data)
        self.env.process(app.run(cluster, placement=[0]*(10*len(compute))))
        self.env.run()

        print("Variables provided to the Sim")
        print(f"compute={compute}, reads={reads}, read_bw={read_bw}")
        print(f"compute={compute}, writes={writes}, write_bw={write_bw}")

        # Extract app execution signals
        print("original signals")
        print(f"timestamps={timestamps}, read_signal={read_signal}, write_signal={write_signal}")
        output = get_execution_signal_2(data)
        print("Signals extracted from the Sim")
        print(f"timestamps={output[app.name]['time']}, "
              f"read_signal={output[app.name]['read_bw']}, "
              f"write_signal={output[app.name]['write_bw']}")


        #self.assertListEqual(output[app.name]["time"], [0, 1, 2, 3])
        #self.assertListEqual(output[app.name]["read_bw"], [1, 0, 0, 0])


        if SHOW_FIGURE:
            fig1 = plt.figure("Throughput data")
            plt.plot(timestamps, read_signal/1e6, marker='o', label="read signal from IOI")
            plt.plot(output[app.name]['time'], output[app.name]['read_bw'], marker='o',
                     label="read signal from ExecSim")
            plt.grid(True)
            plt.legend()
            plt.show()

    @patch.object(Configuration, 'get_kc_token')
    @patch.object(ComplexDecomposer, 'get_job_timeseries')
    def test_job_read_2(self, mock_get_timeseries, mock_get_kc_token):
        """Test if JobDecomposer initializes well from dumped files containing job timeseries."""
        # mock the method to return some manual content
        timestamps = np.arange(4)
        read_signal = np.array([0, 1e6, 0, 0])
        write_signal = np.array([0, 0, 0, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        mock_get_kc_token.return_value = 'token'
        # init the job decomposer
        cd = ComplexDecomposer()
        compute, reads, read_bw, writes, write_bw = cd.get_job_representation()

        print(f"representation for read_bw={read_bw}")
        input_bw = list(map(lambda x: x/1e6, read_bw))
        print(f"to be injected in Sim read_bw={input_bw}")
        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app = Application(self.env, name="job#3912",
                          compute=compute,
                           read=reads,
                           write=writes,
                           bw=input_bw,
                           data=data)
        self.env.process(app.run(cluster, placement=[0]*(10*len(compute))))
        self.env.run()

        print("Variables provided to the Sim")
        print(f"compute={compute}, reads={reads}, read_bw={read_bw}")
        print(f"compute={compute}, writes={writes}, write_bw={write_bw}")

        # Extract app execution signals
        print("original signals")
        print(f"timestamps={timestamps}, read_signal={read_signal}, write_signal={write_signal}")
        output = get_execution_signal_2(data)
        print("Signals extracted from the Sim")
        print(f"timestamps={output[app.name]['time']}, "
              f"read_signal={output[app.name]['read_bw']}, "
              f"write_signal={output[app.name]['write_bw']}")


        self.assertListEqual(output[app.name]["time"], [0, 1, 2, 3])
        self.assertListEqual(output[app.name]["read_bw"], [0, 1, 0, 0])


        if SHOW_FIGURE:
            fig1 = plt.figure("Throughput data")
            plt.plot(timestamps, read_signal/1e6, marker='o', label="read signal from IOI")
            plt.plot(output[app.name]['time'], output[app.name]['read_bw'], marker='o',
                     label="read signal from ExecSim")
            plt.grid(True)
            plt.legend()
            plt.show()


    @patch.object(Configuration, 'get_kc_token')
    @patch.object(ComplexDecomposer, 'get_job_timeseries')
    def test_job_long_read(self, mock_get_timeseries, mock_get_kc_token):
        """Test if JobDecomposer initializes well from dumped files containing job timeseries."""
        # mock the method to return some manual content
        timestamps = np.arange(8)
        read_signal = np.array([1e6, 1e6, 1e6, 0, 5e6, 5e6, 5e6, 0])
        write_signal = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        mock_get_kc_token.return_value = 'token'
        # init the job decomposer
        cd = ComplexDecomposer()
        compute, reads, read_bw, writes, write_bw = cd.get_job_representation()
        print(f"representation for read_bw={read_bw}")
        input_bw = list(map(lambda x: x/1e6, read_bw))
        print(f"to be injected in Sim read_bw={input_bw}")
        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app = Application(self.env, name="job#3912",
                          compute=compute,
                           read=reads,
                           write=writes,
                           bw=input_bw,
                           data=data)
        self.env.process(app.run(cluster, placement=[0]*(10*len(compute))))
        self.env.run()

        print("Variables provided to the Sim")
        print(f"compute={compute}, reads={reads}, read_bw={read_bw}")
        print(f"compute={compute}, writes={writes}, write_bw={write_bw}")

        # Extract app execution signals
        print("original signals")
        print(f"timestamps={timestamps}, read_signal={read_signal}, write_signal={write_signal}")
        output = get_execution_signal_2(data)
        print("Signals extracted from the Sim")
        print(f"timestamps={output[app.name]['time']}, "
              f"read_signal={output[app.name]['read_bw']}, "
              f"write_signal={output[app.name]['write_bw']}")

        if SHOW_FIGURE:


            fig2 = plt.figure("Throughput data step")
            plt.plot(timestamps, read_signal/1e6, marker='o', label="read signal from IOI")
            plt.plot(output[app.name]['time'], output[app.name]['read_bw'], marker='o',
                     label="read signal from ExecSim")
            plt.grid(True)
            plt.legend()
            plt.show()

        self.assertListEqual(output[app.name]["time"], [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertListEqual(output[app.name]["read_bw"], [1.0, 1.0, 1.0, 0, 5.0, 5.0, 5.0, 0])
        self.assertListEqual(output[app.name]["write_bw"], [0, 0, 0, 0, 0, 0, 0, 0])



    @patch.object(Configuration, 'get_kc_token')
    @patch.object(ComplexDecomposer, 'get_job_timeseries')
    def test_job_long_read_with_merge(self, mock_get_timeseries, mock_get_kc_token):
        """Test if JobDecomposer initializes well from dumped files containing job timeseries."""
        # mock the method to return some manual content
        timestamps = np.arange(8)
        read_signal = np.array([1e6, 2e6, 1e6, 0, 5e6, 6e6, 5e6, 0])
        write_signal = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        mock_get_timeseries.return_value = timestamps, read_signal, write_signal
        mock_get_kc_token.return_value = 'token'
        # init the job decomposer
        cd = ComplexDecomposer()
        compute, reads, read_bw, writes, write_bw = cd.get_job_representation()
        print(f"representation for read_bw={read_bw}")
        input_bw = list(map(lambda x: x/1e6, read_bw))
        print(f"to be injected in Sim read_bw={input_bw}")
        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app = Application(self.env, name="job#3912",
                          compute=compute,
                           read=reads,
                           write=writes,
                           bw=input_bw,
                           data=data)
        self.env.process(app.run(cluster, placement=[0]*(10*len(compute))))
        self.env.run()

        print("Variables provided to the Sim")
        print(f"compute={compute}, reads={reads}, read_bw={read_bw}")
        print(f"compute={compute}, writes={writes}, write_bw={write_bw}")

        # Extract app execution signals
        print("original signals")
        print(f"timestamps={timestamps}, read_signal={read_signal}, write_signal={write_signal}")
        output = get_execution_signal_2(data)
        print("Signals extracted from the Sim")
        print(f"timestamps={output[app.name]['time']}, "
              f"read_signal={output[app.name]['read_bw']}, "
              f"write_signal={output[app.name]['write_bw']}")

        SHOW_FIGURE = True
        if SHOW_FIGURE:


            fig1 = plt.figure("Throughput data step")
            plt.plot(timestamps, read_signal/1e6, marker='o', label="read signal from IOI")
            plt.plot(output[app.name]['time'], output[app.name]['read_bw'], marker='o',
                     label="read signal from ExecSim")
            plt.grid(True)
            plt.legend()
            plt.show()

            fig2 = plt.figure("Cumulative data")
            plt.plot(output[app.name]['time'], np.cumsum(output[app.name]['read_bw']), marker='.', label="read signal from execution")
            plt.plot(timestamps, np.cumsum(read_signal/1e6), marker='.', label="read signal")
            plt.grid(True)
            plt.legend()
            plt.show()

        third = np.float(1/3)
        self.assertListEqual(output[app.name]["time"], [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertListEqual(output[app.name]["read_bw"], [1.0+third, 1.0+third, 1.0+third, 0, 5.0+third, 5.0+third, 5.0+third, 0])
        self.assertListEqual(output[app.name]["write_bw"], [0, 0, 0, 0, 0, 0, 0, 0])



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
        jobid=3912
        SHOW_FIGURE = True
        mock_get_timeseries.return_value = get_job_timeseries_from_file(job_id=jobid)
        mock_get_kc_token.return_value = 'token'
        # init the job decomposer
        cd = ComplexDecomposer()


        compute, reads, read_bw, writes, write_bw = cd.get_job_representation()
        # This is the app encoding representation for Execution Simulator
        print(f"compute={compute}, reads={reads}, read_bw={read_bw}")
        print(f"compute={compute}, writes={writes}, write_bw={write_bw}")

        # converting bws into MB/s
        print(read_bw)
        read_bw = list(map(lambda x: x/1e6, read_bw))
        print(read_bw)

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
        # Extract app execution signals
        output = get_execution_signal_2(data)
        time = output[app.name]["time"]
        read_bw = output[app.name]["read_bw"]
        write_bw = output[app.name]["write_bw"]
        # plot_job_signal(jobid=jobid)
        timestamps = (cd.timestamps.flatten() - cd.timestamps.flatten()[0])/5
        read_signal =  cd.read_signal.flatten()/1e6
        write_signal = cd.write_signal.flatten()/1e6


        print(f" time : {min(time)} -> {max(time)}")
        print(f" read_bw : {min(read_bw)} -> {max(read_bw)} len = {len(read_bw)}")
        print(f" timestamps : {min(timestamps)} -> {max(timestamps)}")
        print(f" read_signal : {min(read_signal)} -> {max(read_signal)} len = {len(read_signal)}")
        print(f" read_signal and read_bw integ : {np.trapz(read_signal, dx=np.diff(timestamps)[0])} -> {np.trapz(read_bw, dx=np.diff(timestamps)[0])}")

        if SHOW_FIGURE:
            fig1 = plt.figure("Throughput data")
            plt.plot(time, read_bw, marker='o', label="read signal from execution")
            plt.plot(timestamps, read_signal, marker='o', label="read signal")
            plt.grid(True)
            plt.legend()
            plt.title(f"timeserie for jobid = {jobid}")

            fig2 = plt.figure("Cumulative data")
            plt.plot(time, np.cumsum(read_bw), marker='.', label="read signal from execution")
            plt.plot(timestamps, np.cumsum(read_signal), marker='.', label="read signal")
            plt.grid(True)
            plt.legend()
            plt.title(f"timeserie for jobid = {jobid}")
            plt.show()

    @patch.object(Configuration, 'get_kc_token')
    @patch.object(ComplexDecomposer, 'get_job_timeseries')
    def test_job_with_plots(self, mock_get_timeseries, mock_get_kc_token):
        """Test if JobDecomposer initializes well from dumped files containing job timeseries."""
        # mock the method to return some dataset file content
        jobid=3911
        merge_clusters = True
        mock_get_timeseries.return_value = get_job_timeseries_from_file(job_id=jobid)
        mock_get_kc_token.return_value = 'token'
        # init the job decomposer

        # decompose and app/job
        cd = ComplexDecomposer()
        compute, reads, read_bw, writes, write_bw = cd.get_job_representation(merge_clusters=merge_clusters)
        read_bw = list(map(lambda x: x/1e6, read_bw))
        write_bw = list(map(lambda x: x/1e6, write_bw))

        # Apply same BW as measured by IOI
        io_bw = list(map(lambda x, y: x + y, read_bw, write_bw))

        # Apply different BW and simulate app execution on another storage system
        # with fixed values
        # io_bw = list(map(lambda x, y: 1e4 if x else 0 + 1e3 if y else 0, read_bw, write_bw))
        # with relative values
        # io_bw = list(map(lambda x, y: x/2 + y/2, read_bw, write_bw))

        timestamps = (cd.timestamps.flatten() - cd.timestamps.flatten()[0])/5
        original_read =  cd.read_signal.flatten()/1e6
        original_write = cd.write_signal.flatten()/1e6

        # Run the simulation with computed app representation
        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app = Application(self.env, name=f"job#{jobid}",
                          compute=compute,
                           read=reads,
                           write=writes,
                           bw=io_bw,
                           data=data)
        self.env.process(app.run(cluster, placement=[0]*(10*len(compute))))
        self.env.run()
        # Extract app execution signals
        output = get_execution_signal_2(data)
        time = list(map(lambda x: x, output[app.name]["time"]))
        read_bw = output[app.name]["read_bw"]
        write_bw = output[app.name]["write_bw"]
        # plot_job_signal(jobid=jobid)


        # if SHOW_FIGURE:
        fig = display_original_sim_signals((time, read_bw, write_bw),
                                           (timestamps, original_read, original_write),
                                           width=800, height=900)
        fig.show()

