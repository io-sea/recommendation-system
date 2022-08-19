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

def get_job_timeseries_from_file(job_id=None, df=None):
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

    def test_generate_figure_show_decomposition_read_no_merge(self):
        """Generates figures to show how AppDecomposer slices the signal into representation."""
        timestamps, read_signal, write_signal = get_job_timeseries_from_file(job_id=2537) #457344
        ab = np.arange(len(read_signal))
        read_dec = KmeansSignalDecomposer(read_signal, v0_threshold=0.05)
        read_bkps, read_labels = read_dec.decompose()
        rec_signal = read_dec.reconstruct(read_bkps)
        compute, volume, bandwidth = get_signal_representation(timestamps, read_signal, read_labels)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ab.flatten(), y=read_signal.flatten(),
                                 mode='lines+markers', name='original bytesRead signal from IOI', line_width=1.5,
                                 text=list(map(lambda x: "class="+str(x), read_labels))))
        fig.add_trace(go.Scatter(x=ab.flatten(), y=rec_signal.flatten(), mode='lines', name='signal slices with constant bw', line_width=1))
        fig.add_trace(go.Scatter(x=ab.flatten(), y=rec_signal.flatten(), mode='lines', name='phase model preview for execution simulation', line_shape='vh', line_width=1))
        # , line_dash='longdash'
        fig.update_layout(
            title=  "Read signal decomposition",
            legend=dict(orientation="h", yanchor="top"),
            #xaxis_title="timestamps",
            yaxis_title="volume conveyed by the application",
            legend_title="Signals",
            )
        fig.show()
        current_dir = dirname(dirname(dirname(abspath(__file__))))
        file_path = os.path.join(current_dir, "app_decomposer", "docs", "decomposing_read_signal.html")
        fig.write_html(file_path)
        decomposition = f"""Representation: events = {compute},
                        volumes = {volume},
                        bandwidth = {bandwidth}"""
        print(decomposition)
        # fig = go.Figure()
        # x = np.arange(len(signal))
        # fig.add_trace(go.Scatter(x=x.flatten(), y=signal.flatten(), mode='lines', name='signal'))
        # fig.add_trace(go.Scatter(x=x.flatten(), y=rec_signal.flatten(), mode='lines', name='reconstructed signal'))
        # fig.show()
    def test_generate_figure_show_decomposition_read_merge(self):
        """Generates figures to show how AppDecomposer slices the signal into representation."""
        timestamps, read_signal, write_signal = get_job_timeseries_from_file(job_id=2537) #457344
        ab = np.arange(len(read_signal))
        read_dec = KmeansSignalDecomposer(read_signal, merge=True)
        read_bkps, read_labels = read_dec.decompose()
        rec_signal = read_dec.reconstruct(read_bkps)
        compute, volume, bandwidth = get_signal_representation(timestamps, read_signal, read_labels)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ab.flatten(), y=read_signal.flatten(),
                                 mode='lines+markers', name='original bytesRead signal from IOI', line_width=1.5,
                                 text=list(map(lambda x: "class="+str(x), read_labels))))
        fig.add_trace(go.Scatter(x=ab.flatten(), y=rec_signal.flatten(), mode='lines', name='signal slices with constant bw', line_width=1))
        fig.add_trace(go.Scatter(x=ab.flatten(), y=rec_signal.flatten(), mode='lines', name='phase model preview for execution simulation', line_shape='vh', line_width=1))#line_dash='longdash'))
        fig.update_layout(
            title=  "Read signal decomposition",
            legend=dict(orientation="h", yanchor="top"),
            #xaxis_title="timestamps",
            yaxis_title="volume conveyed by the application",
            legend_title="Signals",
            )
        fig.show()
        current_dir = dirname(dirname(dirname(abspath(__file__))))
        file_path = os.path.join(current_dir, "app_decomposer", "docs", "decomposing_read_signal_merge.html")
        fig.write_html(file_path)
        decomposition = f"""Representation: events = {compute},
                        volumes = {volume},
                        bandwidth = {bandwidth}"""
        print(decomposition)


    def test_generate_figure_show_decomposition_write_merge(self):
        """Generates figures to show how AppDecomposer slices the signal into representation."""
        timestamps, read_signal, write_signal = get_job_timeseries_from_file(job_id=2537) #457344
        ab = np.arange(len(write_signal))
        write_dec = KmeansSignalDecomposer(write_signal, merge=True)
        write_bkps, write_labels = write_dec.decompose()
        rec_signal = write_dec.reconstruct(write_bkps)
        compute, volume, bandwidth = get_signal_representation(timestamps, write_signal, write_labels)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ab.flatten(), y=write_signal.flatten(),
                                 mode='lines+markers', name='original bytesRead signal from IOI', line_width=1.5,
                                 text=list(map(lambda x: "class="+str(x), write_labels))))
        fig.add_trace(go.Scatter(x=ab.flatten(), y=rec_signal.flatten(), mode='lines', name='signal slices with constant bw', line_width=1))
        fig.add_trace(go.Scatter(x=ab.flatten(), y=rec_signal.flatten(), mode='lines', name='phase model preview for execution simulation', line_shape='vh', line_width=1)) #line_dash='longdash'))
        fig.update_layout(
            title=  "Write signal decomposition with merge option",
            legend=dict(orientation="h", yanchor="top"),
            #xaxis_title="timestamps",
            yaxis_title="volume conveyed by the application",
            legend_title="Signals",
            )
        fig.show()
        current_dir = dirname(dirname(dirname(abspath(__file__))))
        file_path = os.path.join(current_dir, "app_decomposer", "docs", "decomposing_write_signal_with_merge.html")
        fig.write_html(file_path)
        decomposition = f"""Representation: events = {compute},
                        volumes = {volume},
                        bandwidth = {bandwidth}"""
        print(decomposition)


    def test_generate_figure_show_decomposition_write_merge_cumvol(self):
        """Generates figures to show how AppDecomposer slices the signal into representation."""
        timestamps, read_signal, write_signal = get_job_timeseries_from_file(job_id=2537) #457344
        ab = np.arange(len(write_signal))
        write_dec = KmeansSignalDecomposer(write_signal, merge=True)
        write_bkps, write_labels = write_dec.decompose()
        rec_signal = write_dec.reconstruct(write_bkps)
        compute, volume, bandwidth = get_signal_representation(timestamps, write_signal, write_labels)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ab.flatten(), y=np.cumsum(write_signal.flatten()),
                                 mode='lines+markers', name='original bytesRead signal from IOI', line_width=1.5,
                                 text=list(map(lambda x: "class="+str(x), write_labels))))
        fig.add_trace(go.Scatter(x=ab.flatten(), y=np.cumsum(rec_signal.flatten()), mode='lines', name='signal slices with constant bw', line_width=1))
        fig.add_trace(go.Scatter(x=ab.flatten(), y=np.cumsum(rec_signal.flatten()), mode='lines', name='phase model preview for execution simulation', line_width=1)) #line_dash='longdash'))
        fig.update_layout(
            title=  "Write cumulative volumes with merge option",
            legend=dict(orientation="h", yanchor="top"),
            #xaxis_title="timestamps",
            yaxis_title="cumulative volume conveyed by the application",
            legend_title="Cumulative volumes",
            )
        fig.show()
        current_dir = dirname(dirname(dirname(abspath(__file__))))
        file_path = os.path.join(current_dir, "app_decomposer", "docs", "decomposing_cumvol_write_signal_with_merge.html")
        fig.write_html(file_path)
        decomposition = f"""Representation: events = {compute},
                        volumes = {volume},
                        bandwidth = {bandwidth}"""
        print(decomposition)
    def test_generate_figure_show_decomposition_2(self):
        """Generates figures to show how AppDecomposer slices the signal into representation."""
        timestamps, read_signal, write_signal = get_job_timeseries_from_file(job_id=2537) #457344
        ab = np.arange(len(write_signal))
        write_dec = KmeansSignalDecomposer(write_signal)
        write_bkps, write_labels = write_dec.decompose()
        rec_signal = write_dec.reconstruct(write_bkps)
        compute, volume, bandwidth = get_signal_representation(timestamps, write_signal, write_labels)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ab.flatten(), y=write_signal.flatten(),
                                 mode='lines+markers', name='original bytesRead signal from IOI', line_width=1.5,
                                 text=list(map(lambda x: "class="+str(x), write_labels))))
        fig.add_trace(go.Scatter(x=ab.flatten(), y=rec_signal.flatten(), mode='lines', name='signal slices with constant bw', line_width=1))
        fig.add_trace(go.Scatter(x=ab.flatten(), y=rec_signal.flatten(), mode='lines', name='phase model preview for execution simulation', line_shape='vh', line_width=1)) #line_dash='longdash'))
        fig.update_layout(
            title=  "Write signal decomposition",
            legend=dict(orientation="h", yanchor="top"),
            #xaxis_title="timestamps",
            yaxis_title="volume conveyed by the application",
            legend_title="Signals",
            )
        fig.show()
        current_dir = dirname(dirname(dirname(abspath(__file__))))
        file_path = os.path.join(current_dir, "app_decomposer", "docs", "decomposing_write_signal.html")
        fig.write_html(file_path)
        decomposition = f"""Representation: events = {compute},
                        volumes = {volume},
                        bandwidth = {bandwidth}"""
        print(decomposition)
        # fig = go.Figure()
        # x = np.arange(len(signal))
        # fig.add_trace(go.Scatter(x=x.flatten(), y=signal.flatten(), mode='lines', name='signal'))
        # fig.add_trace(go.Scatter(x=x.flatten(), y=rec_signal.flatten(), mode='lines', name='reconstructed signal'))
        # fig.show()

    def test_generate_figure_show_decomposition_read_large_v0(self):
        """Generates figures to show how AppDecomposer slices the signal into representation."""
        timestamps, read_signal, write_signal = get_job_timeseries_from_file(job_id=2537) #457344
        ab = np.arange(len(read_signal))
        read_dec = KmeansSignalDecomposer(read_signal, v0_threshold=0.4)
        read_bkps, read_labels = read_dec.decompose()
        rec_signal = read_dec.reconstruct(read_bkps)
        compute, volume, bandwidth = get_signal_representation(timestamps, read_signal, read_labels)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ab.flatten(), y=read_signal.flatten(),
                                 mode='lines+markers', name='original bytesRead signal from IOI', line_width=1.5,
                                 text=list(map(lambda x: "class="+str(x), read_labels))))
        fig.add_trace(go.Scatter(x=ab.flatten(), y=rec_signal.flatten(), mode='lines', name='signal slices with constant bw', line_width=1))
        fig.add_trace(go.Scatter(x=ab.flatten(), y=rec_signal.flatten(), mode='lines', name='phase model preview for execution simulation', line_shape='vh', line_width=1))
        # , line_dash='longdash'
        fig.update_layout(
            title=  "Read signal decomposition with threshold=40%",
            legend=dict(orientation="h", yanchor="top"),
            #xaxis_title="timestamps",
            yaxis_title="volume conveyed by the application",
            legend_title="Signals",
            )
        fig.show()
        current_dir = dirname(dirname(dirname(abspath(__file__))))
        file_path = os.path.join(current_dir, "app_decomposer", "docs", "decomposing_read_signal_high_threshold.html")
        fig.write_html(file_path)
        decomposition = f"""Representation: events = {compute},
                        volumes = {volume},
                        bandwidth = {bandwidth}"""
        print(decomposition)


    def test_generate_figure_show_decomposition_read_compare_low_large_v0(self):
        """Generates figures to show how AppDecomposer slices the signal into representation."""
        timestamps, read_signal, write_signal = get_job_timeseries_from_file(job_id=2537) #457344
        ab = np.arange(len(read_signal))

        high_read_dec = KmeansSignalDecomposer(read_signal, v0_threshold=0.4)
        high_read_bkps, high_read_labels = high_read_dec.decompose()
        high_rec_signal = high_read_dec.reconstruct(high_read_bkps)

        low_read_dec = KmeansSignalDecomposer(read_signal, v0_threshold=0.05)
        low_read_bkps, low_read_labels = low_read_dec.decompose()
        low_rec_signal = low_read_dec.reconstruct(low_read_bkps)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ab.flatten(), y=read_signal.flatten(),
                                 mode='markers', name='original bytesRead signal from IOI', line_width=1.5))
        fig.add_trace(go.Scatter(x=ab.flatten(), y=high_rec_signal.flatten(), mode='lines', name='signal decomposition with high cluster-0-threshold=40%', line_width=1))
        fig.add_trace(go.Scatter(x=ab.flatten(), y=low_rec_signal.flatten(), mode='lines', name='signal decomposition with low cluster-0-threshold=5%', line_width=1))
        fig.update_layout(
            title=  "Comparing signal decomposition with threshold=40% and threshold=5%",
            legend=dict(orientation="h", yanchor="top"),
            #xaxis_title="timestamps",
            yaxis_title="volume conveyed by the application",
            legend_title="Signals",
            )
        fig.show()
        current_dir = dirname(dirname(dirname(abspath(__file__))))
        file_path = os.path.join(current_dir, "app_decomposer", "docs", "compare_decomposing_signal_high_low_threshold.html")
        fig.write_html(file_path)



    @patch.object(JobDecomposer, 'get_job_timeseries')
    def test_generate_simple_compare_app_2(self, mock_get_timeseries):
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
        # compute = [0, 1, 8, 12, 16, 17, 18, 34, 45, 51]
        # read = [3, 0, 0, 0, 2, 0, 3, 4, 2, 0]
        # write = [0, 12, 17, 11, 7, 0, 0, 0, 0, 0]
        ssd_bandwidth = {'read':  {'seq': 100, 'rand': 100},
                         'write': {'seq': 60, 'rand': 60}}
        self.ssd_tier = Tier(self.env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
        mock_get_timeseries.return_value = get_job_timeseries_from_file(job_id=2537) #457344

        # init the job decomposer
        jd = JobDecomposer()
        #app_signal = jd.timestamps.flatten(), jd.read_signal.flatten(), jd.write_signal.flatten()
        compute, reads, writes, read_bw, write_bw = jd.get_job_representation(merge_clusters=True)

        # This is the app encoding representation for Execution Simulator
        # for sig in app_signal:
        #     print(f"min = {min(sig)}")
        #     print(f"max = {max(sig)}")
        #     print(f"mean = {np.mean(sig)}")
        print(f"compute={compute}, reads={reads}, read_bw={read_bw}")
        print(f"compute={compute}, writes={writes}, write_bw={write_bw}")

        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app = Application(self.env, name="#read-compute-write", compute=compute,
                           read=list(map(lambda x:x, reads)),
                           write=list(map(lambda x:x, writes)), data=data)

        self.env.process(app.run(cluster, placement=[0]*len(reads)))

        self.env.run()
        fig = display_run(data, cluster, width=800, height=900)
        fig.show()

        timestamps, reads, writes = get_job_timeseries_from_file(job_id=2537)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timestamps.flatten(), y=reads.flatten(), mode='lines', name='bytesRead'))
        fig.add_trace(go.Scatter(x=timestamps.flatten(), y=writes.flatten(), mode='lines', name='bytesWritten'))
        fig.update_layout(
            title=  "Signal sample from real application instrumented with IOI",
            legend=dict(orientation="h", yanchor="top"),
            #xaxis_title="timestamps",
            yaxis_title="volume conveyed by the application",
            legend_title="Signals",
            )
        fig.show()
        current_dir = dirname(dirname(dirname(abspath(__file__))))
        file_path = os.path.join(current_dir, "app_decomposer", "docs", "figure_timeseries_ioi_signal.html")
        fig.write_html(file_path)



