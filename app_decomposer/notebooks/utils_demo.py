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
    plt.rcParams["figure.figsize"] = (20, 5)
    # plt.rcParams['figure.facecolor'] = 'gray'

    plt.plot(x, read_signal, label="read signal")
    plt.plot(x, write_signal, label="write signal")
    plt.grid(True)
    plt.legend()
    plt.title(f"timeserie for jobid = {jobid}")
    plt.show()

def plot_detected_phases(jobid, merge=False):
    timestamps, read_signal, write_signal = get_job_timeseries_from_file(job_id=jobid) #457344
    ab = np.arange(len(read_signal))
    read_dec = KmeansSignalDecomposer(read_signal, merge=merge)
    read_bkps, read_labels = read_dec.decompose()
    rec_signal = read_dec.reconstruct(read_bkps)
    write_dec = KmeansSignalDecomposer(write_signal, merge=merge)
    write_bkps, write_labels = write_dec.decompose()
    rec_wsignal = write_dec.reconstruct(write_bkps)
    compute, volume, bandwidth = get_signal_representation(timestamps, read_signal, read_labels)
    w_compute, w_volume, w_bandwidth = get_signal_representation(timestamps, read_signal, read_labels)


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ab.flatten(), y=read_signal.flatten(),
                            mode='lines+markers', name='original bytesRead signal from IOI', line_width=1.5,
                            text=list(map(lambda x: "class="+str(x), read_labels))))
    fig.add_trace(go.Scatter(x=ab.flatten(), y=write_signal.flatten(),
                            mode='lines+markers', name='original bytesWritten signal from IOI', line_width=1.5,
                            text=list(map(lambda x: "class="+str(x), read_labels))))
    fig.add_trace(go.Scatter(x=ab.flatten(), y=rec_signal.flatten(), mode='lines', name='read signal slices with constant bw', line_width=2))
    fig.add_trace(go.Scatter(x=ab.flatten(), y=rec_wsignal.flatten(), mode='lines', name='write signal slices with constant bw', line_width=2))
    # fig.add_trace(go.Scatter(x=ab.flatten(), y=rec_signal.flatten(), mode='lines', name='phase model preview for execution simulation', line_shape='vh', line_width=1))#line_dash='longdash'))
    fig.update_layout(
        title=  "IOI Signal decomposition",
        legend=dict(orientation="h", yanchor="top"),
        #xaxis_title="timestamps",
        yaxis_title="volume conveyed by the application",
        legend_title="Signals",
        )
    return fig


def simulate_app(compute, reads, writes, io_bw, app_name=""):
    # Set Simulation Env
    env = simpy.Environment()
    nvram_bandwidth = {'read':  {'seq': 780, 'rand': 760},
                        'write': {'seq': 515, 'rand': 505}}
    ssd_bandwidth = {'read':  {'seq': 1, 'rand': 1},
                        'write': {'seq': 1, 'rand': 1}}

    ssd_tier = Tier(env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
    nvram_tier = Tier(env, 'NVRAM', bandwidth=nvram_bandwidth, capacity=80e9)

    # Run the simulation with computed app representation
    data = simpy.Store(env)
    cluster = Cluster(env,  compute_nodes=1, cores_per_node=2,
                        tiers=[ssd_tier, nvram_tier])
    app = Application(env, name=f"job#{app_name}",
                        compute=compute,
                        read=reads,
                        write=writes,
                        bw=io_bw,
                        data=data)
    env.process(app.run(cluster, placement=[0]*(10*len(compute))))
    env.run()
    # Extract app execution signals
    output = get_execution_signal_2(data)
    time  = output[app.name]["time"]
    read_bw = output[app.name]["read_bw"]
    write_bw = output[app.name]["write_bw"]
    return time, read_bw, write_bw
