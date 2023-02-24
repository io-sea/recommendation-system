__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""
import os
from os.path import dirname
import sys
import pandas as pd
from loguru import logger
from app_decomposer.utils import convert_size
from app_decomposer import DATASET_SOURCE
from typing import List

from performance_data import DATASET_FILE
from performance_data.fakeapp_workload import FakeappWorkload as Workload

__CAPING_VOLUME__ = 10e6

class PhaseData:
    def __init__(self, phases, target, ioi=False, sample=1, lite=False):
        """Initializes the phases data with features extracted from AppDecomposer, runs the associated workload and return data.
        Args:
            phases (list): list of phases to be measured
            target (dict): storage backend file (nfs or lustre)
            ioi (bool): enable/disable IOI, default = False
            sample (int): number of sampling of the same phase
            lite (bool): if True, caps the volume to 1GB, else use the phase volume for bw measurement.
        """
        self.phases = phases
        self.target = target
        self.ioi = ioi
        self.sample = sample
        self.lite = lite

    def run_phase_workload(self, phase, target, accelerator=False):
        """Compute the average bandwidht on a storage tier with the phase features extracted from AppDecomposer
        Args:
            phase (dict): phase to be measured, keys are job_id, nodes, read_volume, write_volume,read_io_pattern, write_io_pattern, read_io_size, write_io_size, ioi_bw
            target (string): storage backend
            accelerator (string): using IO acclerator such as SBB/FIOL
            lite (bool): whether or not use lite IO capping volume to 1GB and Ops to 100.

        Return:
            avg_bw (float):  average bandwidth
        """
        #run fakeapp n times to get the avg bandwidth
        latencies = 0
        volumes = 0
        # phase_volume = max(1e9, 100*phase["IOsize"]) if self.lite and phase["volume"] > 0 else phase["volume"]
        r_volume = phase["read_volume"]
        w_volume = phase["write_volume"]
        if self.lite and (r_volume + w_volume) > __CAPING_VOLUME__:
            phase["read_volume"] = __CAPING_VOLUME__*r_volume/(r_volume + w_volume)
            phase["write_volume"] = __CAPING_VOLUME__*w_volume/(r_volume + w_volume)
            logger.info(f"Phase volume exceeds cap: {convert_size(r_volume + w_volume)} > "
                        f"{convert_size(__CAPING_VOLUME__)}| Capping to read_volume: "
                        f"{convert_size(phase['read_volume'])} | "
                        f"write_volume: {convert_size(phase['write_volume'])}")

        for _ in range(self.sample):
            # TODO : should be able to control volume of workload to adjust accuracy.
            workload = Workload(phase=phase, target_tier=target,
                                accelerator=accelerator, ioi=self.ioi)
            (latency, bw) = workload.get_data()
            latencies += latency
            volumes += phase["read_volume"] + phase["write_volume"]
        # print(f"n_samples = {self.sample}")
        avg_bw = (float)(volumes/latencies) if latencies else 0
        logger.info(f"Measured throughput on tier: {target} | use SBB: {accelerator} | result: {convert_size(avg_bw)}/s")
        return avg_bw

    def get_phase_data(self, target_names: List[str] = ["nfs_bw", "lfs_bw", "sbb_bw"]) -> pd.DataFrame:
        """
        Computes the performance of each phase on the specified tiers and returns the results as a DataFrame.

        Args:
            target_names (List[str], optional): A list of tier names to compute performance for. Default is ["nfs_bw", "lfs_bw", "sbb_bw"].

        Returns:
            A pandas DataFrame containing the performance of each phase on the specified tiers.

        Raises:
            ValueError: If any of the specified tier names is invalid.

        Example:
            # Compute performance for the "nfs_bw" and "lfs_bw" tiers and return the results as a DataFrame.
            phases_perf = PhaseData(phases, targets, ioi, sample=sample, lite=lite)
            perf_df = phases_perf.get_phase_data(target_names=["nfs_bw", "lfs_bw"])
        """
        # Validate input tier names
        valid_tiers = {"nfs_bw", "lfs_bw", "sbb_bw"}
        if not set(target_names).issubset(valid_tiers):
            invalid_tiers = set(target_names) - valid_tiers
            raise ValueError(f"Invalid tier names: {invalid_tiers}")

        # Compute performance on each phase for the specified tiers
        perf = {}
        for tier in target_names:
            perf[tier] = []
        for phase in self.phases:
            if "nfs_bw" in target_names:
                perf["nfs_bw"].append(self.run_phase_workload(phase, self.target["nfs"], False))
            if "lfs_bw" in target_names:
                perf["lfs_bw"].append(self.run_phase_workload(phase, self.target["lfs"], False))
            if "sbb_bw" in target_names:
                perf["sbb_bw"].append(self.run_phase_workload(phase, self.target["lfs"], True))

        # Assemble performance data into a DataFrame
        perf_df = pd.DataFrame(perf)

        return perf_df

class DataTable:
    def __init__(self, targets,  accelerator=False, ioi=False, sample=1, filename=None, lite=False):
        """Initializes the data model with the phase features extracted from AppDecomposer
        Args:
            filename (string): file name of dataset in csv, default to DATASET_FILE
            targets (dict): storage backend file (nfs or lustre) {"lfs":..., "nfs":,...}
            ioi (bool): enable/disable IOI, default = False
            accelerator (string): using IO acclerator such as SBB/FIOL
            sample (int): number of sampling of the same phase
            lite (bool): if True, caps the volume to 1GB, else use the phase volume for bw measurement.
        """
        self.filename = filename or DATASET_SOURCE
        self.targets = targets
        self.accelerator = accelerator
        self.ioi = ioi
        self.sample = sample
        self.lite = lite

    def get_tiers_from_targets(self):
        """Given target directory and accelerator attribute, determine the list of tiers that will be used as columns to fill the dataset."""
        tiers_names = [tier + "_bw" for tier in list(self.targets.keys())]
        if self.accelerator:
            tiers_names.append("sbb_bw")
        return tiers_names

    def get_performance_table(self, output_filename=None):
        """
        Reads performance data from a CSV file, completes the data with performance information, and saves
        the resulting table to a new CSV file.

        If output_filename is not provided, the new file will be named after the original file with "_completed"
        added before the file extension.

        Args:
            output_filename (str, optional): The name of the file to save the completed performance data to.

        Returns:
            A pandas DataFrame containing the completed performance data.

        Raises:
            FileNotFoundError: If the input file does not exist or cannot be read.
            ValueError: If the input file is not in the expected format.

        Example:
            target = dict(lfs="/fsiof/mimounis/tmp", nfs="/scratch/mimounis/tmp")
            acc = "SBB" # currently support onyly SBB with the lfs target
            filename = "/home_nfs/mimounis/iosea-wp3-recommandation-system/performance_data/performance_data/dataset/test_dataset_job_3918.csv"
            complete_filename = f"/home_nfs/mimounis/iosea-wp3-recommandation-system/performance_data"\
                                f"/performance_data/dataset/complete_dataset_"\
                                f"job_3918_{convert_size(__CAPING_VOLUME__)}.csv"
            # targets,  accelerator=False, ioi=False, sample=1, filename=None, lite=False
            pm = DataTable(target, accelerator=True, ioi=False, sample=1, filename=filename, lite=True)
            print(pm)
            df = pm.get_performance_table(filename=complete_filename)
            print(df)
        """
        # adjust target file for complete data
        if output_filename is None:
            base_filename, ext_filename = os.path.splitext(self.filename)
            output_filename = base_filename + "_completed" + ext_filename

        # Load performance data from file
        self.perf_data = pd.read_csv(self.filename)

        # Extract tiers and log messages
        tiers_names = self.get_tiers_from_targets()
        logger.info(f"Phases are extracted from this table: {self.perf_data}")
        logger.info(f"Following tiers will be fed: {tiers_names}")

        # Check if all required tiers are present in the input data
        if set(tiers_names).issubset(self.perf_data.columns):
            # Split data into parts with and without performance information
            old_data = self.perf_data[~self.perf_data.isna().any(axis=1)]
            new_data = self.perf_data[self.perf_data.isna().any(axis=1)]

            # Update performance information for the missing parts
            new_data = new_data.drop(tiers_names, axis=1)
            phases = new_data.to_dict('records')
            phases_perf = PhaseData(phases, self.targets, self.ioi, sample=self.sample, lite=self.lite)
            perf_df = phases_perf.get_phase_data(tiers_names)
            perf_df.index = new_data.index
            new_data = new_data.join(perf_df)

            # Combine old and new data
            self.perf_data = pd.concat([old_data, new_data], axis=0)
        else:
            # Update performance information for all rows
            phases = self.perf_data.to_dict('records')
            phases_perf = PhaseData(phases, self.targets, self.ioi, sample=self.sample, lite=self.lite)
            perf_df = phases_perf.get_phase_data(tiers_names)
            self.perf_data = self.perf_data.join(perf_df)

        # Save completed performance data to file and log message
        self.perf_data.to_csv(output_filename, index=False)
        logger.info(f"Complete table saved to: {output_filename}")

        return self.perf_data


