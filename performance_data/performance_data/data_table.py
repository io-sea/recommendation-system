__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""
import os
from os.path import dirname
import sys
import shutil
import pandas as pd
from loguru import logger
from app_decomposer.utils import convert_size
from app_decomposer import DATASET_SOURCE
from typing import List
from performance_data import DATASET_FILE
from performance_data.fakeapp_workload import FakeappWorkload as Workload

__CAPING_VOLUME__ = 1e9


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
        # run fakeapp n times to get the avg bandwidth
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
        
        # logger.info(f"Latencies: {latencies}")
        avg_bw = (float)(volumes/latencies) if latencies else 0
        logger.info(f"Measured throughput on tier: {target} | use SBB: {accelerator} | result: {convert_size(avg_bw)}/s")
        return avg_bw

    def get_phase_data(self, target_names: List[str] = ["nfs_bw",
                                                        "lfs_bw", "sbb_bw"]) -> pd.DataFrame:
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
        valid_tiers = {tier + "_bw" for tier in list(self.target.keys())}

        valid_tiers = valid_tiers.union({tier + "_sbb_bw" for tier in list(self.target.keys())})
        if not set(target_names).issubset(valid_tiers):
            invalid_tiers = set(target_names) - valid_tiers
            raise ValueError(f"Invalid tier names: {invalid_tiers}")

        # Compute performance on each phase for the specified tiers
        perf = {}
        for tier in target_names:
            perf[tier] = []
        for phase in self.phases:
            for target in self.target.keys():
                if target + "_bw" in target_names:
                    perf[target + "_bw"].append(
                        self.run_phase_workload(phase, 
                                                self.target[target], False))
                if target + "_sbb_bw" in target_names:
                    perf[target + "_sbb_bw"].append(
                        self.run_phase_workload(phase, 
                                                self.target[target], True))

        # Assemble performance data into a DataFrame
        perf_df = pd.DataFrame(perf)

        return perf_df


class DataTable:
    """A class for reading, completing, and saving performance data for storage system simulations.

    Attributes:
        filename (str): The name of the file containing the performance data.
        targets (dict): A dictionary with keys specifying the names of the storage backend files, and values
            specifying their file paths.
        accelerator (str): The name of an I/O accelerator to use, such as SBB or FIOL.
        ioi (bool): A flag indicating whether to use IOI.
        sample (int): The number of times to sample each phase for performance measurements.
        lite (bool): A flag indicating whether to cap the volume of the phase for performance measurement at 1GB.
        perf_data (pandas.DataFrame): A DataFrame containing the performance data.
    """

    def __init__(self, targets,  accelerator=False, ioi=False, 
                 sample=1, filename=None, lite=False):
        """Initialize the data model with the phase features extracted from AppDecomposer.

        Args:
            filename (str, optional): The name of the file containing the data. Defaults to `DATASET_FILE`.
            targets (dict): A dictionary with keys specifying the names of the storage backend files, and values
                specifying their file paths.
            accelerator (str, optional): The name of an I/O accelerator to use, such as SBB or FIOL. Defaults to `False`.
            ioi (bool, optional): A flag indicating whether to use IOI. Defaults to `False`.
            sample (int, optional): The number of times to sample each phase for performance measurements. Defaults to `1`.
            lite (bool, optional): A flag indicating whether to cap the volume of the phase for performance measurement at 1GB. Defaults to `False`.
        """
        self.filename = filename or DATASET_SOURCE
        logger.info(f"Reading performance data from {self.filename}")
        self.targets = targets
        self.accelerator = accelerator
        self.ioi = ioi
        self.sample = sample
        self.lite = lite

    def get_tiers_from_targets(self):
        """Get a list of tiers to use as columns in the dataframe.

        Returns:
            list: A list of tier names, including the names of the storage backend files and the I/O accelerator.
        """
        tiers_names = [tier + "_bw" for tier in list(self.targets.keys())]
        if self.accelerator:
            tiers_names.extend([tier + "_sbb_bw" for tier in list(self.targets.keys()) if self.accelerator])
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
        if self.filename is None:
            raise ValueError("No filename provided.")

        # adjust target file for complete data
        if output_filename is None:
            base_filename, ext_filename = os.path.splitext(self.filename)
            output_filename = base_filename + "_completed" + ext_filename

        try:
            # Load performance data from file
            self.perf_data = pd.read_csv(self.filename)
        except FileNotFoundError as e:
            logger.error(f"File not found: {self.filename}. {e}")
            raise

        # Extract tiers and log messages
        tiers_names = self.get_tiers_from_targets()
        logger.info(f"Phases are extracted from this table: {self.perf_data}")
        logger.info(f"Following tiers will be fed: {tiers_names}")

        total_iterations = len(self.perf_data)
        logger.info(f"Total number of iterations: {total_iterations}")

        # Iterate over phases, compute performance, and update perf_data for each phase
        for i, row in self.perf_data.iterrows():
            remaining_iterations = total_iterations - i
            logger.info(f"Remaining iterations: {remaining_iterations}")

            phase = row.to_dict()
            phases_perf = PhaseData([phase], self.targets, self.ioi, 
                                    sample=self.sample, lite=self.lite)
            perf_df = phases_perf.get_phase_data(tiers_names)
            for tier in tiers_names:
                self.perf_data.loc[i, tier] = perf_df.loc[0, tier]

            # Save completed performance data to file and log message after each phase
            self.perf_data.to_csv(output_filename, index=False)
            logger.info(f"Updated table saved to: {output_filename} after processing phase at index {i}")
            
            # Log the updated ith row of the dataframe
            logger.info(f"Updated row at index {i}: \n {self.perf_data.loc[i]}")

        return self.perf_data
    
    def complete_output_file(self, incomplete_output_filename):
        """
        Completes the output file by computing only lacking results and adding them to the existing CSV file.

        Args:
            incomplete_output_filename (str): The name of the incomplete output file.
        """
        logger.info(f"Attempting to complete the output file: {incomplete_output_filename}")

        # Read the incomplete output file
        existing_output_data = pd.read_csv(incomplete_output_filename)

        # Determine the proxy filename (e.g., append "_proxy" to the filename)
        proxy_filename = os.path.splitext(incomplete_output_filename)[0] + "_proxy.csv"

        # Determine the number of incomplete rows
        incomplete_rows = existing_output_data.isnull().any(axis=1).sum()
        logger.info(f"Total number of incomplete rows: {incomplete_rows}")

        # Iterate over the rows, identify the incomplete rows, compute performance, and update the DataFrame
        tiers_names = self.get_tiers_from_targets()
        for i, row in existing_output_data.iterrows():
            # Check if any value is missing in the current row
            if row.isnull().any():
                logger.info(f"Processing incomplete row at index {i}")
                phase = row.to_dict()
                phases_perf = PhaseData([phase], self.targets, self.ioi, sample=self.sample, lite=self.lite)
                perf_df = phases_perf.get_phase_data(tiers_names)
                # Fill missing values in the current row
                for tier in tiers_names:
                    if pd.isnull(row[tier]):
                        row[tier] = perf_df.loc[0, tier]
                existing_output_data.loc[i] = row  # Update the row in the DataFrame

                # Save the updated DataFrame to the proxy file
                existing_output_data.to_csv(proxy_filename, index=False)
                logger.info(f"Updated table saved to proxy file: {proxy_filename} after processing phase at index {i}")

        # Copy and rename the proxy file to the original filename, effectively overwriting the incomplete file
        backup_filename = os.path.splitext(incomplete_output_filename)[0] + "_proxy" + os.path.splitext(incomplete_output_filename)[1]

        shutil.copy(proxy_filename, backup_filename)
        os.rename(proxy_filename, incomplete_output_filename)
        logger.info("Completion of the output file successful.")
        return existing_output_data

