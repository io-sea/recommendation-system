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
from performance_data import DATASET_FILE
from performance_data.fakeapp_workload import FakeappWorkload as Workload

class PhaseData:
    def __init__(self, phases, target, ioi=False, sample=1):
        """Initializes the phases data with features extracted from AppDecomposer, runs the associated workload and return data.
        Args:
            phases (list): list of phases to be measured
            target (dict): storage backend file (nfs or lustre)
            ioi (bool): enable/disable IOI, default = False
            accelerator (string): using IO accelerator such as SBB/FIOL
            sample (int): number of sampling of the same phase
        """
        self.phases = phases
        self.target = target
        self.ioi = ioi
        self.sample = sample

    def run_phase_workload(self, phase, target, accelerator=False):
        """Compute the average bandwidht on a storage tier with the phase features extracted from AppDecomposer
        Args:
            phase (dict): phase to be measured
            target (string): storage backend
            accelerator (string): using IO acclerator such as SBB/FIOL

        Return:
            avg_bw (float):  average bandwidth
        """
        #run fakeapp n times to get the avg bandwidth
        latencies = 0
        volumes = 0
        for _ in range(self.sample):
            # TODO : should be able to control volume of workload to adjust accuracy.
            workload = Workload(volume=phase["volume"], mode=phase["mode"],
                                io_pattern=phase["IOpattern"], io_size=phase["IOsize"],
                                nodes=phase["nodes"], target_tier=target,
                                accelerator=accelerator, ioi=self.ioi)
            (latency, bw) = workload.get_data()
            latencies += latency
            volumes += phase["volume"]
        # print(f"n_samples = {self.sample}")
        avg_bw = (float)(volumes/latencies) if latencies else 0
        logger.info(f"Measured throughput on tier: {target} | use SBB: {accelerator} | result: {convert_size(avg_bw)}/s")
        return avg_bw

    def get_phase_data(self, target_names=["nfs_bw", "lfs_bw", "sbb_bw"]):
        """Compute the performance on each tier with the phase features extracted from AppDecomposer
        Args:

        Return:
            per_df (pandas): performances in each tier
        """
        #self.extract_phases()
        if "nfs_bw" in target_names:
            perf_nfs = []
        if "lfs_bw" in target_names:
            perf_lfs = []
        if "sbb_bw" in target_names:
            perf_sbb = []
        for phase in self.phases:
            #run the fakeapp to mesure the bandwidth on all tiers
            if "nfs_bw" in target_names:
                perf_nfs.append(self.run_phase_workload(phase, self.target["nfs"], False))
            if "lfs_bw" in target_names:
                perf_lfs.append(self.run_phase_workload(phase, self.target["lfs"], False))
            if "sbb_bw" in target_names:
                perf_sbb.append(self.run_phase_workload(phase, self.target["lfs"], True))

        #update phase performance on the dataframe
        perf_df = pd.DataFrame()
        if "nfs_bw" in target_names:
            perf_df["nfs_bw"] = perf_nfs
        if "lfs_bw" in target_names:
            perf_df["lfs_bw"] = perf_lfs
        if "sbb_bw" in target_names:
            perf_df["sbb_bw"] = perf_sbb
        return perf_df

class DataTable:
    def __init__(self, targets,  accelerator=False, ioi=False, sample=1, filename=None):
        """Initializes the data model with the phase features extracted from AppDecomposer
        Args:
            filename (string): file name of dataset in csv, default to DATASET_FILE
            targets (dict): storage backend file (nfs or lustre) {"lfs":..., "nfs":,...}
            ioi (bool): enable/disable IOI, default = False
            accelerator (string): using IO acclerator such as SBB/FIOL
            sample (int): number of sampling of the same phase
        """
        self.filename = filename if filename else DATASET_SOURCE
        self.targets = targets
        self.accelerator = accelerator
        self.ioi = ioi
        self.sample = sample

    def get_tiers_from_targets(self):
        """Given target directory and accelerator attribute, determine the list of tiers that will be used as columns to fill the dataset."""
        tiers_names = [tier + "_bw" for tier in list(self.targets.keys())]
        if self.accelerator:
            tiers_names.append("sbb_bw")
        return tiers_names

    def get_performance_table(self, filename=None):
        # adjust target file for complete data
        if filename is None:
            filename = DATASET_FILE

        #load performance data from file
        self.perf_data = pd.read_csv(self.filename)#, index_col= 0)
        tiers_names = self.get_tiers_from_targets()
        logger.info(f"Phases are extracted from this table: {self.perf_data}")
        logger.info(f"Following tiers will be feeded: {tiers_names}")
        if set(tiers_names).issubset(self.perf_data.columns):
            #keep the part already has performance
            old_data = self.perf_data[~self.perf_data.isna().any(axis=1)]
            #print(old_data)

            #update the performance phases which has NAN value in the bandwidth
            new_data = self.perf_data[self.perf_data.isna().any(axis=1)]
            new_data = new_data.drop(tiers_names, axis=1)
            #print(new_data)
            phases = new_data.to_dict('records')
            phases_perf = PhaseData(phases, self.targets, self.ioi)
            perf_df = phases_perf.get_phase_data(tiers_names)
            perf_df.index = new_data.index
            new_data = new_data.join(perf_df)
            #print(new_data)

            self.perf_data = pd.concat([old_data, new_data], axis=0)
            #self.perf_data.reset_index(drop=True)

        else:
            # transform rows in the dataframe to a list of phase features
            phases = self.perf_data.to_dict('records')
            phases_perf = PhaseData(phases, self.targets, self.ioi)
            perf_df = phases_perf.get_phase_data(tiers_names)
            self.perf_data = self.perf_data.join(perf_df)

        logger.info(f"Complete phases table: {self.perf_data}")
        self.perf_data.to_csv(filename, index=False)
        logger.info(f"Complete table saved in: {filename}")
        return self.perf_data

if __name__ == '__main__':
    target = dict(lfs="/fsiof/mimounis/tmp", nfs="/scratch/mimounis/tmp")
    acc = "SBB" # currently support onyly SBB with the lfs target
    filename = "/home_nfs/mimounis/iosea-wp3-recommandation-system/performance_model/dataset/performance_model_dataset.csv"
    # filename = "/home_nfs/mimounis/iosea-wp3-recommandation-system/performance_model/dataset/performance_model_dataset_small_partial.csv"

    pm = DataTable(filename, target, acc)
    df = pm.get_perfomance_table()
    print(df)
    filename_pm = "/home_nfs/mimounis/iosea-wp3-recommandation-system/performance_model/dataset/performance_model_dataset_completed_2.csv"
    export_to_csv(df, filename_pm)

"""
    phase0=dict(volume=100000000, mode="write", IOpattern="stride", IOsize=10000, nodes=1)
    phase1=dict(volume=100000000, mode="write", IOpattern="rand", IOsize=10000, nodes=1)
    phase2=dict(volume=100000000, mode="read", IOpattern="seq", IOsize=10000, nodes=1)
    phases = [phase0, phase1, phase2]
    perf_data = PhaseData(phases, target, acc)
    df=perf_data.get_perfomrances(1)
    print(df)
"""
