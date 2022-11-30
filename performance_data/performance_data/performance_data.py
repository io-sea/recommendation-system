__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""
import os
import sys
import pandas as pd
from performance_data.fakeapp_workload import FakeappWorkload as Workload

# TODO: iodrop cache before perf measurement
# TODO: exclusive mode for sbatch to avoid perf crosstalk
# TODO: update dataframe for each row so dataset updates iteratively (more robust)
# TODO: refactor + unittests

class PhasePerformance:

    def __init__(self, phases, target, ioi=False, sample=1):
        """Initializes the performance model with the phase features as extracted from AppDecomposer
        Args:
            phases (list): list of phases to be measured
            targets (dict): storage backend file (nfs or lustre)
            ioi (bool): enable/disable IOI, default = False
            accelerator (string): using IO acclerator such as SBB/FIOL
            sample (int): number of sampling of the same phase
        """
        self.phases = phases
        self.target = target
        self.ioi = ioi
        self.sample = sample

    def get_performances(self):
        """Compute the performance on each tier with the phase features extracted from AppDecomposer
        Args:

        Return:
            per_df (pandas): performances in each tier
        """
        #self.extract_phases()
        perf_nfs = []
        perf_lfs = []
        perf_sbb = []
        for phase in self.phases:
            #run the fakeapp to mesure the bandwidth on all tiers
            perf_nfs.append(self.get_phase_bandwidth(phase, self.target["nfs"], False))
            perf_lfs.append(self.get_phase_bandwidth(phase, self.target["lfs"], False))
            perf_sbb.append(self.get_phase_bandwidth(phase, self.target["lfs"], True))

        #update phase performance on the dataframe
        perf_df = pd.DataFrame()
        perf_df["nfs_bw"] = perf_nfs
        perf_df["lfs_bw"] = perf_lfs
        perf_df["sbb_bw"] = perf_sbb

        return perf_df

    def get_phase_bandwidth(self, phase, target, accelerator=False):
        """Compute the average bandwidht on a storage tier with the phase features extracted from AppDecomposer
        Args:
            phase (dict): phase to be measured
            target (string): storage backend
            accelerator (string): using IO acclerator such as SBB/FIOL

        Return:
            avg_bw (float):  average bandwidth
        """
        #run fakeapp n times to get the avg bandwidth
        bandwidths = 0
        for i in range(0, self.sample):
            # TODO : should be able to control volume of workload to adjust accuracy.
            workload = Workload(volume=phase["volume"], mode=phase["mode"],
                                io_pattern=phase["IOpattern"], io_size=phase["IOsize"],
                                nodes=phase["nodes"], target_tier=target,
                                accelerator=accelerator, ioi=self.ioi)
            (t, bw) = workload.get_data()
            # (t, bw) = fakeapp_generator.gen_fakeapp(phase["volume"], phase["mode"],
            #                                         phase["IOpattern"], phase["IOsize"], phase["nodes"], target, accelerator, self.ioi)
            bandwidths += bw
        avg_bw = (float)(bandwidths/self.sample)
        print("Performance on tier", target, accelerator, ": ", format(avg_bw/(1024*1024), '.2f'), "(Mb/s)")
        return avg_bw

def load_dataset(filename):
    data = pd.read_csv(filename)#, index_col= 0)
    return data

def export_to_csv(data, filename):
    data.to_csv(filename, index=False)

class PerformanceModel:
    def __init__(self, filename, targets,  accelerator=False, ioi=False, sample=1):
        """Initializes the performance model with the phase features extracted from AppDecomposer
        Args:
            filename (string): file name of dataset in csv
            targets (dict): storage backend file (nfs or lustre)
            ioi (bool): enable/disable IOI, default = False
            accelerator (string): using IO acclerator such as SBB/FIOL
            sample (int): number of sampling of the same phase
        """
        self.filename = filename
        self.targets = targets
        self.accelerator = accelerator
        self.ioi = ioi
        self.sample = sample

    def get_perfomance_table(self):

        #load performance model from file
        self.perf_data = load_dataset(self.filename)

        print(self.perf_data)
        if {'nfs_bw', 'lfs_bw', 'sbb_bw'}.issubset(self.perf_data.columns):
            #keep the part already has performance
            old_data = self.perf_data[~self.perf_data.isna().any(axis=1)]
            #print(old_data)

            #update the performance phases which has NAN value in the bandwidth
            new_data = self.perf_data[self.perf_data.isna().any(axis=1)]
            new_data = new_data.drop(['nfs_bw', 'lfs_bw', 'sbb_bw'], axis=1)
            #print(new_data)
            phases = new_data.to_dict('records')
            phases_perf = PhasePerformance(phases, self.targets, self.accelerator, self.ioi)
            perf_df = phases_perf.get_performances()
            perf_df.index = new_data.index
            new_data = new_data.join(perf_df)
            #print(new_data)

            self.perf_data = pd.concat([old_data, new_data], axis=0)
            #self.perf_data.reset_index(drop=True)

        else:
            # transform rows in the dataframe to a list of phase features
            phases = self.perf_data.to_dict('records')
            phases_perf = PhasePerformance(phases, self.targets, self.accelerator, self.ioi)
            perf_df = phases_perf.get_performances()
            self.perf_data = self.perf_data.join(perf_df)

        print(self.perf_data)
        return self.perf_data

if __name__ == '__main__':
    target = dict(lfs="/fsiof/mimounis/tmp", nfs="/scratch/mimounis/tmp")
    acc = "SBB" # currently support onyly SBB with the lfs target
    filename = "/home_nfs/mimounis/iosea-wp3-recommandation-system/performance_model/dataset/performance_model_dataset.csv"
    # filename = "/home_nfs/mimounis/iosea-wp3-recommandation-system/performance_model/dataset/performance_model_dataset_small_partial.csv"

    pm = PerformanceModel(filename, target, acc)
    df = pm.get_perfomance_table()
    print(df)
    filename_pm = "/home_nfs/mimounis/iosea-wp3-recommandation-system/performance_model/dataset/performance_model_dataset_completed_2.csv"
    export_to_csv(df, filename_pm)

"""
    phase0=dict(volume=100000000, mode="write", IOpattern="stride", IOsize=10000, nodes=1)
    phase1=dict(volume=100000000, mode="write", IOpattern="rand", IOsize=10000, nodes=1)
    phase2=dict(volume=100000000, mode="read", IOpattern="seq", IOsize=10000, nodes=1)
    phases = [phase0, phase1, phase2]
    perf_data = PhasePerformance(phases, target, acc)
    df=perf_data.get_perfomrances(1)
    print(df)
"""
