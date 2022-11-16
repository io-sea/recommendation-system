__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""
import os
import sys
sys.path.append('../../')
import pandas as pd
from fakeapp_generator import fakeapp_generator

class PhasePerformance:

    def __init__(self, phases, target,  accelerator="", ioi=False):
        """Initializes the performance model with the phase features extracted from AppDecomposer
        Args:
            phases (list): list of phases to be measured
            targets (dict): storage backend file (nfs or lustre)
            ioi (bool): enable/disable IOI, default = False
            accelerator (string): using IO acclerator such as SBB/FIOL
        """
        self.phases = phases
        self.target = target
        self.ioi = ioi
        self.accelerator = accelerator

    def get_performances(self, sample=1):
        """Compute the performance on each tier with the phase features extracted from AppDecomposer
        Args:
            sample (int): number of sampling of the phase

        Return:
            per_df (pandas): performances in each tier
        """
        #self.extract_phases()
        perf_nfs = []
        perf_lfs = []
        perf_sbb = []
        for phase in self.phases:
            #run the fakeapp to mesure the bandwidth on all tiers
            perf_nfs.append(self.get_phase_bandwidth(phase, self.target["nfs"], "", 2))
            perf_lfs.append(self.get_phase_bandwidth(phase, self.target["lfs"], "", 2))
            perf_sbb.append(self.get_phase_bandwidth(phase, self.target["lfs"], self.accelerator, 2))

        #update phase performance on the dataframe
        perf_df = pd.DataFrame()
        perf_df["nfs_bw"] = perf_nfs
        perf_df["lfs_bw"] = perf_lfs
        perf_df["sbb_bw"] = perf_sbb

        return perf_df

    def get_phase_bandwidth(self, phase, target, accelerator="", sample=1):
        """Compute the average bandwidht on a storage tier with the phase features extracted from AppDecomposer
        Args:
            phase (dict): phase to be measured
            target (string): storage backend
            accelerator (string): using IO acclerator such as SBB/FIOL
            sample (int): number of sampling of the phase

        Return:
            avg_bw (float):  average bandwidth
        """
        #run fakeapp n times to get the avg bandwidth
        sum = 0
        for i in range(1, sample+1):
            (t, bw) = fakeapp_generator.gen_fakeapp(phase["volume"], phase["mode"], phase["IOpattern"],
                    phase["IOsize"], phase["nodes"], target, accelerator, self.ioi)
            sum += bw
        avg_bw = (float)(sum/sample)
        print("Performance on tier", target, accelerator, ": ", format(avg_bw/(1024*1024), '.2f'), "(Mb/s)")
        return avg_bw

def load_dataset(filename):
    data = pd.read_csv(filename)
    return data

def export_to_csv(data, filename):
    data.to_csv(filename)

class PerformanceModel:
    def __init__(self, filename, targets,  accelerator="", ioi=False):
        """Initializes the performance model with the phase features extracted from AppDecomposer
        Args:
            filename (string): file name of dataset in csv
            targets (dict): storage backend file (nfs or lustre)
            ioi (bool): enable/disable IOI, default = False
            accelerator (string): using IO acclerator such as SBB/FIOL
        """
        self.filename = filename
        self.targets = targets
        self.accelerator = accelerator
        self.ioi = ioi

    def get_perfomance_table(self):

        #load performance model from file
        self.perf_data = load_dataset(self.filename)

        print(self.perf_data)
        if {'nfs_bw', 'lfs_bw', 'sbb_bw'}.issubset(self.perf_data.columns):
            #update the performance table with new phases only
            new_data = self.perf_data.loc[self.perf_data['nfs_bw'] == '']
            phases = new_data.to_dict('records')

        else:
            # transform rows in the dataframe to a list of phase features
            phases = self.perf_data.to_dict('records')
            phases_perf = PhasePerformance(phases, self.targets, self.accelerator, self.ioi)
            perf_df = phases_perf.get_performances()
            self.perf_data = self.perf_data.join(perf_df)

        return self.perf_data

if __name__ == '__main__':
    target = dict(lfs="/fsiof/phamtt/tmp", nfs="/scratch/phamtt/tmp")
    acc = "SBB" # currently SBB works only with the lfs target
    filename = "/home_nfs/phamtt/IO-SEA/iosea-wp3-recommandation-system/performance_model/dataset/performance_model_dataset.csv"
    pm = PerformanceModel(filename, target, acc)
    df = pm.get_perfomance_table()
    print(df)
"""
    phase0=dict(volume=100000000, mode="write", IOpattern="stride", IOsize=10000, nodes=1)
    phase1=dict(volume=100000000, mode="write", IOpattern="rand", IOsize=10000, nodes=1)
    phase2=dict(volume=100000000, mode="read", IOpattern="seq", IOsize=10000, nodes=1)
    phases = [phase0, phase1, phase2]
    perf_data = PhasePerformance(phases, target, acc)
    df=perf_data.get_perfomrances(1)
    print(df)
"""
