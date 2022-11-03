__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""
import os
import pandas as pd
import fakeapp_generator

class PhaseFeatures:
    def __init__(self, volume, mode, IOpattern, IOsize, nodes):
        """Initializes the performance model with the phase features extracted from AppDecomposer
        Args:
            volume (int): volume of io will be generated
            mode (string): read/write
            IOpattern (string): Seq/Stride/Random
            IOsize (string): size of IO
            nodes (int): number of nodes, default = 1
        """
        self.volume = volume
        self.mode = mode
        self.IOpattern = IOpattern
        self.IOsize = IOsize
        self.nodes = nodes

    def get_phase_srtring(self, sep=","):
        s = str(self.volume) + sep + str(self.mode) + sep + str(self.IOpattern) + sep + str(self.IOsize) + sep + str(self.nodes)
        return s

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

    def load_dataset(self, filename):
        self.df = pd.read_csv(filename)

    def extract_phases(self):
        phase = dict()
        self.phases.append(phase)

    def export_to_csv(self, filename):
        self.df.to_csv(filename)

    def run_phases(self, sample=1):
        self.extract_phases()
        for phase in self.phases:
            #run the fakeapp to mesure the bandwidth on all tiers
            perf_nfs = self.get_phase_bandwidth(phase, self.target["nfs"], "", 2)
            perf_lfs = self.get_phase_bandwidth(phase, self.target["lfs"], "", 2)
            perf_sbb = self.get_phase_bandwidth(phase, self.target["lfs"], self.accelerator, 2)

            #update phase performance on the dataframe

    def get_phase_bandwidth(self, phase, target, accelerator="", sample=1):
        #run fakeapp n times to get the avg bandwidth
        sum = 0
        for i in range(1, sample+1):
            (t, bw) = fakeapp_generator.gen_fakeapp(phase["volume"], phase["mode"], phase["IOpattern"],
                    phase["IOsize"], phase["nodes"], target, accelerator, self.ioi)
            sum += bw
        avg_bw = (float)(sum/sample)
        print("Performance on tier", target, accelerator, ": ", format(avg_bw/(1024*1024), '.2f'), "(Mb/s)")
        return avg_bw

if __name__ == '__main__':
    target = dict(lfs="/fsiof/phamtt/tmp", nfs="/scratch/phamtt/tmp")
    acc = "SBB" # currently SBB works only with the lfs target
    phase1=dict(volume=100000000, mode="Write", IOpattern="Random", IOsize=10000, nodes=1)
    phase2=dict(volume=100000000, mode="Read", IOpattern="Seq", IOsize=10000, nodes=1)
    phases = [phase1, phase2]
    perf_data = PhasePerformance(phases, target, acc)
    perf_data.run_phases(2)
