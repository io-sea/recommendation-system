__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""
import os
#import pandas as pd
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

    def __init__(self, phase, target,  accelerator="", ioi=False):
        """Initializes the performance model with the phase features extracted from AppDecomposer
        Args:
            phase (list): list of phases features to be generated
            targets (list): storage backend file (nfs/fs1)
            ioi (bool): enable/disable IOI, default = False
            accelerator (string): using IO acclerator such as SBB/FIOL
        """
        self.phase = phase
        self.target = target
        self.ioi = ioi
        self.accelerator = accelerator

    def get_data():

        return

    def export_to_csv():

        return

    def get_phase_bandwidth(self, sample=1):
        #run app n times to get the avg bandwidth
        sum = 0
        for i in range(1, sample+1):
            (t, bw) = fakeapp_generator.gen_fakeapp(self.phase["volume"], self.phase["mode"], self.phase["IOpattern"],
                    self.phase["IOsize"], self.phase["nodes"], self.target, self.ioi, self.accelerator)
            sum += bw
        avg_bw = (float)(sum/sample)
        print("Performance on tier ", self.target, self.accelerator, ": ", format(avg_bw/(1024*1024), '.2f'), "(Mb/s)")
        return avg_bw

if __name__ == '__main__':
    lfs="/fsiof/phamtt/tmp"
    nfs="/scratch/phamtt/tmp"
    acc = "SBB"
    phase1=dict(volume=100000000, mode="Write", IOpattern="Random", IOsize=10000, nodes=1)

    perf_data_nfs = PhasePerformance(phase1, nfs)
    perf_data_lfs = PhasePerformance(phase1, lfs)
    perf_data_sbb = PhasePerformance(phase1, lfs, acc)

    perf = perf_data_sbb.get_phase_bandwidth(2)
