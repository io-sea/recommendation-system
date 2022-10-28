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

    def __init__(self, target,  accelerator="", ioi=False):
        """Initializes the performance model with the phase features extracted from AppDecomposer
        Args:
            phase (list): list of phases features to be generated
            targets (list): storage backend file (nfs/fs1)
            ioi (bool): enable/disable IOI, default = False
            accelerator (string): using IO acclerator such as SBB/FIOL
        """
        #self.phase = phase
        self.target = target
        self.ioi = ioi
        self.accelerator = accelerator

    def get_data():

        return

    def export_to_csv():

        return

    def get_phase_bandwidth(self, phase, sample=1):
        #run app n times to get the avg bandwidth
        bw = 0
        bw = fakeapp_generator.gen_fakeapp(phase["volume"], phase["mode"], phase["IOpattern"],
                    phase["IOsize"], phase["nodes"], self.ioi, self.accelerator)
        return bw

if __name__ == '__main__':
    lfs="/fsiof/phamtt/tmp"
    nfs="/scratch/phamtt/tmp"
    acc = "SBB"
    perf_data_nfs = PhasePerformance(nfs)
    perf_data_lfs = PhasePerformance(lfs)
    perf_data_sbb = PhasePerformance(lfs, acc)

    phase1=dict(volume=1000000000, mode="Write", IOpattern="Random", IOsize=10000, nodes=2)
    perf_data_nfs.get_phase_bandwidth(phase1, 2)