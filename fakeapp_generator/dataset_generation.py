__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""
import os
import pandas as pd
from fakeapp_generator import fakeapp_generator

class PhaseFeatures:
    def __init__(self, volume, mode, IOpattern, IOsize, nodes):
        """Initializes the performance model with the phase features extracted from AppDecomposer
        Args:
            volume (int): volume of io will be generated
            mode (string): read/write
            IOpattern (string): Seq/Stride/Random
            IOsize (string): size of IO
        """
        self.volume = volume
        self.mode = mode
        self.IOpattern = IOpattern
        self.IOsize = IOsize
        self.nodes = nodes

    def get_phase_srtring(self, sep=","):
        s = str(self.volume) + sep + str(self.mode) + sep + str(self.IOpattern) + sep + str(self.IOsize) + sep + str(self.nodes)
        return s

class DatasetGeneration:

    def __init__(self, phase, target, ioi=True, accelerator=""):
        """Initializes the performance model with the phase features extracted from AppDecomposer
        Args:
            nodes (int): number of nodes, default = 1
            target (string): storage backend file (nfs/fs1)
            ioi (bool): enable/disable IOI, default = False
            accelerator (string): using IO acclerator such as SBB/FIOL
        """
        self.phase = phase
        self.target = target
        self.ioi = ioi
        self.accelerator = accelerator

    def get_phase_performance():

        return

    def export_to_csv():

        return

    def run_app(sample=1):
        #run app n times to get the avg bandwidth
        bw = 0
        bw = fakeapp_generator.gen_fakeapp()
        return