import unittest
import time
import numpy as np
import pandas as pd
import simpy
import os, random
import ruptures as rpt
import matplotlib.pyplot as plt
from scipy import integrate

from job_decomposer.job_decomposer import JobDecomposer




random.seed(52)

def choose_random_job(dataset_path = "C:\\Users\\a770398\\IO-SEA\\io-sea-3.4-analytics\\dataset_generation\\dataset_generation"):
    job_files = []
    for root, dirs, files in os.walk(dataset_path):
        for csv_file in files:
            if csv_file.endswith(".csv"):
                job_files.append(os.path.join(root, csv_file))
    return random.choice(job_files)


def plot_job_phases(x, signal, breakpoints):
    plt.plot(x, signal, lw=2, label="IO")
            
    for i_brk, brk in enumerate(breakpoints[:-1]):
        if i_brk % 2 == 0: # opening point
            plt.plot(x[brk], signal[brk], '>g')
        else:
            plt.plot(x[brk], signal[brk], '<r')
    plt.grid(True)
    plt.show()
class TestJobDecInit(unittest.TestCase):
    def setUp(self):
        """Prepare random job dataframe."""
        csv_file = choose_random_job()
        self.df = pd.read_csv(csv_file, index_col=0)
        
       
    def test_job_decomposer_init(self):
        jd = JobDecomposer(self.df)
        # self.df.plot(x="timestamp", figsize=(16,4), sharex=True, grid=True, subplots=True, layout=(1, 4))
        self.assertTrue(isinstance(jd.dataframe, pd.DataFrame))
        
        
    def test_decompose_real_signal(self):
        jd = JobDecomposer(self.df)
        breakpoints, loss = jd.get_breakpoints(self.df[["bytesWritten"]].to_numpy(),
                                         rpt.Pelt, rpt.costs.CostNormal(), 1e-3)
        
        expected_breakpoints = [2, 5, 69, 71, 122]
        plot_job_phases(self.df[["timestamp"]].to_numpy().tolist(),
                   self.df[["bytesWritten"]].to_numpy().tolist(),
                   breakpoints)
        print(f"{breakpoints = } | {loss = }")
        self.assertListEqual(breakpoints, expected_breakpoints)
        self.assertAlmostEqual(loss, -1424.5753100957786)
    
    def test_get_phases_synthetic_signal_1(self):
        jd = JobDecomposer(self.df)
        signal = [0, 0, 10, 10, 0, 0]
        x = [0, 1, 2, 3, 4, 5]
        breakpoints, loss = jd.get_breakpoints(np.array(signal), rpt.Pelt, rpt.costs.CostL2(), 1e-3)
        print(f"{breakpoints = } | {loss = }")
        compute, data, bandwidth = jd.get_phases(x, signal, breakpoints)
        print(f"{compute = } | {data = } | {bandwidth=}")
        plot_job_phases(x, signal, breakpoints)
        self.assertListEqual(breakpoints[:-1], [2, 4])
        self.assertListEqual(data, [0, 10])
        self.assertListEqual(bandwidth, [0, 5])
        
        
    def test_get_phases_synthetic_signal_2(self):
        jd = JobDecomposer(self.df)
        signal = [10, 10, 0, 0, 0, 0]
        x = [0, 1, 2, 3, 4, 5]
        breakpoints, loss = jd.get_breakpoints(np.array(signal), rpt.Pelt, rpt.costs.CostL2(), 1e-3)
        print(f"{breakpoints = } | {loss = }")
        compute, data, bandwidth = jd.get_phases(x, signal, breakpoints)
        print(f"{compute = } | {data = } | {bandwidth=}")
        plot_job_phases(x, signal, breakpoints)
        
        # self.assertListEqual(breakpoints[:-1], [2, 4])
        # self.assertListEqual(write, [0, 10])
        # self.assertListEqual(bandwidth, [0, 5])
        
    # def test_get_compute_write(self):
    #     jd = JobDecomposer(self.df)
    #     signal = self.df[["bytesWritten"]].to_numpy().flatten()
    #     x = self.df[["timestamp"]].to_numpy().flatten()
    #     breakpoints, loss = jd.decompose(signal, rpt.Pelt, rpt.costs.CostNormal(), 10)
    #     compute, write, bandwidth = jd.get_compute_vector(x, signal, breakpoints)
    #     print(compute)
    #     print(write)
    #     print(bandwidth)
        # self.df.plot(x="timestamp", figsize=(16,4), sharex=True, grid=True, subplots=True, layout=(1, 4))
        # plt.show()

       

if __name__ == '__main__':
    unittest.main(verbosity=2)
    