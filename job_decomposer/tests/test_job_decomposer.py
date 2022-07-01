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

random.seed(42)

def choose_random_job(dataset_path = "C:\\Users\\a770398\\IO-SEA\\io-sea-3.4-analytics\\dataset_generation\\dataset_generation"):
    job_files = []
    for root, dirs, files in os.walk(dataset_path):
        for csv_file in files:
            if csv_file.endswith(".csv"):
                job_files.append(os.path.join(root, csv_file))
    return random.choice(job_files)

class TestJobDecInit(unittest.TestCase):
    def setUp(self):
        """Prepare random job dataframe."""
        csv_file = choose_random_job()
        self.df = pd.read_csv(csv_file, index_col=0)
        
       
    def test_job_decomposer_init(self):
        jd = JobDecomposer(self.df)
        self.df.plot(x="timestamp", figsize=(16,4), sharex=True, grid=True, subplots=True, layout=(1, 4))
        self.assertTrue(isinstance(jd.dataframe, pd.DataFrame))
        
        
    def test_decompose(self):
        jd = JobDecomposer(self.df)
        breakpoints, loss = jd.decompose(self.df[["bytesWritten"]].to_numpy(),
                                         rpt.Pelt, rpt.costs.CostNormal(), 10)
        
        expected_breakpoints = [5, 70, 75, 122]
        self.assertListEqual(breakpoints, expected_breakpoints)
        self.assertAlmostEqual(loss, -1260.0282214487108)
        
    def test_get_compute_write(self):
        jd = JobDecomposer(self.df)
        signal = self.df[["bytesWritten"]].to_numpy().flatten()
        breakpoints, loss = jd.decompose(signal, rpt.Pelt, rpt.costs.CostNormal(), 10)
        compute, write = jd.get_compute_vector(signal, breakpoints)
        print(compute)
        print(write)
        self.df.plot(x="timestamp", figsize=(16,4), sharex=True, grid=True, subplots=True, layout=(1, 4))
        plt.show()

       

if __name__ == '__main__':
    unittest.main(verbosity=2)
    