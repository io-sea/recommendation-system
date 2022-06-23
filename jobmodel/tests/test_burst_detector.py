import unittest

from jobmodel.job_decomposer.burst_detector import BurstDectector
import pybursts as pyb
import pickle
import numpy as np

class TestBurstDetector(unittest.TestCase):
    def setUp(self):
        with open('data_samples.pkl', 'rb') as fp:
            data = pickle.load(fp)
        jobids = list(data.keys())
        j = 12
        R = data[jobids[j]]['bytesRead']
        W = data[jobids[j]]['bytesWritten']
        self.signal = W[250:450]
        self.offsets = [4, 17, 23, 27, 33, 35, 37, 76, 77, 82, 84, 88, 90, 92]

    def test_tau(self):
        b = pyb.pybursts.kleinberg(self.offsets)
        print(b)
        #self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
