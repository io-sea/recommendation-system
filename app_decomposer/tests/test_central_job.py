import unittest
import numpy as np
import pandas as pd
from app_decomposer.central_job import WorkflowSynthesizer, CentralJob


class TestWorkflowSynthesizer(unittest.TestCase):

    def setUp(self):
        self.ws = WorkflowSynthesizer()

        self.job1 = {
            'bytesRead': np.array([6457, 0, 1090522169, 2055211225]),
            'bytesWritten': np.array([115404800, 191795200, 0, 0]),
            'timestamp': np.array([1581607510, 1581607515, 1581607520, 1581607525])
        }

        self.job2 = {
            'bytesRead': np.array([2457, 8000, 0, 0]),
            'bytesWritten': np.array([15404800, 91795200, 0, 0]),
            'timestamp': np.array([1581607520, 1581607525, 1581607530, 1581607535])
        }

        self.job3 = {
            'bytesRead': np.array([6457, 0, 1090522169, 2055211225]),
            'bytesWritten': np.array([115404800, 191795200, 0, 0]),
            'timestamp': np.array([1581607530, 1581607535, 1581607540, 1581607545])
        }

    def test_synthetize(self):
        self.ws.synthetize([self.job1, self.job2, self.job3])
        self.assertIsInstance(self.ws.workflow, pd.DataFrame)
        self.assertEqual(len(self.ws.workflow), 8)
        self.assertEqual(self.ws.workflow['sumBytesRead'].sum(),
                         self.job1['bytesRead'].sum() +
                         self.job2['bytesRead'].sum() +
                         self.job3['bytesRead'].sum())
        self.assertEqual(self.ws.workflow['sumBytesWritten'].sum(), 
                         self.job1['bytesWritten'].sum() + 
                         self.job2['bytesWritten'].sum() + 
                         self.job3['bytesWritten'].sum())

    def test_to_dict(self):
        self.ws.synthetize([self.job1, self.job2, self.job3])

        result = self.ws.to_dict()

        self.assertIsInstance(result, dict)
        self.assertListEqual(result['timestamp'], 
                             self.ws.workflow['index'].tolist())
        self.assertListEqual(result['bytesRead'], 
                             self.ws.workflow['sumBytesRead'].tolist())
        self.assertListEqual(result['bytesWritten'], 
                             self.ws.workflow['sumBytesWritten'].tolist())



class TestCentralJob(unittest.TestCase):

    def setUp(self):
        self.job1 = {'timestamp': list(range(0,10)), 
                     'bytesRead': list(range(1,11)), 
                     'bytesWritten': list(range(1,11))}
        self.job2 = {'timestamp': list(range(0,20,2)), 
                     'bytesRead': list(range(1,21,2)), 
                     'bytesWritten': list(range(1,21,2))}
        self.job3 = {'timestamp': list(range(0,30,3)), 
                     'bytesRead': list(range(1,31,3)), 
                     'bytesWritten': list(range(1,31,3))}
        self.jobs = [self.job1, self.job2, self.job3]
        self.cj = CentralJob(self.jobs)

    def test_extract_features(self):
        self.cj.extract_features()
        self.assertIsNotNone(self.cj.features)
        self.assertEqual(len(self.cj.features), len(self.jobs))

    # def test_scale_features(self):
    #     self.cj.extract_features()
    #     self.cj.scale_features()
        #self.assertIsNotNone(self.cj.features)
        # Make sure all means are very close to zero and standard deviations very close to 1 (zscore properties)
        #print(self.cj.features)
        # self.assertTrue(np.allclose(self.cj.features.mean(axis=0), 0, atol=1e-6))
        # self.assertTrue(np.allclose(self.cj.features.std(axis=0), 1, atol=1e-6))

    def test_scale_features_minmax(self):
        self.cj.normalization_type = 'minmax'
        self.cj.extract_features()
        self.cj.scale_features()
        self.assertIsNotNone(self.cj.features)
        # Make sure all values are between 0 and 1 (minmax properties)
        self.assertTrue((self.cj.features >= 0).all() and (self.cj.features <= 1).all())

    # def test_find_central_job(self):
    #     self.cj.extract_features()
    #     self.cj.scale_features()
    #     central_job_index = self.cj.find_central_job()
    #     self.assertIsInstance(central_job_index, int)


if __name__ == '__main__':
    unittest.main()
    

