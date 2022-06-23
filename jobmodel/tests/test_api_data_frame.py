import unittest
import requests, json
import numpy as np
import pandas as pd
from ioanalyticstools.api_connector import TimeSeries
from clust_pipeline.datafetcher import API_DICT_TS
import matplotlib.pyplot as plt
import os
from jobmodel.api_data_frame import ApiDataFrame


class TestRegression(unittest.TestCase):
    def setUp(self):
        self.adf = ApiDataFrame()

    def test_job_list(self):
        self.adf.get_job_list()
        self.assertIsInstance(self.adf.jobids, list)

    def test_job_dataframe(self):
        self.adf.get_job_list()
        jobid = self.adf.jobids[0]
        df = self.adf.job_dataframe(jobid)
        print(df.shape)
        print(df.head())
        self.assertIsInstance(df, pd.DataFrame)

    def test_get_db_dataframe(self):
        filepath = os.path.join(os.getcwd(), 'data\job_data')
        self.adf.get_db_dataframe(njobs=3, filepath=filepath)
        self.assertIsInstance(self.adf.dataframe, pd.DataFrame)
        self.assertTrue(self.adf.dataframe.size > 0)


if __name__ == '__main__':
    unittest.main()
