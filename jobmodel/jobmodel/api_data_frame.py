import sys
import requests, json
import numpy as np
import pandas as pd
from ioanalyticstools.api_connector import TimeSeries
from clust_pipeline.datafetcher import API_DICT_TS
import matplotlib.pyplot as plt


class ApiDataFrame:
    def __init__(self, api_uri="http://localhost:7688",
                 api_token="3f9f66fbdb1f2982980521d1ca9ffb9aaf2af5c4"):
        self.api_uri = api_uri
        self.api_token = api_token
        self.jobids = None
        self.dataframe = pd.DataFrame()

    def get_job_list(self):
        # getting all jobs ids in database
        url = f"{self.api_uri}/backend/api/job/search?apikey={self.api_token}"
        r = requests.get(url=url)
        joblist = r.json()['data']
        self.jobids = []
        for job in joblist:
            self.jobids.append(job['id'])

    def job_dataframe(self, jobid):
        # appending the dataframe for 1 job
        ts_type = list(API_DICT_TS.keys())[0]
        data = TimeSeries(self.api_uri, self.api_token, jobid, ts_type).get_data_by_label()
        df = pd.DataFrame(data)
        for ts_type in list(API_DICT_TS.keys())[1:]:
            # print(ts_type + "\n")
            ts = TimeSeries(self.api_uri, self.api_token, jobid, ts_type)
            data = ts.get_data_by_label()
            for ts_serie in list(data.keys())[1:]:
                df[ts_serie] = data[ts_serie][:df.shape[0]]
        return df

    def save_to_csv(self, filepath):
        self.dataframe.to_csv(filepath + '.csv')

    def save_to_pkl(self, filepath):
        self.dataframe.to_pickle(filepath + '.pkl')

    @staticmethod
    def progress(count, total, suffix=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
        sys.stdout.flush()

    def get_db_dataframe(self, njobs=None, filepath=None):
        # concat all jobs dfs into dataframe attribute
        self.get_job_list()
        njobs = min(len(self.jobids), njobs) if njobs is not None else len(self.jobids)
        print(f"{njobs} jobs will be dumped")
        count = 0
        for jobid in self.jobids[:njobs]:
            count += 1
            self.progress(count, njobs)
            df = self.job_dataframe(jobid)
            self.dataframe = self.dataframe.append(df)
        if filepath is not None:
            self.save_to_csv(filepath)
            #self.save_to_pkl(filepath)


