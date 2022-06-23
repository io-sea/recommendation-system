import time
import numpy as np
import requests
from clust_pipeline.datafetcher import API_DICT_TS
from clust_pipeline.datafetcher.api_group_connector import ApiGroupConnector
from ioanalyticstools.api_connector import request_delegator
import pickle


def get_all_jobs(api_uri, api_token, n_jobs):
    """Fetch data from IOI database, exposed by API on :
    inputs :
        api_uri :   base_uri of the api + path
        api_token : apikey
        n_jobs :    number of jobs to extract,
                    n : n jobs
                    0 : all the jobs available
    outputs :
        job_id_list : a list of job ids
        data_list : whole json data structure for each job"""
    # Url request elements
    base_uri = api_uri
    api_url = base_uri + '/backend/api/job/search'
    params = {'apikey': api_token}
    # Encoding the request url
    rqst_url = requests.Request('GET', api_url, params=params).prepare().url
    # Passing the request
    rqst = request_delegator(requests.get, rqst_url)
    data_list = rqst.json()['data']
    n_max_jobs = len(data_list)
    # Preparing list of job id
    job_id_list = []
    if isinstance(n_jobs, int):
        if n_jobs >= 1:
            job_id_list = [data_list[i]['id'] for i in range(np.min([n_jobs, n_max_jobs]))]
        else:
            job_id_list = [data_list[i]['id'] for i in range((n_max_jobs))]

    else:
        job_id_list = [data_list[i]['id'] for i in range((n_max_jobs))]

    return job_id_list, data_list


def select_from_job_list(data_list, entry, value):
    """ Fetch job_id  where :
    self.data_list[...][entry] == value
    inputs:
        self.data_list : json load of API response
        entry : dict key on which selection is done
        value : dict value to be selected
    outputs :
        list of job_ids from self.data_list['jobid']"""
    selected_jobs = []
    for j in range(len(data_list)):
        if data_list[j][entry] == value:
            selected_jobs.append(data_list[j]['id'])

    # logging.info(f'Job selection : {len(self.selected_jobs)} jobs are selected from data')
    return selected_jobs


def get_classified_jobs_from_api():
    api_uri = 'http://localhost:8080'
    api_token = 'bb8fcd04b9eb9cf6123c43def49cc66cbb43b12f'
    job_ids, data_list = get_all_jobs(api_uri, api_token, 0)
    job_name_list = [f'job{i}-classification{j}' for j in [2, 3, 4] for i in [1]]
    selected_jobs = []
    truth_labels = []
    counter = 0
    for job_list in job_name_list:
        selected_job_group = select_from_job_list(data_list, 'jobname', job_list)
        # take only first k jobs
        k = 5
        k = np.min([k, len(selected_job_group)])
        selected_jobs += selected_job_group[:k]
        truth_labels += [counter for i in range(k)]
        counter += 1

    # ts_list = ['bytesRead', 'bytesWritten', 'filesCreated', 'filesDeleted', 'filesRO', 'filesWO', 'filesRW']
    SMALL_DICT_TS = {'volume': ['bytesRead', 'bytesWritten']}
    ts_list = []
    current_dict = API_DICT_TS
    for entry, ts_type in current_dict.items():
        for label in current_dict[entry]:
            ts_list.append(label)
    t0 = time.time()
    return selected_jobs, ts_list

def get_job_data(selected_jobs, ts_list):
    api_uri = 'http://localhost:8080'
    api_token = 'bb8fcd04b9eb9cf6123c43def49cc66cbb43b12f'
    data = ApiGroupConnector(api_uri, api_token, selected_jobs, ts_list).get_data()
    data_connector = data['timeseries']
    jobs = []
    job_series = dict()
    for job, job_data in data_connector.items():
        jobs.append(job)
        print(job)
        job_series[job] = dict()
        for group_name, connector in job_data.items():
            for ts_name, ts_data in connector().items():
                job_series[job][ts_name] = ts_data
    return job_series

def get_any_job_data():
    api_uri = 'http://localhost:8080'
    api_token = 'bb8fcd04b9eb9cf6123c43def49cc66cbb43b12f'
    job_ids, data_list = get_all_jobs(api_uri, api_token, 10)
    ts_list = ['bytesRead', 'bytesWritten']
    data = ApiGroupConnector(api_uri, api_token, job_ids, ts_list).get_data()
    print(job_ids)
    print(type(job_ids))
    data_connector = data['timeseries']
    jobs = []
    job_series = dict()
    for job, job_data in data_connector.items():
        jobs.append(job)
        print(job)
        job_series[job] = dict()
        for group_name, connector in job_data.items():
            for ts_name, ts_data in connector().items():
                job_series[job][ts_name] = ts_data

    return job_series


if __name__ == '__main__':
    #job_data = get_any_job_data()
    jobs, ts = get_classified_jobs_from_api()
    job_series = get_job_data(jobs, ts)
    with open('job_data.pkl', 'wb') as fp:
        pickle.dump(job_series, fp)
