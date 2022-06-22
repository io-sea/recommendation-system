import os
import pymongo
import bson
import random
import pandas as pd


def get_dump_path(dir_list=["dump", "cmdb_database"], filename="JobItem.bson"):
    """Get the path to the JobItem.bson file.

    Args:
        dir_list (list): list of folder names that leads the the JobItem file. Defaults to ["dump", "cmdb_database"]
        filename (string): file to scrap, defaults to "JobItem.bson", but could be "FileIOSummaryGw.bson" or "ProcessSummary.bson"
    """
    # indicate here the path to the bson objects, starting from local
    # "dump", "cmdb_database"
    return os.path.join(os.getcwd(), *dir_list, filename)


def get_random_jobs(filepath, number_of_jobs=3):
    """Select a random set of jobs from the JobItem.bson pathfile.

    Args:
        filepath (os.path): path to JobItem.bson file.
        
    Returns:
        job_ids (list): list of jobids to extract.
    """
    with open(filepath,'rb') as f:
        # list of metadata jobs like:
        # {'_id': ObjectId('5e14cdce928715c8ce939865'), 'jobid': 1775, 'version': '1.3-Bull.9', 'autoPurge': False, 'uid': 16214, 'username': 'roberts', 'startTime': datetime.datetime(2020, 1, 7, 18, 28, 20), 'program': 'job_1578321465_shaman_header.sbatch',
        jobitems = random.choices(bson.decode_all(f.read()), k = number_of_jobs)
    
    return [jobitems[i]["jobid"] for i in range(number_of_jobs)]
    


def get_io_dataframe(job_id):
    # filepath = os.path.join(os.getcwd(), "dump", "cmdb_database", "FileIOSummaryGw.bson")
    df = pd.DataFrame(columns=["timestamp", "bytesRead", "bytesWritten"])
    for filepath in [get_dump_path(filename="FileIOSummary.bson"),
                     get_dump_path(filename="FileIOSummaryGw.bson")]:
        
        with open(filepath,'rb') as f:
            # Get a list of timestamps elements for all jobs
            # {'_id': ObjectId('591ac4e02db96207ae2b5e92'), 'jobid': 2665, 'hostname': 'lima16.bullx', 'timeFrame': datetime.datetime(2017, 5, 16, 9, 22, 35), 'bytesRead': 0, 'bytesWritten': 0, 'filesRW': 0, 'filesRO': 0, 'filesWO': 0, 'filesCreated': 0, 'filesDeleted': 0, 'accessRandRead': 0, 'accessSeqRead': 0, 'accessStrRead': 0, 'accessUnclRead': 0, 'accessRandWrite': 0, 'accessSeqWrite': 0, 'accessStrWrite': 0, 'accessUnclWrite': 0}
            jdata = bson.decode_all(f.read())

        # Get specific ts from specific jobid with ts available in bson file
        x = [dp["bytesRead"] for dp in jdata if dp["jobid"]==job_id]
        y = [dp["bytesWritten"] for dp in jdata if dp["jobid"]==job_id]
        t = [dp["timeFrame"].timestamp() for dp in jdata if dp["jobid"]==job_id]
    
        df = pd.concat([df, pd.DataFrame({"timestamp": t,
                                          "bytesRead": x,
                                          "bytesWritten": y})])
    return df.drop_duplicates().sort_values(by=['timestamp'])

# df = pd.DataFrame(jdata)
# print(df.describe())
# print(df.head(5))

# filepath = os.path.join(os.getcwd(), "dump", "cmdb_database", "ProcessSummary.bson")
# with open(filepath,'rb') as f:
#     # Get a list of timestamps elements for all jobs
#     # {'_id': ObjectId('591ac4e02db96207ae2b5e92'), 'jobid': 2665, 'hostname': 'lima16.bullx', 'timeFrame': datetime.datetime(2017, 5, 16, 9, 22, 35), 'bytesRead': 0, 'bytesWritten': 0, 'filesRW': 0, 'filesRO': 0, 'filesWO': 0, 'filesCreated': 0, 'filesDeleted': 0, 'accessRandRead': 0, 'accessSeqRead': 0, 'accessStrRead': 0, 'accessUnclRead': 0, 'accessRandWrite': 0, 'accessSeqWrite': 0, 'accessStrWrite': 0, 'accessUnclWrite': 0}
#     jdata = bson.decode_all(f.read())

# print((jdata[0]))
# print((jdata[1]))

# {'_id': ObjectId('591ac4e02db96207ae2b5e91'), 'jobid': 2665, 'hostname': 'lima16.bullx', 'timeFrame': datetime.datetime(2017, 5, 16, 9, 22, 35), 'processCount': 1, 'ioActiveProcessCount': 0}
# {'_id': ObjectId('591ac52b2db96207ae2b5e97'), 'jobid': 2667, 'hostname': 'lima16.bullx', 'timeFrame': datetime.datetime(2017, 5, 16, 9, 23, 50), 'processCount': 2, 'ioActiveProcessCount': 0}

def get_process_dataframe(job_id):
    # filepath = os.path.join(os.getcwd(), "dump", "cmdb_database", "FileIOSummaryGw.bson")
    df = pd.DataFrame(columns=["timestamp", "processCount", "ioActiveProcessCount"])
    for filepath in [get_dump_path(filename="ProcessSummary.bson"),
                     get_dump_path(filename="ProcessSummaryGw.bson")]:
        
        with open(filepath,'rb') as f:
            # Get a list of timestamps elements for all jobs
            # {'_id': ObjectId('591ac4e02db96207ae2b5e92'), 'jobid': 2665, 'hostname': 'lima16.bullx', 'timeFrame': datetime.datetime(2017, 5, 16, 9, 22, 35), 'bytesRead': 0, 'bytesWritten': 0, 'filesRW': 0, 'filesRO': 0, 'filesWO': 0, 'filesCreated': 0, 'filesDeleted': 0, 'accessRandRead': 0, 'accessSeqRead': 0, 'accessStrRead': 0, 'accessUnclRead': 0, 'accessRandWrite': 0, 'accessSeqWrite': 0, 'accessStrWrite': 0, 'accessUnclWrite': 0}
            jdata = bson.decode_all(f.read())

        # Get specific ts from specific jobid with ts available in bson file
        x = [dp["processCount"] for dp in jdata if dp["jobid"]==job_id]
        y = [dp["ioActiveProcessCount"] for dp in jdata if dp["jobid"]==job_id]
        t = [dp["timeFrame"].timestamp() for dp in jdata if dp["jobid"]==job_id]
    
        df = pd.concat([df, pd.DataFrame({"timestamp": t,
                                          "processCount": x,
                                          "ioActiveProcessCount": y})])
    return df.drop_duplicates().sort_values(by=['timestamp'])


if __name__ == '__main__':
    job_list = get_random_jobs(filepath=get_dump_path(), number_of_jobs=5)
    print(job_list)
    df_io = get_io_dataframe(job_list[0])
    print(df_io)
    df_pr = get_process_dataframe(job_list[0])
    print(df_pr)
    df_job = df_io.merge(df_pr)
    print(df_job)
    