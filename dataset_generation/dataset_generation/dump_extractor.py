import os
import pymongo
import bson
import random
import pandas as pd


class DumpExtractor:
    """This class aims to provide a simple and consistant way to extract job data from a rough dump of a mongodb database.
    This make casting timeseries faster and easier without need to install IOI product and connect to API, login to BIRD in order to make local http requests and finally get timeseries.
    This way will also circumvent the need of updating old job db schema to a newer one.
    """
    def __init__(self, prefix=None, absolute_dump_path=None, target_folder=None, jobs=1):
        """Initiates the DumpExtractor

        Args:
            prefix (str): title to prefix the dataset folder, if specified
            absolute_dump_path (string): the absolute path to the dataset directory
            target_folder (string): the path to the folder where extracted jobs will be generated
            jobs (int or list): number or list of jobs to be extracted
        """
        self.absolute_path = absolute_dump_path or os.getcwd()
        self.target_folder = target_folder
        if not prefix:
            prefix = "_".join(self.absolute_path.split(os.sep)[-1:])
        self.prefix = prefix
        self.n_jobs = 0
        self.jobs = []
        if isinstance(jobs, int):
            self.n_jobs = jobs
        elif isinstance(jobs, list):
            self.jobs = jobs
        
    def get_bson_path(self, dir_list=["dump", "mat_db"], filename="JobItem.bson"):
        """Get the path to the JobItem.bson file or any specified bson path.

        Args:
            dir_list (list): list of folder names that leads the the JobItem file. Defaults to      ["dump", "cmdb_database"]
            filename (string): file to scrap, defaults to "JobItem.bson", but could be "FileIOSummaryGw.bson" or "ProcessSummary.bson"
        """
        # indicate here the path to the bson objects, starting from local
        # "dump", "cmdb_database"
        #return os.path.join(os.getcwd(), *dir_list, filename)
        return os.path.join(self.absolute_path, *dir_list, filename)


    def get_random_jobs(self, filepath, number_of_jobs=3):
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
    

    def get_io_dataframe(self, job_id):
        """Extract IO timeseries into a dataframe for a specific job_id.

        Args:
            job_id (int): s specific job_id.

        Returns:
            dataframe: dataframe having timestamp, bytesRead and bytesWritten columns for job_id.
        """
        # filepath = os.path.join(os.getcwd(), "dump", "cmdb_database", "FileIOSummaryGw.bson")
        df = pd.DataFrame(columns=["timestamp", "bytesRead", "bytesWritten"])
        for filepath in [self.get_bson_path(filename="FileIOSummary.bson"),
                         self.get_bson_path(filename="FileIOSummaryGw.bson")]:
            
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


    def get_process_dataframe(self, job_id):
        # filepath = os.path.join(os.getcwd(), "dump", "cmdb_database", "FileIOSummaryGw.bson")
        df = pd.DataFrame(columns=["timestamp", "processCount", "ioActiveProcessCount"])
        for filepath in [self.get_bson_path(filename="ProcessSummary.bson"),
                         self.get_bson_path(filename="ProcessSummaryGw.bson")]:
            
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
    
    def target_file(self, job_id):
        """Defines the path to the target file where to store the csv.
        
        Args:
            job_id (int): job id to specify a unique filename
            target_folder (os.path): folder where to put the job file
            
        Returns:
            pathfile (os.path): complete path to the file.
        """
        # "dataset_" + self.prefix + "_" + str(job_id) + ".csv"
        if not self.target_folder:
            self.target_folder = os.path.join(os.getcwd(), "dataset_" + self.prefix)
        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)
        return os.path.join(self.target_folder, f"job_{job_id}.csv")
        
    def extract_job(self):
        # if number of jobs are given, then choose randomly
        job_item_file = self.get_bson_path()
        if self.n_jobs:
            list_of_jobs = self.get_random_jobs(job_item_file, number_of_jobs=self.n_jobs)
        elif not self.n_jobs and self.jobs:
            list_of_jobs = self.jobs
            
        for i, job_id in enumerate(list_of_jobs):
            print(f"\n Extracting job {job_id} step #{i+1}/{len(list_of_jobs)} from dataset {self.prefix} -> {self.target_file(job_id)}")
            df_io = self.get_io_dataframe(job_id)
            df_pr = self.get_process_dataframe(job_id)
            df_job = df_io.merge(df_pr)
            print(df_job)
            df_job.to_csv(self.target_file(job_id))
            
            

if __name__ == '__main__':
    # take randomly 3 jobs and dumps them in separate csv file
    # dump_extractor = DumpExtractor(jobs = 3)
    
    # take a list of job_ids and dumps them in separate csv file
    # dump_extractor = DumpExtractor(jobs = [3800, 1310])
    # dump_extractor = DumpExtractor(absolute_dump_path = "/home_nfs/mimounis/iosea-wp3-recommandation-system/dataset_generation/dataset_name", jobs = [3800, 1310])
    
    # specifying the target_folder and number of jobs to extract randomly
    # dump_extractor = DumpExtractor(absolute_dump_path = "/home_nfs/mimounis/iosea-wp3-recommandation-system/dataset_generation/dataset_name",
    #                                target_folder="/home_nfs/mimounis/iosea-wp3-recommandation-system/hoops",
    #                                jobs = 2)
    #dump_extractor = DumpExtractor(absolute_dump_path = "/home_nfs/mimounis/iosea-wp3-recommandation-system/dataset_generation/dataset_name", jobs = 2)
    #dump_extractor.extract_job()
    
    # Genji
    # dump_extractor = DumpExtractor(absolute_dump_path = "/fs1/PUBLIC/bds_dm_datasets/general_purpose/genji_dumps/genji_IOI_BD260919", jobs = 2)
    # dump_extractor.extract_job()
    
    # Gysela
    # dump_extractor = DumpExtractor(absolute_dump_path = "/fs1/PUBLIC/bds_dm_datasets/general_purpose/spartan_dumps/gysela", jobs = 12)
    # dump_extractor.extract_job()
    
    # Valid
    # dump_extractor = DumpExtractor(absolute_dump_path = "/fs1/PUBLIC/bds_dm_datasets/general_purpose/valid_dumps", jobs = 12)
    # dump_extractor.extract_job()
    
    # Sophie shaman
    # dump_extractor = DumpExtractor(absolute_dump_path = "/fs1/PUBLIC/bds_dm_datasets/clustering_experiences/shaman_experiment_clean/sop_db", jobs = 12)
    # dump_extractor.extract_job()
    
    # Othmane
    # dump_extractor = DumpExtractor(absolute_dump_path = "/fs1/PUBLIC/bds_dm_datasets/clustering_experiences/othmane_internship_clean/oth_db", jobs = 12)
    # dump_extractor.extract_job()
    
    # Mathieu
    # dump_extractor = DumpExtractor(absolute_dump_path = "/fs1/PUBLIC/bds_dm_datasets/clustering_experiences/mathieu_internship_clean/mat_db", jobs = 12)
    # dump_extractor.extract_job()
    
