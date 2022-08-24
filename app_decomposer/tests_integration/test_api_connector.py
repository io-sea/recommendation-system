#!/usr/bin/env python
"""
This module proposes integration tests for the API connector module
"""
from __future__ import division, absolute_import, generators, print_function, unicode_literals, \
    with_statement, nested_scopes  # ensure python2.x compatibility

__copyright__ = """
Copyright (C) 2020 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import unittest
import requests
import urllib3
from shutil import rmtree

from tests_integration.iopa_integration_config import get_backend_hostname, get_backend_ipv4_port
from tests_integration.iopa_integration_config import get_keycloak_token
from tests_integration.iopa_integration_config import get_mongo_hostname, get_mongo_ipv4_port
from tests_integration.iopa_integration_config import MongoDB, __MONGO_DUMP__
from app_decomposer.api_connector import request_delegator, check_http_code, TimeSeries, \
    MetaData, MinMax, MinMaxDuration, JobSearch


class TestFunctions(unittest.TestCase):
    """TestCase to test the functions of the module."""

    def setUp(self):
        """Prepare the test suite."""
        self.keycloakToken = get_keycloak_token()
        # To disable useless warnings with requests module on https without certificate
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def test_request_delegator(self):
        """Test the 'request_delegator' standard behavior."""
        api_uri = f"https://{get_backend_hostname()}:{get_backend_ipv4_port()}" \
            "/backend/api/admin/db/stats"
        rqst = request_delegator(requests.get,
                                 api_uri,
                                 headers={'Authorization': f"Bearer {self.keycloakToken}"})

        self.assertIsInstance(rqst, requests.Response)
        conf = rqst.json()
        self.assertIn('dbName', conf)
        self.assertEqual(conf['dbName'], "cmdb_database")

    def test_request_delegator_exception(self):
        """Test the 'request_delegator' error management."""
        api_uri = f"https://{get_backend_hostname()}:{get_backend_ipv4_port()}" \
            "/backend/api/admin/db/stats"

        # Test authentification errors (bad token)
        with self.assertRaises(RuntimeError):
            request_delegator(requests.get,
                              api_uri,
                              headers={'Authorization': 'Bearer foo'})

        # Test url error (post in place of get)
        with self.assertRaises(RuntimeError):
            request_delegator(requests.post,
                              api_uri,
                              headers={'Authorization': f"Bearer {self.keycloakToken}"})

    def test_check_http_code_200(self):
        """Test 'check_http_code' method returns the True if the tested request succeeds."""
        api_uri = f"https://{get_backend_hostname()}:{get_backend_ipv4_port()}" \
            "/backend/api/admin/db/stats"
        code = check_http_code(api_uri,
                               200,
                               headers={'Authorization': f"Bearer {self.keycloakToken}"})
        self.assertTrue(code)

    def test_check_http_code_403(self):
        """Test 'check_http_code' method returns the True if the tested request succeeds."""
        api_uri = f"https://{get_backend_hostname()}:{get_backend_ipv4_port()}" \
            "/backend/api/admin/db/stats"
        code = check_http_code(api_uri, 403)
        self.assertTrue(code)

    def test_check_http_code_ko(self):
        """Test 'check_http_code' method returns the False if the tested request does not
        succeed."""
        api_uri = f"https://fake_url/backend/api/admin/db/stats"
        code = check_http_code(api_uri, 403)
        self.assertFalse(code)


class TestTimeSeries(unittest.TestCase):
    """Test Time Series functionalities."""

    def setUp(self):
        self.keycloakToken = get_keycloak_token()
        # To disable useless warnings with requests module on https without certificate
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self.api_uri = f"https://{get_backend_hostname()}:{get_backend_ipv4_port()}"

    def test_get_data_by_label_no_label(self):
        """Test get_data_by_label method without specifying a set of labels.

        Test that 'get_data_by_label' produces the dict with good labels as keys and numpy array
        as values, without specifying a set of labels. In this case, all available labels and
        corresponding data should be retrieved.

        The data fetched are from the job-666 which were manually defined.
        """
        time_series = TimeSeries(api_uri=self.api_uri,
                                 api_token=f"Bearer {self.keycloakToken}",
                                 object_id='5e78805e3a185aaa08517c1b',
                                 type_series='volume')
        data = time_series.get_data_by_label()
        self.assertIn('timestamp', data)
        self.assertEqual(data['timestamp'].tolist(), [1570636800000, 1570636805000])
        self.assertEqual(data['bytesRead'].tolist(), [138336721, 0])
        self.assertEqual(data['bytesWritten'].tolist(), [53198720, 13681])


class TestMetadata(unittest.TestCase):
    """Test Metadata class functionalities"""

    def setUp(self):
        self.keycloakToken = get_keycloak_token()
        # To disable useless warnings with requests module on https without certificate
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self.api_uri = f"https://{get_backend_hostname()}:{get_backend_ipv4_port()}"

    def test_get_all_metadata(self):
        """Test get_all_metadata method returns the good information for the selected job.

        It just tests that 'get_all_metadata' produces the dict with the main good fields as keys.

        The data fetched are from the job-666 which were manually defined.
        """
        metadata = MetaData(api_uri=self.api_uri,
                            api_token=f"Bearer {self.keycloakToken}",
                            object_id='5e78805e3a185aaa08517c1b')
        data = metadata.get_all_metadata()

        expected_main_meta_field = ['version', 'program', 'cmdLine', 'username', 'jobid', 'jobname']
        # Verify the main field are in the data retrieved
        for field in expected_main_meta_field:
            self.assertIn(field, data)
        # Verify the information corresponds to the good job.
        self.assertEqual(data['jobid'], 666)
        self.assertEqual(data['jobname'], 'job-666')
        self.assertEqual(data['program'], 'program-666')


class TestMinMax(unittest.TestCase):
    """Test MinMax class functionalities"""

    def setUp(self):
        self.keycloakToken = get_keycloak_token()
        # To disable useless warnings with requests module on https without certificate
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def test_get_data(self):
        """Test get_data method returns the good information of the min and max timeseries values
        on all jobs in database.

        Only some min/max timeseries values are tested.
        """
        api_uri = f"https://{get_backend_hostname()}:{get_backend_ipv4_port()}"

        minmax = MinMax(api_uri=api_uri,
                        api_token=f"Bearer {self.keycloakToken}")
        data = minmax.get_data()

        some_expected_field = ['bytesRead', 'bytesWritten', 'IODurationReadRange0',
                               'processCount', 'SbbBbStatSizesReadRange2']
        # Verify the main field are in the data retrieved
        for field in some_expected_field:
            self.assertIn(field, data)
        # Verify all values are a dict with max and min keys
        for field, val in data.items():
            self.assertSetEqual(set(val.keys()), {'max', 'min'})


class TestMinMaxDuration(unittest.TestCase):
    """Test MinMaxDuration class functionalities"""

    def setUp(self):
        self.keycloakToken = get_keycloak_token()
        # To disable useless warnings with requests module on https without certificate
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def test_get_data(self):
        """Test get_data method returns the the min and max duration values on all jobs in database.
        """
        api_uri = f"https://{get_backend_hostname()}:{get_backend_ipv4_port()}"

        minmax_duration = MinMaxDuration(api_uri=api_uri,
                                         api_token=f"Bearer {self.keycloakToken}")
        data = minmax_duration.get_data()
        self.assertDictEqual(data, {'min': 5, 'max': 604800})

class TestJobSearch(unittest.TestCase):
    """ Test JobSearch functionalities """

    def setUp(self):
        self.keycloakToken = get_keycloak_token()
        # To disable useless warnings with requests module on https without certificate
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self.api_uri = f"https://{get_backend_hostname()}:{get_backend_ipv4_port()}"

        self.db = MongoDB(host=get_mongo_hostname(),
                          port=get_mongo_ipv4_port()).database

    def test_job_search_no_parameters(self):
        """ Test job search without parameters returns all the jobs from database. """
        job_search = JobSearch(self.api_uri, f"Bearer {self.keycloakToken}")

        total_jobs = job_search.total_jobs
        total_filter = job_search.total_filter

        db_count = self.db.JobItem.count_documents({})

        self.assertEqual(total_jobs, db_count)
        self.assertEqual(total_filter, db_count)

    def test_job_search_filter(self):
        """ Test job search with filter returns only the requested jobs. """
        job_filter = {
            'username': {
                'contains': 'phamtt'
            }
        }
        job_search = JobSearch(self.api_uri,
                               f"Bearer {self.keycloakToken}",
                               job_filter=job_filter,
                               limit=2,
                               colorder='duration',
                               order='desc')

        total_filter = job_search.total_filter
        count = job_search.count
        ids = job_search.job_objids_list

        db_count = self.db.JobItem.count_documents({'username': 'phamtt'})
        expected_ids = ['5d52cd5c609bdd51a565a633',
                        '5d4a8d17609bdd51a564aca1']

        self.assertEqual(total_filter, db_count)
        self.assertEqual(count, len(expected_ids))
        self.assertEqual(ids, expected_ids)

class TestGetJobData(unittest.TestCase):
    """Test functionnalities needed for job decomposer to get job data."""

    def setUp(self):
        self.keycloakToken = get_keycloak_token()
        # To disable useless warnings with requests module on https without certificate
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self.api_uri = f"https://{get_backend_hostname()}:{get_backend_ipv4_port()}"


    def test_job_search_by_jobid(self):
        """Test job retrieval when having only the job id that is visible in IOI."""
        job_search = JobSearch(self.api_uri,
                               f"Bearer {self.keycloakToken}",
                               job_filter={"jobid": {"contains": "3336"}})

        self.assertEqual(job_search.job_objids_list[0], "5d7bd95e0187784d71fbbf38")

    def test_get_nodecount_metadata(self):
        """Test nodecount method returns the good information for the selected job.
        """
        metadata = MetaData(api_uri=self.api_uri,
                            api_token=f"Bearer {self.keycloakToken}",
                            object_id='5e78805e3a185aaa08517c1b')
        data = metadata.get_all_metadata()
        # Verify the main field are in the data retrieved
        self.assertIn("nodeCount", data)
        # Verify the information corresponds to the good job.
        self.assertEqual(data["nodeCount"], 1)

    def test_get_job_timeseries(self):
        """Test method that gathers all needed data from slurm jobid necessary for the app decomposer."""
        target_jobid = 666
        job_search = JobSearch(self.api_uri,
                               f"Bearer {self.keycloakToken}",
                               job_filter={"jobid": {"contains": str(target_jobid)}})
        object_id = job_search.job_objids_list[0]
        metadata = MetaData(api_uri=self.api_uri,
                            api_token=f"Bearer {self.keycloakToken}",
                            object_id=object_id)
        data = metadata.get_all_metadata()
        node_count = data["nodeCount"]
        time_series = TimeSeries(api_uri=self.api_uri,
                                 api_token=f"Bearer {self.keycloakToken}",
                                 object_id=object_id,
                                 type_series='volume')
        data = time_series.get_data_by_label()
        self.assertEqual(node_count, 1)
        self.assertEqual(object_id, "5e78805e3a185aaa08517c1b")
        self.assertIn('timestamp', data)
        self.assertEqual(data['timestamp'].tolist(), [1570636800000, 1570636805000])


if __name__ == '__main__':
    unittest.main(verbosity=2)
