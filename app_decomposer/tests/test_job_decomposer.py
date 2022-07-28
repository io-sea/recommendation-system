import unittest
import time
import numpy as np
import simpy
from loguru import logger
import sklearn

from app_decomposer.job_decomposer import JobDecomposer

class TestJobDecomposer(unittest.TestCase):
    """Test that the app decomposer follows some pattern."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def test_get_job_timeseries_from_file_nojobid(self):
        """Test if getting timeseries from jobs data dumped into files works well."""
        jd = JobDecomposer()
        self.assertList(jd.read_signal.flatten().tolist())

    def test_get_job_timeseries_from_file_no_jobid(self):
        """Test if getting timeseries from jobs data dumped into files works well."""
        jd = JobDecomposer(job_id=2301)
        print(jd.read_signal.flatten().tolist())

