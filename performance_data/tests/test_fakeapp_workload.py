#!/usr/bin/env python

"""Tests for `performance_data` package."""


import os
import unittest
from performance_data.fakeapp_workload import FakeappWorkload
from performance_data import cli


class TestPerformanceData(unittest.TestCase):
    """Tests for `performance_data` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_update_sbatch_template(self):
        """Test something."""
        workload = FakeappWorkload(volume=1e6, mode="read",
                                   io_pattern="rand",
                                   io_size=4e3,
                                   nodes=1,
                                   target_tier="/fsiof/mimounis/tmp",
                                   accelerator=False,
                                   ioi=False)
        workload.write_sbatch_file()
        self.assertTrue(os.path.isfile(workload.sbatch_file))
        os.remove(workload.sbatch_file)

    def test_cmdline_sbatch_no_clean(self):
        """Test something."""
        workload = FakeappWorkload(volume=0.5e9, mode="read",
                                   io_pattern="rand",
                                   io_size=4e3,
                                   nodes=1,
                                   target_tier="/fsiof/mimounis/tmp",
                                   accelerator=False,
                                   ioi=False)
        workload.write_sbatch_file()
        job_time = workload.run_sbatch_file()
        self.assertTrue(isinstance(job_time, float))
        self.assertTrue(os.path.isfile(workload.sbatch_file))

    def test_cmdline_sbatch_with_clean(self):
        """Test something."""
        workload = FakeappWorkload(volume=0.5e9, mode="read",
                                   io_pattern="rand",
                                   io_size=4e3,
                                   nodes=1,
                                   target_tier="/fsiof/mimounis/tmp",
                                   accelerator=False,
                                   ioi=False)
        workload.write_sbatch_file()
        job_time = workload.run_sbatch_file(clean=True)
        self.assertTrue(isinstance(job_time, float))
        self.assertFalse(os.path.isfile(workload.sbatch_file))
