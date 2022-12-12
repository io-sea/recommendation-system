#!/usr/bin/env python

"""Tests for `performance_data` package."""


import os
import unittest
from performance_data.fakeapp_workload import FakeappWorkload
from performance_data import cli


class TestFakeappWorkload(unittest.TestCase):
    """Tests for `performance_data` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.phase = dict(read_volume=1e8, read_io_pattern="stride", read_io_size=1e4, 
                          write_volume=1e8, write_io_pattern="uncl", write_io_size=1e4, 
                          nodes=1)
        self.target_tier="/fsiof/phamtt/tmp"

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_update_sbatch_template(self):
        """Test something."""
        
        workload = FakeappWorkload(self.phase,
                                   self.target_tier, 
                                   accelerator=False,
                                   ioi=False)
        workload.write_sbatch_file()
        self.assertTrue(os.path.isfile(workload.sbatch_file))
        os.remove(workload.sbatch_file)

    def test_cmdline_sbatch_no_clean(self):
        """Test something."""
        workload = FakeappWorkload(self.phase,
                                   self.target_tier, 
                                   accelerator=False,
                                   ioi=False)
        workload.write_sbatch_file()
        job_time = workload.run_sbatch_file()
        self.assertTrue(isinstance(job_time, float))
        self.assertTrue(os.path.isfile(workload.sbatch_file))

    def test_cmdline_sbatch_with_clean(self):
        """Test something."""
        workload = FakeappWorkload(self.phase,
                                   self.target_tier, 
                                   accelerator=False,
                                   ioi=False)
        workload.write_sbatch_file()
        job_time = workload.run_sbatch_file(clean=True)
        self.assertTrue(isinstance(job_time, float))
        self.assertFalse(os.path.isfile(workload.sbatch_file))

    def test_cmdline_sbatch_with_SBB(self):
        """Test something."""
        workload = FakeappWorkload(self.phase,
                                   self.target_tier, 
                                   accelerator=True,
                                   ioi=False)
        workload.write_sbatch_file()
        job_time = workload.run_sbatch_file(clean=True)
        self.assertTrue(isinstance(job_time, float))
        self.assertFalse(os.path.isfile(workload.sbatch_file))

    def test_get_data(self):
        """Test something."""
        workload = FakeappWorkload(self.phase,
                                   self.target_tier, 
                                   accelerator=True,
                                   ioi=False)

        elapsed_time, bw = workload.get_data()
        self.assertTrue(isinstance(elapsed_time, float))

