#!/usr/bin/env python

"""Tests for `performance_data` package."""


import os
from os.path import dirname
import unittest
from unittest.mock import patch
import sys
from performance_data.fakeapp_workload import FakeappWorkload
from app_decomposer.utils import convert_size
from performance_data.performance_data import PhaseData, DataTable
from performance_data.fakeapp_workload import FakeappWorkload as Workload
from performance_data import cli
import pandas as pd
import numpy as np


class TestFakeappWorkload(unittest.TestCase):
    """Tests for `performance_data` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.phase = dict(read_volume=1e8, read_io_pattern="stride", read_io_size=1e4,
                          write_volume=1e8, write_io_pattern="uncl", write_io_size=1e4,
                          nodes=1)
        self.target_tier="/fsiof/mimounis/tmp"

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_init_fakeapp_workload(self):
        """Initialization of the class"""

        workload = FakeappWorkload(self.phase,
                                   self.target_tier,
                                   accelerator=False,
                                   ioi=False)

    def test_update_sbatch_template(self):
        """Initialization of the class and sbatch write"""

        workload = FakeappWorkload(self.phase,
                                   self.target_tier,
                                   accelerator=False,
                                   ioi=False)
        workload.write_sbatch_file()
        self.assertTrue(os.path.isfile(workload.sbatch_file))
        os.remove(workload.sbatch_file)

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires linux")
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

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires linux")
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

    @patch.object(FakeappWorkload, 'run_sbatch_file')
    def test_cmdline_sbatch_launched(self, mock_run_sbatch_file):
        """Test sbatch execution is called."""
        mock_run_sbatch_file.return_value = 1.36
        workload = FakeappWorkload(self.phase,
                                   self.target_tier,
                                   accelerator=False,
                                   ioi=False)
        workload.write_sbatch_file()
        job_time = workload.run_sbatch_file(clean=True)
        self.assertTrue(isinstance(job_time, float))
        self.assertEqual(job_time, 1.36)

    @patch.object(FakeappWorkload, 'run_sbatch_file')
    def test_cmdline_sbatch_with_SBB(self, mock_run_sbatch_file):
        """Test sbatch execution is called with SBB."""
        mock_run_sbatch_file.return_value = 1.36
        workload = FakeappWorkload(self.phase,
                                   self.target_tier,
                                   accelerator=True,
                                   ioi=False)
        workload.write_sbatch_file()
        job_time = workload.run_sbatch_file(clean=True)
        self.assertTrue(isinstance(job_time, float))

    @patch.object(FakeappWorkload, 'run_sbatch_file')
    def test_get_data(self, mock_run_sbatch_file):
        """Test get data."""
        mock_run_sbatch_file.return_value = 2.
        workload = FakeappWorkload(self.phase,
                                   self.target_tier,
                                   accelerator=True,
                                   ioi=False)

        elapsed_time, bw = workload.get_data()
        self.assertTrue(isinstance(elapsed_time, float))
        self.assertEqual(elapsed_time, 2)
        self.assertAlmostEqual(bw, 1e8)

@unittest.skipIf(sys.platform.startswith("linux"), "skip on Linux")
class TestBwRepeatability(unittest.TestCase):
    """Tests for `performance_data` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.dataset_file = os.path.join(dirname(dirname(os.path.abspath(__file__))),
                                         "performance_data", "dataset",
                                         "performance_model_dataset_RW.csv")
        self.dataset_stats = os.path.join(dirname(dirname(os.path.abspath(__file__))),
                                         "performance_data", "dataset",
                                         "performance_model_dataset_stats_lfs_sbb_rw.csv")
        self.dataset = pd.read_csv(self.dataset_file)
        self.phases = self.dataset.to_dict('records')
        self.target = dict(lfs="/fsiof/phamtt/tmp")
        self.ioi = False

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_get_data(self):
        """Launch stats study on bandwidth."""
        print(self.dataset_file)
        df = pd.read_csv(self.dataset_file)
        print(df)
        # stat_df = pd.DataFrame(columns=list(df.columns) + ["wl_volume", "lfs_t", "lfs_bw", "lfs_str__bw", "lfs_avg", "lfs_std"],  index=list(df.index))
        #stat_df = pd.DataFrame()
        phases = df.to_dict('records')
        print(phases)
        lite = True
        phases_stats = []
        for phase in phases:
            print(phase)
            results = {}
            latencies = 0
            volumes = 0

            io_phase=dict(read_volume=phase["read_volume"], read_io_pattern=phase["read_io_pattern"], read_io_size=phase["read_io_size"],
                          write_volume=phase["write_volume"], write_io_pattern=phase["write_io_pattern"], write_io_size=phase["write_io_size"],
                          nodes=phase["nodes"])
            workload = Workload(io_phase, target_tier=self.target["lfs"],
                                accelerator=True, ioi=self.ioi)
            phase_stat = phase
            phase_stat["wl_volume"] = convert_size(phase["read_volume"]+phase["write_volume"])

            for _ in range(10):
                elapsed_time, bw = workload.get_data()
                results.setdefault("duration", []).append(elapsed_time)
                results.setdefault("bandwidth", []).append(bw)
                results.setdefault("str_bw", []).append(convert_size(bw))
                print(f"Elaplsed time: {elapsed_time} | bandwidth: {convert_size(bw)}")

            phase_stat["lfs_t"] = results["duration"]
            phase_stat["lfs_bw"] = results["bandwidth"]
            phase_stat["lfs_str_bw"] = results["str_bw"]
            phase_stat["lfs_avg"] = np.mean(phase_stat["lfs_bw"])
            phase_stat["lfs_std"] = np.std(phase_stat["lfs_bw"])
            phase_stat["std/avg"] = 100*np.std(phase_stat["lfs_bw"])/np.mean(phase_stat["lfs_bw"])

            phases_stats.append(phase_stat)
            df2 = pd.DataFrame.from_dict(phases_stats)
            df2.to_csv(self.dataset_stats, index=False)
        # # print(results)
        # self.assertTrue(True)

        print(df2)


