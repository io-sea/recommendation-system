#!/usr/bin/env python

"""Tests for `performance_data` package."""


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
                                   target_tier="nfs",
                                   accelerator=False,
                                   ioi=False)
        temp_file = workload.updated_sbatch_template()
        print(temp_file)

    def test_cmdline_sbatch(self):
        """Test something."""
        workload = FakeappWorkload(volume=1e6, mode="read",
                                   io_pattern="rand",
                                   io_size=4e3,
                                   nodes=1,
                                   target_tier="nfs",
                                   accelerator=False,
                                   ioi=False)
        temp_file = workload.updated_sbatch_template()
        cmd = workload.run_sbatch(temp_file, ioi=False)
        self.assertEquals(cmd, "sbatch --wait /home_nfs/mimounis/iosea-wp3-recommandation-system/performance_data/performance_data/defaults/mod_sbatch.sbatch")

    # def test_command_line_interface(self):
    #     """Test the CLI."""
    #     runner = CliRunner()
    #     result = runner.invoke(cli.main)
    #     assert result.exit_code == 0
    #     assert 'performance_data.cli.main' in result.output
    #     help_result = runner.invoke(cli.main, ['--help'])
    #     assert help_result.exit_code == 0
    #     assert '--help  Show this message and exit.' in help_result.output
