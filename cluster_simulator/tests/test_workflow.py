import simpy
import os
import json
import unittest
from loguru import logger
from unittest.mock import patch, mock_open, MagicMock
from cluster_simulator.application import Application
from cluster_simulator.workflow import Workflow
from cluster_simulator.cluster import Cluster, Tier

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CLUSTER_CONFIG = os.path.join(CURRENT_DIR, "test_data", "config.yaml")

def mock_run_job(env, job_id):
    # Mock implementation that simply triggers the job event after a set amount of time
    delay = 5  # Mock delay for job execution
    yield env.timeout(delay)

class MockApplication(Application):
    def run(self, cluster, placement, use_bb=None):
        # Mock behavior: complete after a short delay
        yield self.env.timeout(1)



class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        self.cluster = Cluster(self.env, CLUSTER_CONFIG)

    @patch("cluster_simulator.application.Application.__init__", return_value=None)
    def test_workflow_initialization(self, MockApplication):
        jobs = {'job1': MockApplication(), 'job2': MockApplication()}
        dependencies = [
            ['job1', 'job2', {'type': 'sequential'}]
        ]

        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        assert len(workflow.jobs) == 2
        assert len(workflow.dependencies) == 1
        assert all(job_id in workflow.job_events for job_id in jobs)

    #@patch("cluster_simulator.workflow.schedule_job", return_value=None)
    @patch("cluster_simulator.application.Application.__init__", return_value=None)
    def test_setup_dependencies(self, MockApplication):#, mock_schedule_job):
        jobs = {'job1': MockApplication(), 'job2': MockApplication(),
                'job3': MockApplication()}
        dependencies = [
            ['job1', 'job2', {'type': 'delay', 'delay': 10}],
            ['job2', 'job3', {'type': 'parallel'}]
        ]

        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.setup_dependencies()

    def test_setup_delay(self):
        compute = [0,  10]
        read = [1e9, 0]
        write = [0, 5e9]
        tiers = [1, 0]
        jobs = {'job1': Application(self.env, compute=compute, read=read,
                                    write=write),
                'job2': Application(self.env, compute=compute, read=read,
                                    write=write)}
        # Use MockApplication instead of Application in your tests
        # jobs = {
        #     'job1': MockApplication(self.env, compute=compute, read=read, write=write),
        #     'job2': MockApplication(self.env, compute=compute, read=read, write=write)
        # }
        dependencies = [('job1', 'job2', {'type': 'parallel'})]
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        # Execute
        workflow.setup_dependencies()
        self.env.run()
        # self.env.process(jobs["job1"].run(self.cluster, placement=[0, 0], use_bb=[False, False]))
        # self.env.run()

    @patch("cluster_simulator.application.Application.__init__", return_value=None)
    def test_schedule_job(self, MockApplication):
        compute = [0,  10]
        read = [1e9, 0]
        write = [0, 5e9]
        tiers = [1, 0]
        job_events = {
            'job1': self.env.event(),
            'job2': self.env.event(),
            'job3': self.env.event()
        }
        jobs = {'job1': Application(self.env, compute=compute, read=read, write=write),
                'job2': Application(self.env, compute=compute, read=read, write=write),
                'job3': MockApplication()}
        workflow = Workflow(self.env, jobs, job_events, [])

        # Assume job1 completes at time 5
        self.env.process(workflow.schedule_job('job2', job_events['job1'], delay=5))
        job_events['job1'].succeed(at=5)

        self.env.run(until=9)
        assert not job_events['job2'].processed  # Job 2 should not have started yet (delay of 5)

        self.env.run(until=11)
        assert job_events['job2'].processed  # Job 2 should start after the delay


