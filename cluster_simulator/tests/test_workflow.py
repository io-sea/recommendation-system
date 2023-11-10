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
        yield self.env.timeout(10)



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
        # jobs = {'job1': Application(self.env, name='job1',
        #                             compute=compute, read=read,
        #                             write=write),
        #         'job2': Application(self.env, name='job2',
        #                             compute=compute, read=read,
        #                             write=write)}
        # Use MockApplication instead of Application in your tests
        jobs = {
            'job1': MockApplication(self.env, compute=compute, read=read, write=write),
            'job2': MockApplication(self.env, compute=compute, read=read, write=write)
        }
        dependencies = [('job1', 'job2', {'type': 'parallel'})]
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        # Execute
        workflow.setup_dependencies()
        self.env.run()
        # self.env.process(jobs["job1"].run(self.cluster, placement=[0, 0], use_bb=[False, False]))
        # self.env.run()

    @patch("cluster_simulator.application.Application.__init__", return_value=None)
    def test_schedule_job(self, mock_init):
        compute = [0,  10]
        read = [1e9, 0]
        write = [0, 5e9]
        tiers = [1, 0]
        job_events = {
            'job1': self.env.event(),
            'job2': self.env.event(),
            'job3': self.env.event()
        }
        jobs = {'job1': Application(self.env, name='job1',
                                    compute=compute, read=read,
                                    write=write),
                'job2': Application(self.env, name='job2',
                                    compute=compute, read=read,
                                    write=write),
                'job2': Application(self.env, name='job3',
                                    compute=compute, read=read,
                                    write=write),}
        workflow = Workflow(self.env, jobs, job_events, [])

        # Assume job1 completes at time 5
        self.env.process(workflow.schedule_job('job2', job_events['job1'],
                                               {"type": "delay", "delay":5}))
        job_events['job1'].succeed()

        self.env.run(until=9)
        assert not job_events['job2'].processed  # Job 2 should not have started yet (delay of 5)

        self.env.run(until=11)
        assert job_events['job2'].processed  # Job 2 should start after the delay

    @patch("cluster_simulator.application.Application.run")
    def test_run_job(self, mock_run):
        mock_run.return_value = iter([None])  # Simulate an instantaneous job run

        job_id = 'job1'
        jobs = {job_id: MockApplication(self.env, compute=[0, 10],
                                        read=[1e9, 0],
                                        write=[0, 5e9])}
        dependencies = []

        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        self.env.process(workflow.run_job(job_id))
        self.env.run()
        # Check if the job event is triggered
        self.assertTrue(workflow.job_events[job_id].processed)


    @patch("cluster_simulator.application.Application.run")
    def test_schedule_job_with_delay(self, mock_run):
        # Set the mock to yield a delay, simulating job execution time
        mock_run.return_value = iter([None])

        # Create the jobs and dependencies
        job_id = 'job1'
        dependent_job_id = 'job2'
        delay = 10  # The delay you want to test

        jobs = {
            job_id: MockApplication(self.env, compute=[0, 10], read=[1e9, 0],
                                    write=[0, 5e9]),
            dependent_job_id: MockApplication(self.env, compute=[0, 10],
                                              read=[1e9, 0],
                                              write=[0, 5e9])
        }

        dependencies = [(job_id, dependent_job_id,
                         {'type': 'delay', 'delay': delay})]

        # Initialize the workflow
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)

        # Schedule the jobs with dependencies
        workflow.setup_dependencies()

        # Start the simulation environment
        start_time = self.env.now
        self.env.run()

        # The dependent job should not start until after the delay
        dependent_job_start_time = workflow.job_events[dependent_job_id].env.now
        # Check if the job started after the correct delay
        self.assertEqual(dependent_job_start_time, start_time + 30)

        # Check if the job event is triggered
        self.assertTrue(workflow.job_events[dependent_job_id].processed)


    @patch("cluster_simulator.application.Application.run")
    def test_schedule_job_sequential(self, mock_run):
        mock_run.return_value = iter([None])
        # Setup two jobs where the second depends on the first to finish sequentially
        jobs = {
            'job1': MockApplication(self.env, compute=[0, 10], read=[1e9, 0], write=[0, 5e9]),
            'job2': MockApplication(self.env, compute=[0, 10], read=[1e9, 0], write=[0, 5e9])
        }
        dependencies = [('job1', 'job2', {'type': 'sequential'})]

        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.setup_dependencies()

        # Run the simulation
        self.env.run()

        # The job2 should not start until job1 has completed
        self.assertTrue(workflow.job_events['job1'].processed)
        self.assertTrue(workflow.job_events['job2'].processed)
        self.assertGreaterEqual(workflow.job_events['job2'].env.now, workflow.job_events['job1'].env.now)

    @patch("cluster_simulator.application.Application.run")
    def test_schedule_job_sequential_and_parallel(self, mock_run):
        #mock_run.return_value = iter([None])
        # Setup two jobs where the second depends on the first to finish sequentially
        jobs = {
            'job1': MockApplication(self.env, compute=[0, 10], read=[1e9, 0], write=[0, 5e9]),
            'job2': MockApplication(self.env, compute=[0, 10], read=[1e9, 0], write=[0, 5e9]),
            'job3': MockApplication(self.env, compute=[0, 10], read=[1e9, 0], write=[0, 5e9])
        }
        dependencies = [('job1', 'job2', {'type': 'sequential'}),
                        ('job2', 'job3', {'type': 'parallel'})]

        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.setup_dependencies()

        # Initial check before simulation
        for job_id in jobs:
            assert not workflow.job_events[job_id].processed, f"Job {job_id} should not start yet"

        # Run the simulation
        self.env.run()
        # Verify sequential dependency (job1 before job2)
        self.assertTrue(workflow.job_events['job1'].processed)
        self.assertTrue(workflow.job_events['job2'].processed)
        self.assertGreaterEqual(workflow.job_events['job2'].env.now,
                                workflow.job_events['job1'].env.now)

    @patch("cluster_simulator.application.Application.run")
    def test_check_other_dependencies(self, mock_run):
        mock_run.return_value = iter([None])  # Simulate an instantaneous job run

        # Define jobs and dependencies
        job_id = 'job4'  # This is the job with multiple dependencies
        jobs = {
            'job1': MockApplication(self.env, compute=[0, 10], read=[1e9, 0], write=[0, 5e9]),
            'job2': MockApplication(self.env, compute=[0, 10], read=[1e9, 0], write=[0, 5e9]),
            'job3': MockApplication(self.env, compute=[0, 10], read=[1e9, 0], write=[0, 5e9]),
            job_id: MockApplication(self.env, compute=[0, 10], read=[1e9, 0], write=[0, 5e9])
        }
        dependencies = [
            ('job1', job_id, {'type': 'sequential'}),
            ('job2', job_id, {'type': 'sequential'}),
            ('job3', job_id, {'type': 'sequential'})
        ]
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.setup_dependencies()

        # Simulate the completion of job1 and job2 dependencies
        workflow.job_events['job1'].succeed()
        workflow.job_events['job2'].succeed()

        # Run the simulation until just before the expected start time of job4
        print(workflow.job_events[job_id].processed)
        self.env.run(until=self.env.now + 1)
        assert not workflow.job_events[job_id].processed, "Job 4 should not start yet"

        # Now complete the final dependency
        workflow.job_events['job3'].succeed()

        # Continue the simulation
        self.env.run()

        # Check that job4 has started only after all dependencies are met
        assert workflow.job_events[job_id].processed, "Job 4 should have started after all dependencies are met"





