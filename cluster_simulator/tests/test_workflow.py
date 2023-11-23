import simpy
import os
import json
import unittest
from loguru import logger
from unittest.mock import patch, mock_open, MagicMock
from cluster_simulator.application import Application
from cluster_simulator.workflow import Workflow
from cluster_simulator.cluster import Cluster, Tier, EphemeralTier, get_tier

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CLUSTER_CONFIG = os.path.join(CURRENT_DIR, "test_data", "workflow_config.yaml")

def mock_run_job(env, job_id):
    # Mock implementation that simply triggers the job event after a set amount of time
    delay = 5  # Mock delay for job execution
    yield env.timeout(delay)

class MockApplication(Application):
    def run(self, cluster, placement, use_bb=None):
        # Mock behavior: complete after a short delay
        yield self.env.timeout(10)

class TestWorkflowPlacement(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        self.cluster = Cluster(self.env, CLUSTER_CONFIG)

    def test_placement_tiers_sequential(self):
        dependencies = [
            ('job1', 'job2', {'type': 'sequential'})
        ]
        compute, read, write = [0, 10], [1e6, 0], [0, 5e6]
        jobs = {'job1': Application(self.env, name='job1', compute=compute,
                                    read=read, write=write),
                'job2': Application(self.env, name='job2', compute=compute,
                                    read=read, write=write)}
        jobs_placements = {
            'job1': {
                'placement': [1, 1, 1],  # For job1, phase 1 data is placed at tier 0, and phase 2 data is placed at tier 1
                'use_bb': [False, False, False]  # Assuming you also want to specify burst buffer usage for each phase
            },
            'job2': {
                'placement': [0, 0, 0],  # For job2, phase 1 data is placed at tier 1, and phase 2 data is placed at tier 0
                'use_bb': [True, False, False]
            }
        }
        workflow = Workflow(self.env, jobs, dependencies, self.cluster,
                            jobs_placements=jobs_placements)
        workflow.run()

    def test_placement_tiers_parallel(self):
        dependencies = [
            ('job1', 'job2', {'type': 'parallel'})
        ]
        compute, read, write = [0, 10], [1e6, 0], [0, 5e6]
        jobs = {'job1': Application(self.env, name='job1', compute=compute,
                                    read=read, write=write),
                'job2': Application(self.env, name='job2', compute=compute,
                                    read=read, write=write)}
        jobs_placements = {
            'job1': {
                'placement': [1, 1, 1],  # For job1, phase 1 data is placed at tier 0, and phase 2 data is placed at tier 1
                'use_bb': [False, True]  # Assuming you also want to specify burst buffer usage for each phase
            },
            'job2': {
                'placement': [1, 0, 0],  # For job2, phase 1 data is placed at tier 1, and phase 2 data is placed at tier 0
                'use_bb': [False, True]
            }
        }
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.run(jobs_placements=jobs_placements)

    def test_placement_tiers_sequential_bb(self):
        dependencies = [
            ('job1', 'job2', {'type': 'sequential'})
        ]
        compute, read, write = [0, 10], [1e6, 0], [0, 5e6]
        jobs = {'job1': Application(self.env, name='job1', compute=compute,
                                    read=read, write=write),
                'job2': Application(self.env, name='job2', compute=compute,
                                    read=read, write=write)}
        jobs_placements = {
            'job1': {
                'placement': [1, 1, 1],  # For job1, phase 1 data is placed at tier 0, and phase 2 data is placed at tier 1
                'use_bb': [True, False]  # Assuming you also want to specify burst buffer usage for each phase
            },
            'job2': {
                'placement': [0, 0, 0],  # For job2, phase 1 data is placed at tier 1, and phase 2 data is placed at tier 0
                'use_bb': [False, False]
            }
        }
        workflow = Workflow(self.env, jobs, dependencies, self.cluster,
                            jobs_placements=jobs_placements)
        workflow.run()


class TestWorkflowScheduler(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        self.cluster = Cluster(self.env, CLUSTER_CONFIG)

    @patch("cluster_simulator.application.Application.__init__", return_value=None)
    def test_workflow_initialization(self, MockApplication):
        # Arrange
        jobs = {'job1': MockApplication(), 'job2': MockApplication()}
        dependencies = [['job1', 'job2', {'type': 'sequential'}]]

        # Act
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)

        # Assert
        self.assertEqual(len(workflow.jobs), 2, "There should be 2 jobs in the workflow.")
        self.assertEqual(len(workflow.dependencies), 1, "There should be 1 dependency in the workflow.")
        self.assertTrue(all(job_id in workflow.job_events for job_id in jobs), "Each job should have an associated event.")

    def test_build_graph(self):
        dependencies = [
            ('job1', 'job2', {'type': 'sequential'}),
            ('job2', 'job3', {'type': 'parallel'})
        ]
        compute, read, write = [0, 10], [1e9, 0], [0, 5e9]
        jobs = {
            'job1': MockApplication(self.env, compute=compute, read=read, write=write),
            'job2': MockApplication(self.env, compute=compute, read=read, write=write)
        }
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.build_graph()

        # Check if the graph has the correct edges and properties
        self.assertIn(('job1', 'job2'), workflow.graph.edges())
        self.assertIn(('job2', 'job3'), workflow.graph.edges())
        self.assertEqual(workflow.graph['job1']['job2']['type'], 'sequential')
        self.assertEqual(workflow.graph['job2']['job3']['type'], 'parallel')

    def test_parallel_chain_unique(self):
        dependencies = [
            ('job1', 'job2', {'type': 'sequential'}),
            ('job2', 'job3', {'type': 'parallel'})
        ]
        compute, read, write = [0, 10], [1e9, 0], [0, 5e9]
        jobs = {
            'job1': MockApplication(self.env, compute=compute, read=read, write=write),
            'job2': MockApplication(self.env, compute=compute, read=read, write=write)
        }
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.build_graph()
        parallel_chain = workflow.find_parallel_chain('job2', visited=set())
        self.assertSetEqual(set(parallel_chain), {"job2", "job3"})

    def test_parallel_chain_empty(self):
        dependencies = [
            ('job1', 'job2', {'type': 'sequential'}),
            ('job2', 'job3', {'type': 'sequential'})
        ]
        compute, read, write = [0, 10], [1e9, 0], [0, 5e9]
        jobs = {
            'job1': MockApplication(self.env, compute=compute, read=read, write=write),
            'job2': MockApplication(self.env, compute=compute, read=read, write=write)
        }
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.build_graph()
        parallel_chain = workflow.find_parallel_chain('job2', visited=set())
        self.assertSetEqual(set(parallel_chain), set())

    def test_parallel_chain_complex(self):
        dependencies = [
            ('job1', 'job2', {'type': 'parallel'}),
            ('job2', 'job3', {'type': 'sequential'}),
            ('job3', 'job4', {'type': 'parallel'}),
            ('job1', 'job5', {'type': 'sequential'}),
            ('job5', 'job6', {'type': 'sequential'}),
            ('job6', 'job7', {'type': 'parallel'})
        ]
        compute, read, write = [0, 10], [1e9, 0], [0, 5e9]
        jobs = {
            'job1': MockApplication(self.env, compute=compute, read=read, write=write),
            'job2': MockApplication(self.env, compute=compute, read=read, write=write),
            'job3': MockApplication(self.env, compute=compute, read=read, write=write),
            'job4': MockApplication(self.env, compute=compute, read=read, write=write),
            'job5': MockApplication(self.env, compute=compute, read=read, write=write),
            'job6': MockApplication(self.env, compute=compute, read=read, write=write),
            'job7': MockApplication(self.env, compute=compute, read=read, write=write),
        }
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.build_graph()

        self.assertSetEqual(set(workflow.find_parallel_chain('job2', visited=set())),
                            {"job1", "job2"})
        self.assertSetEqual(set(workflow.find_parallel_chain('job3', visited=set())),
                            {"job3", "job4"})
        self.assertSetEqual(set(workflow.find_parallel_chain('job6', visited=set())),
                            {"job6", "job7"})

        self.assertListEqual(workflow.find_parallel_chains(),
                             [['job1', 'job2'], ['job3', 'job4'],
                              ['job6', 'job7']])

    def test_is_independent_job(self):
        dependencies = [
            ('job1', 'job2', {'type': 'sequential'}),
            ('job1', 'job3', {'type': 'parallel'})
        ]
        compute, read, write = [0, 10], [1e9, 0], [0, 5e9]
        jobs = {
            'job1': MockApplication(self.env, compute=compute, read=read, write=write),
            'job2': MockApplication(self.env, compute=compute, read=read, write=write),
            'job3': MockApplication(self.env, compute=compute, read=read, write=write)
        }
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)

        self.assertTrue(workflow.is_independent_job('job1'))
        self.assertFalse(workflow.is_independent_job('job2'))
        self.assertFalse(workflow.is_independent_job('job3'))


    def test_schedule_delay(self):
        # Arrange
        compute, read, write = [0, 10], [1e6, 0], [0, 5e6]
        jobs = {'job1': Application(self.env, name='job1', compute=compute,
                                    read=read, write=write),
                'job2': Application(self.env, name='job2', compute=compute,
                                    read=read, write=write)}

        dependencies = [('job1', 'job2', {'type': 'delay', 'delay': 33})]
        # Act
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.run()

    def test_schedule_delay_0(self):
        # Arrange
        compute, read, write = [0, 10], [1e6, 0], [0, 5e6]
        jobs = {'job1': Application(self.env, name='job1', compute=compute,
                                    read=read, write=write),
                'job2': Application(self.env, name='job2', compute=compute,
                                    read=read, write=write)}

        dependencies = [('job1', 'job2', {'type': 'delay', 'delay': 0})]
        # Act
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.run()

    def test_parallel_chains(self):
        # Arrange
        compute, read, write = [0, 10], [1e9, 0], [0, 5e9]
        jobs = {
            'job1': MockApplication(self.env, compute=compute, read=read, write=write),
            'job2': MockApplication(self.env, compute=compute, read=read, write=write)
        }
        dependencies = [('job1', 'job2', {'type': 'parallel'})]

        # Act
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        parallel_chains = workflow.find_parallel_chains()
        self.assertListEqual([['job1', 'job2']], parallel_chains)
        self.assertTrue(workflow.is_in_parallel_chain('job1'))

    def test_schedule_jobs_sequential_simple(self):
        # Arrange
        compute, read, write = [0, 10], [1e6, 0], [0, 5e6]
        jobs = {'job1': Application(self.env, name='job1', compute=compute,
                                    read=read, write=write),
                'job2': Application(self.env, name='job2', compute=compute,
                                    read=read, write=write)}

        dependencies = [('job1', 'job2', {'type': 'sequential'})]
        # Act
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.run()

    def test_schedule_jobs_sequential_0(self):
        # Arrange
        compute, read, write = [0, 10], [1e6, 0], [0, 5e6]
        jobs = {'job1': Application(self.env, name='job1', compute=compute,
                                    read=read, write=write),
                'job2': Application(self.env, name='job2', compute=compute,
                                    read=read, write=write),
                'job3': Application(self.env, name='job3', compute=compute,
                                    read=read, write=write)}

        dependencies = [('job1', 'job2', {'type': 'sequential'}),
                        ('job2', 'job3', {'type': 'sequential'})]
        # Act
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.run()

    def test_schedule_jobs_sequential_1(self):
        # Arrange
        compute, read, write = [0, 10], [1e6, 0], [0, 5e6]
        jobs = {'job1': Application(self.env, name='job1', compute=compute,
                                    read=read, write=write),
                'job2': Application(self.env, name='job2', compute=compute,
                                    read=read, write=write),
                'job3': Application(self.env, name='job3', compute=compute,
                                    read=read, write=write)}

        dependencies = [('job1', 'job2', {'type': 'sequential'}),
                        ('job1', 'job3', {'type': 'sequential'})]
        # Act
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.run()

    def test_schedule_jobs_sequential_2(self):
        compute, read, write = [0, 10], [1e6, 0], [0, 5e6]
        jobs = {'job1': Application(self.env, name='job1', compute=compute,
                                    read=read, write=write),
                'job2': Application(self.env, name='job2', compute=compute,
                                    read=read, write=write),
                'job3': Application(self.env, name='job3', compute=compute,
                                    read=read, write=write),
                'job4': Application(self.env, name='job4', compute=compute,
                                    read=read, write=write)}

        dependencies = [('job1', 'job2', {'type': 'sequential'}),
                        ('job1', 'job3', {'type': 'sequential'}),
                        ('job3', 'job4', {'type': 'sequential'})]
        # Act
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.run()

    def test_schedule_jobs_sequential_3(self):
        compute, read, write = [0, 10], [1e6, 0], [0, 5e6]
        jobs = {'job1': Application(self.env, name='job1', compute=compute,
                                    read=read, write=write),
                'job2': Application(self.env, name='job2', compute=compute,
                                    read=read, write=write),
                'job3': Application(self.env, name='job3', compute=compute,
                                    read=read, write=write),
                'job4': Application(self.env, name='job4', compute=compute,
                                    read=read, write=write)}

        dependencies = [('job1', 'job2', {'type': 'sequential'}),
                        ('job2', 'job3', {'type': 'sequential'}),
                        ('job3', 'job4', {'type': 'sequential'})]
        # Act
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.run()

    def test_schedule_jobs_parallel(self):
        # Arrange
        compute, read, write = [0, 10], [1e6, 0], [0, 5e6]
        jobs = {'job1': Application(self.env, name='job1', compute=compute,
                                    read=read, write=write),
                'job2': Application(self.env, name='job2', compute=compute,
                                    read=read, write=write),
                }

        dependencies = [('job1', 'job2', {'type': 'parallel'})]
        # Act
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.run()

    def test_schedule_jobs_parallel_1(self):
        # Arrange
        compute, read, write = [0, 10], [1e6, 0], [0, 5e6]
        jobs = {'job1': Application(self.env, name='job1', compute=compute,
                                    read=read, write=write),
                'job2': Application(self.env, name='job2', compute=compute,
                                    read=read, write=write),
                'job3': Application(self.env, name='job3', compute=compute,
                                    read=read, write=write)}

        dependencies = [('job1', 'job2', {'type': 'parallel'}),
                        ('job1', 'job3', {'type': 'parallel'})]
        # Act
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.run()

    def test_schedule_jobs_parallel_sequential_0(self):
        # Arrange
        compute, read, write = [0, 10], [1e6, 0], [0, 5e6]
        jobs = {'job1': Application(self.env, name='job1', compute=compute,
                                    read=read, write=write),
                'job2': Application(self.env, name='job2', compute=compute,
                                    read=read, write=write),
                'job3': Application(self.env, name='job3', compute=compute,
                                    read=read, write=write)}

        dependencies = [('job1', 'job2', {'type': 'parallel'}),
                        ('job2', 'job3', {'type': 'sequential'})]
        # Act
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.run()

    def test_schedule_jobs_parallel_sequential_1(self):
        # Arrange
        compute, read, write = [0, 10], [1e6, 0], [0, 5e6]
        jobs = {'job1': Application(self.env, name='job1', compute=compute,
                                    read=read, write=write),
                'job2': Application(self.env, name='job2', compute=compute,
                                    read=read, write=write),
                'job3': Application(self.env, name='job3', compute=compute,
                                    read=read, write=write)}

        dependencies = [('job1', 'job2', {'type': 'sequential'}),
                        ('job2', 'job3', {'type': 'parallel'})]
        # Act
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.run()

    def test_schedule_jobs_parallel_2(self):
        # Arrange
        compute, read, write = [0, 10], [1e6, 0], [0, 5e6]
        jobs = {'job1': Application(self.env, name='job1', compute=compute,
                                    read=read, write=write),
                'job2': Application(self.env, name='job2', compute=compute,
                                    read=read, write=write),
                'job3': Application(self.env, name='job3', compute=compute,
                                    read=read, write=write),
                'job4': Application(self.env, name='job4', compute=compute,
                                    read=read, write=write)}

        dependencies = [('job1', 'job2', {'type': 'sequential'}),
                        ('job2', 'job3', {'type': 'parallel'}),
                        ('job3', 'job4', {'type': 'parallel'})]
        # Act
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.run()

    def test_schedule_jobs_complex_0(self):
        # Arrange
        compute, read, write = [0, 10], [1e6, 0], [0, 5e6]
        jobs = {'job1': Application(self.env, name='job1', compute=[0, 10],
                                    read=[1e6, 0], write=[0, 5e6]),
                'job2': Application(self.env, name='job2', compute=compute,
                                    read=read, write=write),
                'job3': Application(self.env, name='job3', compute=compute,
                                    read=read, write=write),
                'job4': Application(self.env, name='job4', compute=compute,
                                    read=read, write=write),
                'job5': Application(self.env, name='job5', compute=compute,
                                    read=read, write=write)}

        dependencies = [('job1', 'job2', {'type': 'sequential'}),
                        ('job2', 'job3', {'type': 'parallel'}),
                        ('job3', 'job4', {'type': 'parallel'}),
                        ('job4', 'job5', {'type': 'sequential'})]
        # Act
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.run()

    def test_schedule_jobs_complex_1(self):
        # Arrange
        compute, read, write = [0, 10], [1e6, 0], [0, 5e6]
        jobs = {'job1': Application(self.env, name='job1', compute=[0, 10],
                                    read=[1e6, 0], write=[0, 5e6]),
                'job2': Application(self.env, name='job2', compute=compute,
                                    read=read, write=write),
                'job3': Application(self.env, name='job3', compute=compute,
                                    read=read, write=write),
                'job4': Application(self.env, name='job4', compute=compute,
                                    read=read, write=write),
                'job5': Application(self.env, name='job5', compute=compute,
                                    read=read, write=write)}

        dependencies = [('job1', 'job2', {'type': 'sequential'}),
                        ('job2', 'job3', {'type': 'parallel'}),
                        ('job3', 'job4', {'type': 'parallel'}),
                        ('job2', 'job5', {'type': 'delay', 'delay': 10})]
        # Act
        workflow = Workflow(self.env, jobs, dependencies, self.cluster)
        workflow.run()




    # def test_setup_deps_2(self):
    #     dependencies = [
    #         ('job1', 'job2', {'type': 'parallel'}),
    #         ('job1', 'job3', {'type': 'parallel'})
    #     ]
    #     compute, read, write = [0, 10], [1e9, 0], [0, 5e9]
    #     jobs = {
    #         'job1': MockApplication(self.env, compute=compute, read=read, write=write),
    #         'job2': MockApplication(self.env, compute=compute, read=read, write=write),
    #         'job3': MockApplication(self.env, compute=compute, read=read, write=write)
    #     }
    #     workflow = Workflow(self.env, jobs, dependencies, self.cluster)
    #     workflow.build_graph()
    #     workflow.setup_dependencies()

    #     # Check if the independent jobs are identified correctly
    #     expected_independent_jobs = {'job1', 'job2', 'job3'}
    #     actual_independent_jobs = workflow.independent_jobs
    #     self.assertEqual(expected_independent_jobs, actual_independent_jobs)

    # def test_setup_deps_3(self):
    #     dependencies = [
    #         ('job1', 'job2', {'type': 'parallel'}),
    #         ('job1', 'job3', {'type': 'sequential'}),
    #         ('job2', 'job4', {'type': 'parallel'})
    #     ]
    #     compute, read, write = [0, 10], [1e9, 0], [0, 5e9]
    #     jobs = {
    #         'job1': MockApplication(self.env, compute=compute, read=read, write=write),
    #         'job2': MockApplication(self.env, compute=compute, read=read, write=write),
    #         'job3': MockApplication(self.env, compute=compute, read=read, write=write),
    #         'job4': MockApplication(self.env, compute=compute, read=read, write=write)
    #     }
    #     workflow = Workflow(self.env, jobs, dependencies, self.cluster)
    #     workflow.build_graph()
    #     workflow.setup_dependencies()

    #     # Check if the independent jobs are identified correctly
    #     expected_independent_jobs = {'job1', 'job2', 'job4'}
    #     actual_independent_jobs = workflow.independent_jobs
    #     self.assertEqual(expected_independent_jobs, actual_independent_jobs)

    # def test_setup_deps_4(self):
    #     dependencies = [
    #         ('job1', 'job2', {'type': 'parallel'}),
    #         ('job1', 'job3', {'type': 'sequential'}),
    #         ('job2', 'job4', {'type': 'parallel'}),
    #         ('job3', 'job5', {'type': 'parallel'})

    #     ]
    #     compute, read, write = [0, 10], [1e9, 0], [0, 5e9]
    #     jobs = {
    #         'job1': MockApplication(self.env, compute=compute, read=read, write=write),
    #         'job2': MockApplication(self.env, compute=compute, read=read, write=write),
    #         'job3': MockApplication(self.env, compute=compute, read=read, write=write),
    #         'job4': MockApplication(self.env, compute=compute, read=read, write=write),
    #         'job5': MockApplication(self.env, compute=compute, read=read, write=write)
    #     }
    #     workflow = Workflow(self.env, jobs, dependencies, self.cluster)
    #     workflow.build_graph()
    #     workflow.setup_dependencies()

    #     # Check if the independent jobs are identified correctly
    #     expected_independent_jobs = {'job1', 'job2', 'job4'}
    #     actual_independent_jobs = workflow.independent_jobs
    #     self.assertEqual(expected_independent_jobs, actual_independent_jobs)

    # def test_setup_deps_5(self):
    #     dependencies = [
    #         ('job1', 'job2', {'type': 'sequential'}),
    #         ('job1', 'job3', {'type': 'sequential'}),
    #         ('job2', 'job4', {'type': 'sequential'})
    #     ]
    #     compute, read, write = [0, 10], [1e9, 0], [0, 5e9]
    #     jobs = {
    #         'job1': MockApplication(self.env, compute=compute, read=read, write=write),
    #         'job2': MockApplication(self.env, compute=compute, read=read, write=write),
    #         'job3': MockApplication(self.env, compute=compute, read=read, write=write)
    #     }
    #     workflow = Workflow(self.env, jobs, dependencies, self.cluster)
    #     workflow.build_graph()
    #     workflow.setup_dependencies()

    #     # Check if the independent jobs are identified correctly
    #     expected_independent_jobs = {"job1"}
    #     actual_independent_jobs = workflow.independent_jobs
    #     self.assertEqual(expected_independent_jobs, actual_independent_jobs)

