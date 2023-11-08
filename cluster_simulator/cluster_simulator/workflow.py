from cluster_simulator.application import Application
import json
import simpy
from loguru import logger
from cluster_simulator.cluster import Cluster, Tier

class Workflow:
    """Class to manage the execution of jobs within a workflow with dependencies.

    Attributes:
        env (simpy.Environment): The simulation environment.
        jobs (dict): A dictionary mapping job IDs to Application instances.
        dependencies (list): A list of dependencies. Each dependency is represented as a list
            with the format [pre_job_id, post_job_id, relation]. Relation is a dict
            with the format {'type':'delay', 'delay':10}.
        job_events (dict): A dictionary mapping job IDs to SimPy events. These events signal job completion.

    """
    def __init__(self, env, jobs, dependencies, cluster):
        """
        Initialize the workflow with the environment, workflow definition, applications, and cluster.

        Args:
            env (simpy.Environment): The simulation environment.
            jobs (dict): A dictionary of Application instances keyed by job ID.
            dependencies (dict): A dictionary of dependencies keyed by job ID. Each           dependency is a list with the format [pre_job_id, post_job_id, relation].
            cluster (Cluster): The cluster on which to run the applications.
        """
        logger.debug("Initializing Workflow instance.")
        self.env = env
        self.jobs = jobs
        self.dependencies = dependencies
        self.cluster = cluster
        self.job_events = {job_id: self.env.event() for job_id in jobs.keys()}
        logger.info(f"Workflow initialized with {len(self.jobs)} jobs and {len(self.dependencies)} dependencies.")

    def setup_dependencies(self):
        """Setup dependencies between jobs or schedule jobs with no dependencies to start immediately."""
        logger.debug("Setting up dependencies.")
        # Identify all jobs that are not a post-job in any dependency
        independent_jobs = {job_id for job_id in self.jobs} - {post_job_id for _, post_job_id, _ in self.dependencies}

        # Schedule independent jobs to start immediately
        for job_id in independent_jobs:
            logger.debug(f"Scheduling independent job {job_id} to start immediately.")
            self.env.process(self.run_job(job_id))

        # Schedule dependent jobs based on their dependencies
        for pre_job_id, post_job_id, relation in self.dependencies:
            pre_job_event = self.job_events[pre_job_id]
            logger.debug(f"Setting up dependency {pre_job_id} -> {post_job_id} with relation {relation}.")
            self.env.process(self.schedule_job(post_job_id, pre_job_event, relation))

        logger.info("Job dependencies setup complete.")


    def schedule_job(self, job_id, pre_job_event, relation):
        """
        Schedules a job to start after its dependencies have been met.

        Args:
            job_id (str): The identifier for the job to be scheduled.
            pre_job_event (simpy.Event): The event indicating the completion of the prerequisite job.
            relation (dict): The relationship dict containing type and optionally delay.
        """
        # Wait for the prerequisite job to complete if there is one
        if pre_job_event is not None:
            yield pre_job_event

        # Extract relation type, defaulting to None if relation is None
        relation_type = relation.get('type') if relation else None

        # If there is a delay relation, introduce a delay
        if relation_type == 'delay':
            delay = relation.get('delay', 0)
            if delay:
                logger.debug(f"Introducing a delay of {delay} for job {job_id}.")
                yield self.env.timeout(delay)

        # Check for other sequential dependencies if the relation type is 'sequential'
        if relation_type == 'sequential':
            logger.debug(f"Checking other sequential dependencies for job {job_id}.")
            yield self.env.process(self.check_other_dependencies(job_id))

        # No special handling required for 'parallel' relation type as it's the default behavior
        if relation_type == 'parallel':
            logger.debug(f"Job {job_id} will run in parallel after the completion of its prerequisite job.")

        logger.info(f"Scheduling job {job_id}.")
        # Schedule the job for execution
        self.job_events[job_id] = self.env.process(self.run_job(job_id))


    def run_job(self, job_id, cluster=None, placement=None, use_bb=None):
        """
        Manages the execution of a job within the workflow.

        Args:
            job_id (str): The identifier for the job to be executed.
            cluster (Cluster): The cluster on which the job will be run.
            placement (list): The placement strategy for the job's data.
            use_bb (list): A list indicating whether to use burst buffer for each phase of the job.

        Yields:
            simpy.Event: An event that is triggered when the job execution is completed.
        """
        logger.debug(f"Preparing to run job {job_id}.")

        # Ensure the cluster and other parameters are ready for job execution.
        # These would be set based on optimization algorithms/results just before running the job.
        if cluster is None:
            # Use the default cluster if not provided
            cluster = self.cluster
        if placement is None:
            # Default placement
            placement = [0] * len(self.jobs[job_id].compute)
        if use_bb is None:
            # Default use of burst buffer
            use_bb = [False] * len(self.jobs[job_id].compute)

        # Log the running conditions which have been optimized and passed at the last moment
        logger.debug(f"Running job {job_id} on cluster {cluster} with placement {placement} and burst buffer usage {use_bb}.")

        # Start the job's execution within the simulation environment
        job_process = self.env.process(self.jobs[job_id].run(cluster=cluster,
                                            placement=placement,
                                            use_bb=use_bb))

        # Wait for the job to complete
        yield job_process
        # Once the job process is done, trigger the job's event to signal completion
        assert self.job_events[job_id] is not None, f"Event for job_id {job_id} is None"
        self.job_events[job_id].succeed()

