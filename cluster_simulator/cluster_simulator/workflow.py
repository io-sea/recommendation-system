from cluster_simulator.application import Application
import json
import networkx as nx
import simpy
import os
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
    def __init__(self, env, jobs, dependencies, cluster, jobs_placements=None):
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
        self.graph = self.build_graph()  # Reconstruct the graph from dependencies
        self.cluster = cluster
        self.jobs_placements = jobs_placements or {}  # Default to empty dict if not provided
        self.job_events = {job_id: self.env.event() for job_id in jobs.keys()}
        logger.info(f"Workflow initialized with {len(self.jobs)} jobs and {len(self.dependencies)} dependencies.")

    def print_graph(self):
        """
        Prints the graph to the console.
        """

        logger.debug("Job Dependency Graph:")
        for node in self.graph.nodes():
            logger.debug(f"Job: {node}")
        for edge in self.graph.edges(data=True):
            logger.debug(f"From {edge[0]} to {edge[1]} - Type: {edge[2]['type']}")

    def build_graph(self):
        """
        Builds a graph from dependencies, which can be a JSON file path or a dictionary.
        """
        logger.debug("Building the dependency graph.")
        self.graph = nx.DiGraph()
        if isinstance(self.dependencies, str):  # dependencies is a file path
            if not os.path.exists(self.dependencies):
                raise FileNotFoundError(f"File {self.dependencies} does not exist.")
            with open(self.dependencies, 'r') as f:
                graph_data = json.load(f)
            self.graph = nx.node_link_graph(graph_data)
        elif isinstance(self.dependencies, list):  # dependencies is a list of tuples
            for pre_job, post_job, relation in self.dependencies:
                self.graph.add_edge(pre_job, post_job, **relation)
        else:
            raise ValueError("Invalid dependencies format. Must be file path or list of tuples.")
         # Log the entire graph structure
        logger.debug("Graph structure:")
        for node in self.graph.nodes:
            logger.debug(f"Node: {node}, Incoming Edges: {list(self.graph.in_edges(node))}, Outgoing Edges: {list(self.graph.out_edges(node))}")
        return self.graph

    def find_parallel_chains(self):
        """
        Identifie et retourne toutes les chaînes parallèles dans le graphe.

        Returns:
            list of lists: Une liste de chaînes, où chaque chaîne est une liste de job IDs.
        """
        visited = set()
        parallel_chains = []

        for job_id in self.jobs:
            if job_id not in visited:
                chain = self.find_parallel_chain(job_id, visited)
                if chain:
                    parallel_chains.append(chain)

        return parallel_chains

    def find_parallel_chain(self, job_id, visited):
        """
        Trouve récursivement tous les jobs dans une chaîne parallèle à partir d'un job donné.

        Args:
            job_id (str): ID du job de départ.
            visited (set): Ensemble des jobs déjà visités.

        Returns:
            list: Une chaîne parallèle sous forme de liste de job IDs.
        """
        chain = []
        stack = [job_id]

        while stack:
            current_job = stack.pop()
            if current_job not in visited:
                visited.add(current_job)
                chain.append(current_job)
                # Ajouter les prédécesseurs et successeurs connectés par des relations parallèles
                for neighbor in self.graph.predecessors(current_job):
                    if self.graph[neighbor][current_job].get('type') == 'parallel':
                        stack.append(neighbor)
                for neighbor in self.graph.successors(current_job):
                    if self.graph[current_job][neighbor].get('type') == 'parallel':
                        stack.append(neighbor)

        return chain if len(chain) > 1 else []

    def setup_dependencies(self):
        """
        Setup dependencies between jobs, identifying independent and parallel chains and scheduling them.
        """
        logger.info("Setting up job dependencies.")

        # Identify all parallel chains in the graph
        parallel_chains = self.find_parallel_chains()

        # Identify and schedule independent jobs and their parallel chains
        for job_id in self.jobs:
            if self.is_independent_job(job_id):
                if self.is_in_parallel_chain(job_id):
                    chain = self.find_parallel_chain(job_id, set())
                    self.schedule_parallel_chain(chain)
                else:
                    self.env.process(self.run_job(job_id))

        # Schedule parallel chains that are independent or have resolved dependencies
        for chain in parallel_chains:
            if self.are_chain_dependencies_resolved(chain):
                self.schedule_parallel_chain(chain)

        # Schedule remaining jobs considering their dependencies
        for job_id in self.jobs:
            if not self.is_in_parallel_chain(job_id) and not self.is_independent_job(job_id):
                self.schedule_job_with_dependencies(job_id)

        logger.info("Job dependencies setup is complete.")

    def is_independent_job(self, job_id):
        """
        Check if a job is independent (no incoming dependencies).

        Args:
            job_id (str): The job ID to check.

        Returns:
            bool: True if the job is independent, False otherwise.
        """
        return self.graph.in_degree(job_id) == 0

    def is_in_parallel_chain(self, job_id):
        """
        Determines if a job is part of a parallel chain.

        Args:
            job_id (str): The identifier for the job.

        Returns:
            bool: True if the job is part of a parallel chain, False otherwise.
        """
        logger.debug(f"Checking if job {job_id} is in a parallel chain.")
        parallel_chains = self.find_parallel_chains()

        for chain in parallel_chains:
            if job_id in chain:
                logger.debug(f"Job {job_id} is in a parallel chain.")
                return True

        logger.debug(f"Job {job_id} is not in a parallel chain.")
        return False

    def schedule_job_with_dependencies(self, job_id):
        """
        Schedule a job taking into account its dependencies.

        Args:
            job_id (str): The identifier for the job.
        """
        logger.debug(f"Scheduling job with dependencies: {job_id}")

        # Retrieve the predecessor, if any
        predecessor = next(self.graph.predecessors(job_id), None)

        # If there is a predecessor, wait for it to complete before scheduling this job
        if predecessor:
            logger.debug(f"Job {job_id} has a predecessor: {predecessor}")
            pre_job_event = self.job_events[predecessor]

            # Define a process to wait for the predecessor to complete
            def wait_and_run(env, pre_job_event):
                yield pre_job_event
                logger.debug(f"Predecessor {predecessor} completed, now running job {job_id}")
                yield env.process(self.run_job(job_id))

            # Add the process to the simulation environment
            self.env.process(wait_and_run(self.env, pre_job_event))
        else:
            # If there is no predecessor, schedule the job immediately
            logger.debug(f"Job {job_id} has no predecessors, scheduling immediately")
            self.env.process(self.run_job(job_id))

    def schedule_parallel_chain(self, chain):
        """
        Schedule all jobs in a parallel chain to run simultaneously.

        Args:
            chain (list): A list of job IDs that are in the parallel chain.
        """
        logger.debug(f"Scheduling parallel chain: {chain}")

        for job_id in chain:
            # Only schedule the job if all dependencies are resolved
            if self.are_chain_dependencies_resolved(chain):
                logger.debug(f"Scheduling job {job_id} in parallel chain for immediate execution.")
                self.env.process(self.run_job(job_id, is_parallel=True))
            else:
                # Otherwise, wait for the dependencies to be resolved before scheduling
                predecessor = next(self.graph.predecessors(job_id), None)
                if predecessor:
                    self.env.process(self.wait_and_schedule_job(job_id, predecessor))

        logger.info(f"Parallel chain scheduled: {chain}")

    # def are_chain_dependencies_resolved(self, chain):
    #     """
    #     Check if the dependencies of a parallel chain are resolved.

    #     Args:
    #         chain (list of str): List of job IDs in the parallel chain.

    #     Returns:
    #         bool: True if the dependencies are resolved, False otherwise.
    #     """
    #     for job_id in chain:
    #         if self.graph.in_degree(job_id) > 0:
    #             predecessor = next(self.graph.predecessors(job_id), None)
    #             if not self.job_events[predecessor].processed:
    #                 return False
    #     return True
    def are_chain_dependencies_resolved(self, chain):
        """
        Checks if all dependencies for a parallel chain are resolved.

        Args:
            chain (list): A list of job IDs in the parallel chain.

        Returns:
            bool: True if all dependencies are resolved, False otherwise.
        """
        for job_id in chain:
            predecessor = next(self.graph.predecessors(job_id), None)
            if predecessor and not self.job_events[predecessor].triggered:
                return False
        return True

    # def schedule_job(self, job_id, pre_job_event, relation):
    #     """
    #     Schedules a job to start after its dependencies have been met.

    #     Args:
    #         job_id (str): The identifier for the job to be scheduled.
    #         pre_job_event (simpy.Event): The event indicating the completion of the prerequisite job.
    #         relation (dict): The relationship dict containing type and optionally delay.
    #     """
    #     # Wait for the prerequisite job to complete if there is one
    #     if pre_job_event is not None:
    #         yield pre_job_event

    #     # Extract relation type, defaulting to None if relation is None
    #     relation_type = relation.get('type') if relation else None

    #     # If there is a delay relation, introduce a delay
    #     if relation_type == 'delay':
    #         delay = relation.get('delay', 0)
    #         yield pre_job_event
    #         if delay:
    #             logger.debug(f"Introducing a delay of {delay} for job {job_id}.")
    #             yield self.env.timeout(delay)

    #     # Check for other sequential dependencies if the relation type is 'sequential'
    #     if relation_type == 'sequential':
    #         logger.debug(f"Checking other sequential dependencies for job {job_id}.")
    #         yield self.env.process(self.check_other_dependencies(job_id))

    #     # No special handling required for 'parallel' relation type as it's the default behavior
    #     if relation_type == 'parallel':
    #         logger.debug(f"Job {job_id} will run in parallel after the completion of its prerequisite job.")

    #     logger.info(f"Scheduling job {job_id}.")
    #     # Schedule the job for execution
    #     self.env.process(self.run_job(job_id, is_parallel=True))

    def run_job(self, job_id, is_parallel=False, cluster=None, placement=None,
                use_bb=None):
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
        job = self.jobs[job_id]
        placement = self.jobs_placements.get(job_id, {}).get('placement', [0])
        use_bb = self.jobs_placements.get(job_id, {}).get('use_bb', [False])

        # Normalize the placement and use_bb lists
        placement, use_bb = self.normalize_placement_and_bb(job, placement, use_bb)

        # Log the running conditions which have been optimized and passed at the last moment
        logger.debug(f"Running job {job_id} on cluster {cluster} with placement {placement} and burst buffer usage {use_bb}.")

        # Start the job's execution within the simulation environment
        # Trigger the job's event at the start for parallel jobs
        if is_parallel:
            self.job_events[job_id].succeed()
        # and wait for the job to complete
        yield self.env.process(self.jobs[job_id].run(cluster=cluster,
                                                     placement=placement,
                                                     use_bb=use_bb))

        # Trigger the job's event to signal completion, ensuring it's the correct type
        # if isinstance(self.job_events[job_id], simpy.events.Event) and not self.job_events[job_id].triggered:
        if not is_parallel and not self.job_events[job_id].triggered:
            self.job_events[job_id].succeed()

    def check_other_dependencies(self, job_id):
        """
        Check if there are other sequential dependencies that need to be completed before the given job can start.

        Args:
            job_id (str): The identifier for the job whose dependencies are to be checked.

        Yields:
            simpy.Event: An event that triggers when all other dependencies have been met.
        """
        sequential_dependencies = [dep for dep in self.dependencies if dep[1] == job_id and dep[2].get('type') == 'sequential']

        # If there are sequential dependencies, we must wait for all of them to complete
        if sequential_dependencies:
            events_to_wait = []
            for pre_job_id, _, _ in sequential_dependencies:
                events_to_wait.append(self.job_events[pre_job_id])

            # Wait for all events to be triggered
            for event in events_to_wait:
                yield event

    def normalize_placement_and_bb(self, job, placement, use_bb):
        # Assuming the number of phases is determined by the length of compute
        num_phases = len(job.compute)

        # Normalize placement
        if not isinstance(placement, list):
            placement = [placement]
        placement.extend([placement[-1]] * (num_phases - len(placement)))

        # Normalize use_bb
        if not isinstance(use_bb, list):
            use_bb = [use_bb]
        use_bb.extend([use_bb[-1]] * (num_phases - len(use_bb)))

        return placement[:num_phases], use_bb[:num_phases]


    def run(self, jobs_placements=None):
        if jobs_placements:
            self.jobs_placements = jobs_placements

        self.setup_dependencies()
        self.env.run()
