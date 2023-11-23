from cluster_simulator.application import Application
import json
import networkx as nx
import simpy
import os
from loguru import logger
from cluster_simulator.cluster import Cluster, Tier


class Workflow:
    """
    Manages the execution of a set of interdependent jobs within a high-performance computing (HPC) workflow.

    This class provides functionality to schedule and run jobs based on their dependencies,
    utilizing a simulation environment for orchestration.

    Attributes:
        env (simpy.Environment): The simulation environment where the jobs are executed.
        jobs (dict): A mapping of job IDs to their corresponding Application instances.
        dependencies (list): A list of dependencies, where each dependency is represented as a tuple (pre_job_id, post_job_id, relation). The relation is a dictionary with keys like 'type' and 'delay'.
        job_events (dict): A mapping of job IDs to SimPy events, used to signal the completion of each job.
        graph (nx.DiGraph): A directed graph representing the dependencies between jobs.
        cluster (Cluster): The computational cluster where the jobs are executed.
        jobs_placements (dict): A mapping of job IDs to their data placement strategies and burst buffer usage.

    """
    def __init__(self, env, jobs, dependencies, cluster, jobs_placements=None):
        """
        Initializes the Workflow instance with the required simulation environment, job definitions,
        dependencies, cluster, and optional job placement strategies.

        Args:
            env (simpy.Environment): The simulation environment for running the workflow.
            jobs (dict): A dictionary mapping job IDs to Application instances, representing each job in the workflow.
            dependencies (list): A list of tuples representing job dependencies. Each tuple is in the form (pre_job_id, post_job_id, relation), where relation is a dictionary specifying the dependency type and delay.
            cluster (Cluster): The cluster resource on which the jobs will be executed.
            jobs_placements (dict, optional): A dictionary specifying the data placement strategy and burst buffer usage for each job. Defaults to None, which means no specific placement strategy is provided.

        Raises:
            ValueError: If the provided dependencies are not in the expected list format.

        The initialization process involves setting up the simulation environment, jobs, dependencies, and the
        cluster. It also constructs a directed graph to represent job dependencies for efficient scheduling.
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
        Constructs a directed graph to represent the dependencies between jobs.

        This method supports building the graph from a JSON file or directly from
        a list of dependency tuples. The graph is used to manage job execution order,
        considering their interdependencies.

        Raises:
            FileNotFoundError: If the dependencies file path is provided but the file does not exist.
            ValueError: If the dependencies format is neither a file path nor a list of tuples.

        Returns:
            networkx.DiGraph: A directed graph representing job dependencies.

        Note:
        The graph is stored as a networkx DiGraph object, providing efficient
        graph operations for the scheduling and execution of the workflow.
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
        Identifies and returns all parallel chains in the workflow graph.

        A parallel chain is defined as a sequence of jobs connected with 'parallel' or 'delay' type dependencies,
        indicating that these jobs can be executed in a parallel or delayed parallel manner.

        Returns:
            list of lists: A list containing parallel chains, where each chain is a list of job IDs. A chain is only
            considered parallel if it contains more than one job connected with parallel dependencies.
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
        Recursively finds all jobs in a parallel chain starting from a given job.

        This method explores both predecessors and successors of the given job, adding them to the chain if they
        are connected by 'parallel' or 'delay' dependencies. It ensures that each job is visited only once to avoid
        infinite loops in the graph traversal.

        Args:
            job_id (str): The ID of the starting job.
            visited (set): A set of job IDs that have already been visited in the graph traversal.

        Returns:
            list: A parallel chain as a list of job IDs. Returns an empty list if no parallel chain is found starting
            from the given job or if the chain contains only the starting job.
        """
        chain = []
        stack = [job_id]

        while stack:
            current_job = stack.pop()
            if current_job not in visited:
                visited.add(current_job)
                chain.append(current_job)
                # Add predecessors and successors connected by parallel and delay relationships.
                for neighbor in self.graph.predecessors(current_job):
                    if self.graph[neighbor][current_job].get('type') in ['parallel', 'delay']:
                        stack.append(neighbor)
                for neighbor in self.graph.successors(current_job):
                    if self.graph[current_job][neighbor].get('type') in ['parallel', 'delay']:
                        stack.append(neighbor)

        return chain if len(chain) > 1 else []

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

    def find_all_external_predecessors(self, chain):
        """
        Find all unique external predecessors of the jobs in the chain.

        Args:
            chain (list): A list of job IDs in the chain.

        Returns:
            list: A list of unique job IDs of external predecessors, or an empty list if none are found.
        """
        external_predecessors = set()

        for job_id in chain:
            # Retrieve the predecessors and check if any of them are outside of the chain
            predecessors = self.graph.predecessors(job_id)
            for pred in predecessors:
                if pred not in chain:
                    external_predecessors.add(pred)

        return list(external_predecessors)

    def setup_dependencies(self):
        """
        Organizes the scheduling of jobs based on their dependencies.

        This method identifies and schedules independent jobs and parallel chains,
        and then schedules the remaining jobs, considering their dependencies.
        It ensures that jobs are executed in an order that respects the
        predefined workflow structure.

        Note:
        The method uses internal utility functions to manage job queues and
        relies on other class methods to determine job independence, chain
        membership, and scheduling.
        """
        logger.info("Setting up job dependencies.")

        def add_to_queue(queue, elements):
            """
            Adds elements to a queue, accepting either a single element or a list of elements.

            Args:
                queue (list): La file d'attente actuelle.
                elements: Un élément individuel ou une liste d'éléments à ajouter.
            """
            if isinstance(elements, list):
                queue.extend(elements)
            else:
                queue.append(elements)

        def remove_from_queue(queue, elements):
            """
            Removes elements from a queue, accepting either a single element or a list of elements.

            Args:
                queue (list): La file d'attente actuelle.
                elements: Un élément individuel ou une liste d'éléments à retirer.
            """
            if isinstance(elements, list):
                for element in elements:
                    if element in queue:
                        queue.remove(element)
            else:
                if elements in queue:
                    queue.remove(elements)
        # A set to keep track of jobs that have been scheduled in a parallel chain
        scheduled_in_parallel = set()
        job_queue = list(self.jobs.keys())

        # Step 1: Identify and schedule independent jobs
        for job_id in self.jobs:
            if self.is_independent_job(job_id):
                logger.debug(f"Job {job_id} is independent.")

                # Check if the job is in a parallel chain and schedule the chain if necessary
                if self.is_in_parallel_chain(job_id) and job_id not in scheduled_in_parallel:
                    chain = self.find_parallel_chain(job_id, set())
                    logger.debug(f"Scheduling parallel chain: {chain}")
                    self.schedule_parallel_chain_with_dependencies(chain)
                    # Add all jobs in the chain to the scheduled set
                    scheduled_in_parallel.update(chain)
                    remove_from_queue(job_queue, chain)

                # If the job is not in a parallel chain, schedule it for immediate execution
                elif not self.is_in_parallel_chain(job_id):
                    logger.debug(f"Scheduling independent job {job_id} for immediate execution.")
                    remove_from_queue(job_queue, job_id)
                    self.env.process(self.run_job(job_id))

        # Step 2: Schedule parallel chains that have not been scheduled yet
        for chain in self.find_parallel_chains():
            # Check any parallel chain that has not been scheduled yet has all its dependencies resolved
            if not scheduled_in_parallel.intersection(set(chain)):
                logger.debug(f"Scheduling parallel chain: {chain}")
                remove_from_queue(job_queue, chain)
                scheduled_in_parallel.update(chain)
                self.schedule_parallel_chain_with_dependencies(chain)

        # Step 3: Schedule remaining jobs considering their dependencies
        while job_queue:
            for job_id in job_queue:
                logger.debug(f"Job queue: {job_queue}")
                self.schedule_job_with_dependencies(job_id)
                remove_from_queue(job_queue, job_id)

    def schedule_job_with_dependencies(self, job_id):
        """
        Schedules a job to run, considering its dependencies within the workflow.

        This method schedules the specified job only after ensuring all its
        dependencies are resolved. If the job has a predecessor, it waits for the
        predecessor to complete before starting. Otherwise, it starts the job immediately.

        Args:
            job_id (str): The identifier of the job to be scheduled.

        Note:
            This method assumes that the workflow graph correctly reflects the job
            dependencies and that the job events are managed appropriately.
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

    def schedule_parallel_chain_with_dependencies(self, chain):
        """
        Schedules a chain of parallel jobs, taking into account their dependencies.

        This method schedules each job in the specified chain to run in parallel,
        but only after all the dependencies (if any) are resolved. If a chain has
        external predecessors, it waits for these jobs to complete before starting
        the chain. Otherwise, it starts the chain immediately.

        Args:
            chain (list of str): A list of job IDs that are part of the parallel chain.

        Note:
            This method assumes that the job dependencies have been correctly set up
            in the workflow graph and that the job events are managed appropriately.
        """
        logger.debug(f"Scheduling parallel chain with dependencies: {chain}")
        predecessors = self.find_all_external_predecessors(chain)

        if predecessors:
            logger.debug(f"Chain {chain} has a predecessor(s): {predecessors}")
            pre_job_event = self.job_events[predecessors[0]]

            # Define a process to wait for the predecessor to complete
            def wait_and_run(env, pre_job_event):
                yield pre_job_event
                logger.debug(f"Predecessors {predecessors} completed, now running chain {chain}")
                chain_events = [self.env.process(self.run_job(job_id, is_parallel=True)) for job_id in chain]
                yield simpy.AllOf(self.env, chain_events)

            # Add the process to the simulation environment
            self.env.process(wait_and_run(self.env, pre_job_event))
        else:
            # If there is no predecessor, schedule the job immediately
            logger.debug(f"Chain {chain} has no predecessors, scheduling immediately")

            for job_id in chain:
                self.env.process(self.run_job(job_id, is_parallel=True))

    def are_chain_dependencies_resolved(self, chain):
        """
        Evaluates whether all dependencies for a parallel chain of jobs are resolved.

        This method checks the dependency status of a parallel job chain by identifying external predecessors
        and verifying if they have completed. It is essential for managing the execution order in a workflow
        where jobs have interdependencies.

        Args:
            chain (list): A list of job IDs representing a parallel chain in the workflow.

        Returns:
            bool: True if all external dependencies of the chain are resolved, allowing the chain's execution.
                False if any of the external dependencies are yet to be completed.
        """
        # Find all external predecessors of the chain
        external_predecessors = self.find_all_external_predecessors(chain)

        # If there are no external predecessors, the chain is independent and can be executed
        if not external_predecessors:
            return True

        # Check if all external predecessors have completed
        for predecessor in external_predecessors:
            if not self.job_events[predecessor].triggered:
                return False

        # If all external predecessors are completed, return True
        return True

    def run_job(self, job_id, is_parallel=False, cluster=None, placement=None,
                use_bb=None):
        """
        Manages the execution of a specific job within the workflow, considering dependencies, placement, and burst buffer usage.

        This method handles the execution of a job by considering its dependencies (including delay dependencies),
        optimizing placement and burst buffer usage, and managing the lifecycle of the job within the simulation environment.

        Args:
            job_id (str): The identifier for the job to be executed.
            is_parallel (bool, optional): Indicates whether the job is part of a parallel chain of execution. Defaults to False.
            cluster (Cluster, optional): The cluster on which the job will be run. If not specified, the default cluster is used.
            placement (list, optional): The placement strategy for the job's data. If not specified, default placement is used.
            use_bb (list, optional): A list indicating whether to use burst buffer for each phase of the job. Defaults to not using burst buffer.

        Yields:
            simpy.Event: An event that is triggered when the job execution is completed.
        """
        logger.debug(f"Preparing to run job {job_id}.")

        # Determine the cluster for execution
        cluster = cluster or self.cluster

        # Fetch job details and optimize placement and burst buffer usage
        job = self.jobs[job_id]
        placement = self.jobs_placements.get(job_id, {}).get('placement', [0])
        use_bb = self.jobs_placements.get(job_id, {}).get('use_bb', [False])
        placement, use_bb = self.normalize_placement_and_bb(job, placement, use_bb)

        logger.debug(f"Running job {job_id} on cluster {cluster} with placement {placement} and burst buffer usage {use_bb}.")


        # Handle delay dependencies
        predecessors = list(self.graph.predecessors(job_id))
        if predecessors:
            predecessor = predecessors[0]
            if self.graph[predecessor][job_id].get('type') == 'delay':
                delay = self.graph[predecessor][job_id].get('delay', 0)
                if delay > 0:
                    logger.debug(f"Delay dependency detected. Waiting for {delay} time units before executing job {job_id}.")
                    yield self.env.timeout(delay)


        # Execute the job
        yield self.env.process(job.run(cluster=cluster, placement=placement,
                                       use_bb=use_bb))

        # Trigger completion event for the job
        if not self.job_events[job_id].triggered:
            self.job_events[job_id].succeed()

        logger.debug(f"Job {job_id} execution completed.")
        # # Trigger the job's event at the start for parallel jobs
        # if is_parallel and not self.job_events[job_id].triggered:
        #         self.job_events[job_id].succeed()
        # # and wait for the job to complete


        # # Trigger the job's event to signal completion, ensuring it's the correct type
        # # if isinstance(self.job_events[job_id], simpy.events.Event) and not self.job_events[job_id].triggered:
        # if not is_parallel and not self.job_events[job_id].triggered:
        #     self.job_events[job_id].succeed()

        # # Determine if any successor job requires this job to be completed in a sequential manner
        # sequential_successor = any(dependency_type == 'sequential' for _, successor, dependency_type in self.dependencies if _ == job_id)
        # if not self.job_events[job_id].triggered:
        #     self.job_events[job_id].succeed()

    def normalize_placement_and_bb(self, job, placement, use_bb):
        """
        Normalizes the placement and burst buffer (use_bb) configurations for a job to ensure that their lengths match the number of phases in the job.

        The method extends or truncates the 'placement' and 'use_bb' lists to match the number of compute phases of the job. If 'placement' or 'use_bb' is provided as a single value (not a list), it is converted into a list with that value repeated for each compute phase.

        Args:
            job (Job): The job object, which must have a 'compute' attribute representing its compute phases.
            placement (int/list): The initial data placement strategy for the job. Can be a single integer or a list of integers representing the placement for each compute phase.
            use_bb (bool/list): Indicates whether to use a burst buffer for each phase of the job. Can be a single boolean or a list of booleans.

        Returns:
            tuple: A tuple containing two lists:
                - The first list represents the normalized placement for each compute phase of the job.
                - The second list represents the normalized burst buffer usage for each compute phase.

        Raises:
            TypeError: If 'placement' or 'use_bb' is neither a single value nor a list.

        Example:
            >>> job = Job(...)
            >>> workflow = Workflow(...)
            >>> normalized_placement, normalized_use_bb = workflow.normalize_placement_and_bb(job, 1, True)
            >>> print(normalized_placement, normalized_use_bb)
            [1, 1, 1] [True, True, True]
        """
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
        """
        Executes the workflow by setting up job dependencies and running the simulation.

        This method initializes the job placements (if provided), sets up the job dependencies based on the defined workflow,
        and then runs the simulation environment to execute the workflow.

        Args:
            jobs_placements (dict, optional): A dictionary mapping job IDs to their placement strategies and burst buffer usage.
            Each entry in the dictionary should have the format:
            {job_id: {'placement': [list_of_placement_indices], 'use_bb': [list_of_boolean_values]}}
            If not provided, the default placements and burst buffer usages are used.
        """
        # Update job placements if provided
        if jobs_placements:
            self.jobs_placements = jobs_placements

        # Set up job dependencies before starting the simulation
        self.setup_dependencies()

        # Run the simulation environment
        self.env.run()

