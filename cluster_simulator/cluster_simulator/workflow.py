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

    def update_job_events(self):
        """Update the job_events dictionary based on the current graph state."""
        for job_id in self.graph.nodes():
            if job_id not in self.job_events:
                self.job_events[job_id] = self.env.event()

    def setup_dependencies(self):
        """
        Set up dependencies between jobs, identifying independent and parallel chains and scheduling them.
        """
        logger.info("Setting up job dependencies.")

        # Initialisation
        scheduled_jobs = set()
        job_queue = []

        # Planification des Jobs Indépendants
        for job_id in self.jobs:
            if self.is_independent_job(job_id):
                logger.debug(f"Scheduling independent job: {job_id}")
                self.env.process(self.run_job(job_id))
                scheduled_jobs.add(job_id)

        # Identification et Planification des Chaînes Parallèles
        parallel_chains = self.find_parallel_chains()
        for chain in parallel_chains:
            self.schedule_parallel_chain(chain)
            scheduled_jobs.update(chain)

        # Planification des Jobs avec Dépendances
        for job_id in self.jobs:
            if job_id not in scheduled_jobs:
                if self.are_chain_dependencies_resolved([job_id]):
                    self.schedule_job_with_dependencies(job_id)
                    scheduled_jobs.add(job_id)
                else:
                    job_queue.append(job_id)

        # Gestion des Jobs en Attente
        while job_queue:
            logger.debug(f"Job queue: {job_queue}")
            remaining_queue = []
            for job_id in job_queue:
                if self.are_chain_dependencies_resolved([job_id]):
                    self.schedule_job_with_dependencies(job_id)
                    scheduled_jobs.add(job_id)
                else:
                    remaining_queue.append(job_id)
            job_queue = remaining_queue

        logger.info("Job dependencies setup is complete.")


    def setup_dependencies(self):
        """Set up dependencies between jobs, identifying independent and parallel chains and scheduling them."""
        logger.info("Setting up job dependencies.")
        def add_to_queue(queue, elements):
            """
            Ajoute des éléments à la file d'attente. Accepte un élément individuel ou une liste d'éléments.

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
            Retire des éléments de la file d'attente. Accepte un élément individuel ou une liste d'éléments.

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
                    self.schedule_parallel_chain(chain)
                    # Add all jobs in the chain to the scheduled set
                    scheduled_in_parallel.update(chain)
                    remove_from_queue(job_queue, chain)

                elif not self.is_in_parallel_chain(job_id):
                    logger.debug(f"Scheduling independent job {job_id} for immediate execution.")
                    remove_from_queue(job_queue, job_id)
                    self.env.process(self.run_job(job_id))

        # Step 2: Schedule parallel chains that have not been scheduled yet
        parallel_chains = self.find_parallel_chains()
        for chain in parallel_chains:
            # Check any parallel chain that has not been scheduled yet has all its dependencies resolved
            if not scheduled_in_parallel.intersection(set(chain)):
                if self.are_chain_dependencies_resolved(chain):
                    logger.debug(f"Dependencies resolved for parallel chain: {chain}. Scheduling it.")
                    remove_from_queue(job_queue, chain)
                    self.schedule_parallel_chain(chain)
                    scheduled_in_parallel.update(chain)
                else:
                    logger.debug(f"Dependencies not yet resolved for parallel chain: {chain}. Delaying scheduling.")

        # Step 3: Schedule remaining jobs considering their dependencies
        while job_queue:
            logger.debug(f"Job queue: {job_queue}")
            for job_id in job_queue:




                if job_id not in scheduled_in_parallel and not self.is_independent_job(job_id):
                    self.schedule_job_with_dependencies(job_id)
                    remove_from_queue(job_queue, job_id)
                # TODO : This is a temporary fix for the case where a job is scheduled in a parallel chain
                if self.is_in_parallel_chain(job_id):
                    chain = self.find_parallel_chain(job_id, set())
                    if not scheduled_in_parallel.intersection(set(chain)):
                        self.schedule_parallel_chain_with_dependencies(chain)
                        scheduled_in_parallel.update(chain)
                        remove_from_queue(job_queue, chain)


        # for job_id in self.jobs:
        #     if job_id not in scheduled_in_parallel and not self.is_independent_job(job_id):
        #         logger.debug(f"Scheduling job {job_id} with dependencies.")
        #         self.schedule_job_with_dependencies(job_id)

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


    def schedule_parallel_chain_with_dependencies(self, chain):
        """
        Schedule a parallel chain of jobs taking into account their dependencies.

        Args:
            chain (list[str]): The list of job IDs in the parallel chain.
        """
        logger.debug(f"Scheduling parallel chain with dependencies: {chain}")

        # Iterate over jobs in the chain
        for job_id in chain:
            # Retrieve the predecessor, if any
            predecessor = next((pred for pred in self.graph.predecessors(job_id) if pred not in chain), None)

            # If there is a predecessor outside the chain, wait for it to complete before scheduling this job
            if predecessor:
                logger.debug(f"Job {job_id} in the chain has a predecessor: {predecessor}")
                pre_job_event = self.job_events[predecessor]

                # Define a process to wait for the predecessor to complete
                def wait_and_run(env, pre_job_event, job_id):
                    yield pre_job_event
                    logger.debug(f"Predecessor {predecessor} completed, now running job {job_id} in the chain")
                    yield env.process(self.run_job(job_id))

                # Add the process to the simulation environment
                self.env.process(wait_and_run(self.env, pre_job_event, job_id))
            else:
                # If there is no predecessor or predecessor is in the same chain, schedule the job immediately
                logger.debug(f"Job {job_id} in the chain has no external predecessors, scheduling immediately")
                self.env.process(self.run_job(job_id))


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

    def wait_and_schedule_job(self, job_id, predecessor_id):
        """
        Wait for the predecessor job to complete before scheduling the current job.

        Args:
            job_id (str): The ID of the job to schedule.
            predecessor_id (str): The ID of the predecessor job.
        """
        logger.debug(f"Job {job_id} is waiting for predecessor {predecessor_id} to complete.")

        # Define a process to wait for the predecessor to complete
        def wait_for_predecessor(env, pre_job_event):
            yield pre_job_event
            logger.debug(f"Predecessor {predecessor_id} completed. Now scheduling job {job_id}.")
            yield env.process(self.run_job(job_id))

        # Get the event associated with the predecessor job
        pre_job_event = self.job_events[predecessor_id]

        # Add the process to the simulation environment
        self.env.process(wait_for_predecessor(self.env, pre_job_event))

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
            # Retrieve all predecessors of the job
            predecessors = list(self.graph.predecessors(job_id))

            # If there are no predecessors, the job is independent and can be executed
            if not predecessors:
                return True

            # Check if all predecessors have completed
            for predecessor in predecessors:
                if not self.job_events[predecessor].triggered:
                    return False

        # If no job in the chain is independent and some have uncompleted predecessors, return False
        return False

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
    def check_dependencies_resolved(self, job_id):
        """
        Checks if all dependencies of a job are resolved.

        Args:
            job_id (str): The identifier of the job.

        Returns:
            bool: True if all dependencies are resolved, False otherwise.
        """
        for predecessor, _, dependency_type in self.dependencies:
            if _ == job_id and dependency_type in ['sequential', 'delayed']:
                if not self.job_events[predecessor].triggered:
                    return False
        return True

    def reschedule_parallel_chain(self, job_id):
        """
        Reschedules a parallel chain if its dependencies are now resolved.

        Args:
            job_id (str): The identifier of the job that just completed.
        """
        for predecessor, successor, dependency_type in self.dependencies:
            if predecessor == job_id and dependency_type == 'parallel':
                if self.check_dependencies_resolved(successor):
                    # All dependencies of the parallel chain are resolved
                    self.schedule_parallel_chain(successor)

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
        yield self.env.process(self.jobs[job_id].run(cluster=cluster,
                                                     placement=placement,
                                                     use_bb=use_bb))

        # After job completion, check if it was a dependency for any parallel chain
        self.reschedule_parallel_chain(job_id)

        # Trigger the job's event at the start for parallel jobs
        if is_parallel and not self.job_events[job_id].triggered:
                self.job_events[job_id].succeed()
        # and wait for the job to complete


        # Trigger the job's event to signal completion, ensuring it's the correct type
        # if isinstance(self.job_events[job_id], simpy.events.Event) and not self.job_events[job_id].triggered:
        if not is_parallel and not self.job_events[job_id].triggered:
            self.job_events[job_id].succeed()

        # Determine if any successor job requires this job to be completed in a sequential manner
        sequential_successor = any(dependency_type == 'sequential' for _, successor, dependency_type in self.dependencies if _ == job_id)
        if not self.job_events[job_id].triggered:
            self.job_events[job_id].succeed()



    def check_and_schedule_dependents(self, job_id):
        """
        Check and schedule dependent jobs after the completion of a given job.

        Args:
            job_id (str): The identifier of the completed job.
        """
        logger.debug(f"Checking dependents for job {job_id}.")

        for dependent_id in self.graph.successors(job_id):
            logger.debug(f"Checking if dependent job {dependent_id} is ready to be scheduled.")
            # Determine if this dependent job is ready to be scheduled
            if self.is_job_ready_to_schedule(dependent_id):
                logger.debug(f"Dependent job {dependent_id} is ready. Scheduling with dependencies.")
                self.schedule_job_with_dependencies(dependent_id)
            else:
                logger.debug(f"Dependent job {dependent_id} is not ready to be scheduled yet.")

    def is_job_ready_to_schedule(self, job_id):
        """
        Determine if a job is ready to be scheduled based on its dependencies.

        Args:
            job_id (str): The identifier of the job to check.

        Returns:
            bool: True if the job is ready to be scheduled, False otherwise.
        """
        logger.debug(f"Checking if job {job_id} is ready to be scheduled based on dependencies.")
        for predecessor_id in self.graph.predecessors(job_id):
            if not self.job_events[predecessor_id].triggered:
                logger.debug(f"Job {job_id} is not ready to be scheduled. Waiting for predecessor {predecessor_id}.")
                return False
        logger.debug(f"All dependencies resolved for job {job_id}. It is ready to be scheduled.")
        return True

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
