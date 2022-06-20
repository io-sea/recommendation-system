# imports
import simpy
from loguru import logger
import time
import numpy as np
import pandas as pd
import time, os
from cluster_simulator.utils import get_ephemeral_size, get_fitness
from itertools import chain
from cluster_simulator.cluster import Cluster, Tier, EphemeralTier, bandwidth_share_model, compute_share_model, get_tier, convert_size
from cluster_simulator.phase import DelayPhase, ComputePhase, IOPhase
from cluster_simulator.application import Application
from cluster_simulator.analytics import display_run

# imports for surrogate models
from sklearn.gaussian_process import GaussianProcessRegressor
from bbo.optimizer import BBOptimizer
# from bbo.optimizer import timeit
from bbo.heuristics.surrogate_models.next_parameter_strategies import expected_improvement

# imports for genetic algorithms
from bbo.heuristics.genetic_algorithm.selections import tournament_pick
from bbo.heuristics.genetic_algorithm.crossover import double_point_crossover
from bbo.heuristics.genetic_algorithm.mutations import mutate_chromosome_to_neighbor
from loguru import logger


class PlacementOptimizer:
    # store optimization curve
    experiment_data = []
    def __init__(self, env, data, cluster, apps):
        self.env = env
        self.data = data
        self.cluster= cluster
        self.tiers = cluster.tiers
        for tier in self.tiers:
            tier.env = self.env
        self.ephemeral_tier = cluster.ephemeral_tier
        cluster.ephemeral_tier.env = self.env
        self.apps = apps
        for app in self.apps:
            # reinitiate env and data
            app.env = self.env
            app.data = self.data


        self.ios = self.get_io_list()
        self.n_tiers = len(self.cluster.tiers)
        #self.parameter_space = np.array([np.arange(0, self.n_tiers, 1)]*sum(self.ios))
        self.parameter_space = np.array([np.arange(0, self.n_tiers, 1), np.arange(0, 2, 1)]*sum(self.ios))

    def get_io_list(self):
        """Get the list of I/O operations for all applications.
        Example:
            [2, 3, 2] for three apps having respectively 2, 3 and 2 IOs.
        Returns:
            io_app (list): list of len equal to the number of apps. Each element of the list is equal to the number of io to place.
        """
        io_app = []
        for app in self.apps:
            io_app.append(len([io for io in app.read if io > 0]) +
                        len([io for io in app.write if io > 0]))
        return io_app

    def get_max_io_volume(self):
        """Get the total number of I/O operations for all applications"""
        return sum(sum(app.read) + sum(app.write) for app in self.apps)

    def compute(self, placement=None, force_bb_placement=None):  # np.array([[0, 1], [0, 1]]) for two apps
        self.__init__(self.env, self.data, self.cluster, self.apps)
        # https://stackoverflow.com/questions/45061369/simpy-how-to-run-a-simulation-multiple-times
        start_index = 0
        print(f"Full BBO param array = {placement}")
        row = [placement]
        for i_app, app in enumerate(self.apps):
            tier_placement = placement[0:sum(self.ios)] # get the placement param
            bb_placement = placement[sum(self.ios)+1:] # get the use_bb param
            place_tier = tier_placement[start_index: start_index + self.ios[i_app]]
            # converting list of 0/1 values to False/True
            use_bb = list(map(bool, bb_placement[start_index: start_index + self.ios[i_app]]))
            start_index = self.ios[i_app]
            # if indicated, force bb placement
            if force_bb_placement:
                use_bb = force_bb_placement[start_index: start_index + self.ios[i_app]]
            self.env.process(app.run(self.cluster, placement=place_tier, use_bb=use_bb))
            row += [place_tier, use_bb]
            print(f"    | app#{self.apps[i_app].name} : tier placement: {place_tier} | use_bb = {use_bb}")

        # run the simulation
        self.env.run()
        fitness = get_fitness(self.data)
        bb_size = get_ephemeral_size(self.data)
        print(f"    | runtime = {fitness} |  BB_size = {convert_size(bb_size)}")
        row += [fitness, bb_size]
        self.experiment_data.append(row)
        return get_fitness(self.data)

    def display_placement(self, placement):
        self.__init__(self.env, self.data, self.cluster, self.apps)
        start_index = 0
        print(f"Displaying result for placement parameter = {placement}")
        for i_app, app in enumerate(self.apps):
            tier_placement = placement[0:sum(self.ios)] # get the placement param
            bb_placement = placement[sum(self.ios):] # get the use_bb param
            place_tier = tier_placement[start_index: start_index + self.ios[i_app]]
            # converting list of 0/1 values to False/True
            use_bb = list(map(bool, bb_placement[start_index: start_index + self.ios[i_app]]))
            start_index = self.ios[i_app]
            self.env.process(app.run(self.cluster, placement=place_tier, use_bb=use_bb))
            print(f"    | app#{self.apps[i_app].name} : tier placement: {place_tier} | use_bb = {use_bb}")

        # run the simulation
        self.env.run()
        fitness = get_fitness(self.data)
        bb_size = get_ephemeral_size(self.data)
        print(f"    | runtime = {fitness} |  BB_size = {convert_size(bb_size)}")
        fig = display_run(self.data, self.cluster, width=800, height=900)
        return fig
