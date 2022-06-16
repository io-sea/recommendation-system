# imports
import simpy
from loguru import logger
import time
import numpy as np
import pandas as pd
import time, os
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


class ClusterBlackBox:
    # store optimization curve
    experiment_data = []
    def __init__(self):
        self.env = simpy.Environment()
        self.data = simpy.Store(self.env)

        self.nvram_bandwidth = {'read':  {'seq': 780, 'rand': 760},
                                'write': {'seq': 515, 'rand': 505}}
        self.ssd_bandwidth = {'read':  {'seq': 210, 'rand': 190},
                              'write': {'seq': 100, 'rand': 100}}
        self.hdd_bandwidth = {'read':  {'seq': 80, 'rand': 80},
                         'write': {'seq': 40, 'rand': 40}}

        self.ssd_tier = Tier(self.env, 'SSD', bandwidth=self.ssd_bandwidth, capacity=200e9)
        self.hdd_tier = Tier(self.env, 'HDD', bandwidth=self.hdd_bandwidth, capacity=1e12)
        #self.nvram_tier = Tier(self.env, 'NVRAM', bandwidth=self.nvram_bandwidth, capacity=80e9)

        app1 = Application(self.env,
                           compute=[0, 10],
                           read=[1e9, 0],
                           write=[0, 5e9],
                           data=self.data)
        app2 = Application(self.env,
                           compute=[0, 20, 30],
                           read=[3e9, 0, 0],
                           write=[0, 5e9, 10e9],
                           data=self.data)
        app3 = Application(self.env,
                           compute=[0, 10],
                           read=[4e9, 0],
                           write=[0, 7e9],
                           data=self.data)

        self.apps = [app1, app2, app3]

        self.bb = EphemeralTier(self.env, name="BB", persistent_tier=self.ssd_tier,
                                bandwidth=self.nvram_bandwidth, capacity=50e9)
                                #capacity=self.get_max_io_volume())
        self.cluster = Cluster(self.env, compute_nodes=2, cores_per_node=5,
                               tiers=[self.hdd_tier, self.ssd_tier],
                               ephemeral_tier=self.bb)


        self.ios = self.get_io_nbr()
        self.n_tiers = len(self.cluster.tiers)
        #self.parameter_space = np.array([np.arange(0, self.n_tiers, 1)]*sum(self.ios))
        self.parameter_space = np.array([np.arange(0, self.n_tiers, 1), np.arange(0, 2, 1)]*sum(self.ios))





    def get_io_nbr(self):
        """Get the total number of I/O operations for all applications
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

    def compute(self, placement=None):  # np.array([[0, 1], [0, 1]]) for two apps
        self.__init__()
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
            self.env.process(app.run(self.cluster, placement=place_tier, use_bb=use_bb))
            row += [place_tier, use_bb]
            print(f"    | app#{self.apps[i_app].name} : tier placement: {place_tier} | use_bb = {use_bb}")

        # run the simulation
        self.env.run()
        fitness = app.get_fitness()
        bb_size = app.get_ephemeral_size()
        print(f"    | runtime = {fitness} |  BB_size = {convert_size(bb_size)}")
        row += [fitness, bb_size]
        self.experiment_data.append(row)
        return app.get_fitness()
    # def display_placement(self, placement):
    #     self.__init__()
    #     # https://stackoverflow.com/questions/45061369/simpy-how-to-run-a-simulation-multiple-times
    #     start_index = 0
    #     #print(placement)
    #     for i_app, app in enumerate(self.apps):
    #         place_tier = placement[start_index: start_index + self.ios[i_app]]
    #         start_index = self.ios[i_app]
    #         self.env.process(app.run(self.cluster, placement=place_tier))
    #     # run the simulation
    #     self.env.run()
    #     fig = display_run(self.data, self.cluster, width=800, height=900)
    #     fitness = app.get_fitness()
    #     appslist = ", ".join([app.name for app in self.apps])
    #     print(f"The apps {appslist} lasts {round(fitness, 3)} seconds when placement = {placement}")
    #     return fig
    def display_placement(self, placement):
        self.__init__()
        start_index = 0
        print(f"Displaying result for placement parameter = {placement}")
        for i_app, app in enumerate(self.apps):
            tier_placement = placement[0:sum(self.ios)] # get the placement param
            bb_placement = placement[sum(self.ios)+1:] # get the use_bb param
            place_tier = tier_placement[start_index: start_index + self.ios[i_app]]
            # converting list of 0/1 values to False/True
            use_bb = list(map(bool, bb_placement[start_index: start_index + self.ios[i_app]]))
            start_index = self.ios[i_app]
            self.env.process(app.run(self.cluster, placement=place_tier, use_bb=use_bb))
            print(f"    | app#{i_app+1} : tier placement: {place_tier} | use_bb = {use_bb}")

        # run the simulation
        self.env.run()
        fitness = app.get_fitness()
        bb_size = app.get_ephemeral_size()
        print(f"    | runtime = {fitness} |  BB_size = {convert_size(bb_size)}")
        fig = display_run(self.data, self.cluster, width=800, height=900)
        return fig


    def save_experiment(self, filename=str(round(time.time()*1000)), save=False):
        """Organise a dataframe with adequate number of columns to save experiment results.

        Example:
            param | app#1 tier place | app#1 use bb | ... | fitness | bb_size

        Args:
            row_list (list): parameter for black box optimization

        Returns:
            df (dataframe): dataframe containing the results of iterations over parameters.
        """

        columns = ["Param"] + list(chain.from_iterable((f"App#{self.apps[i].name} tier place",
                                                        f"App#{self.apps[i].name} use bb") for i in range(len(self.apps)))) + ["Fitness", "BB_size"]

        self.df = pd.DataFrame(self.experiment_data, columns=columns)
        if save:
            pathfile = os.path.join(os.getcwd(), "notebooks", filename)
            self.df.to_pickle(pathfile)
            print(f"dataframe saved to {pathfile}")
        else:
            return self.df



if __name__ == '__main__':
    logger.remove()
    cbb = ClusterBlackBox()
    PARAMETER_SPACE = cbb.parameter_space
    # combinations are self.n_tiers ** sum(self.ios) + 2**sum(self.ios)
    NBR_ITERATION = 50  # cbb.n_tiers ** sum(cbb.ios)

    np.random.seed(5)
    bbopt = BBOptimizer(black_box=cbb,
                        heuristic="surrogate_model",
                        max_iteration=NBR_ITERATION,
                        initial_sample_size=80,
                        parameter_space=PARAMETER_SPACE,
                        next_parameter_strategy=expected_improvement,
                        regression_model=GaussianProcessRegressor)
    start_time = time.time()
    bbopt.optimize()
    print("-----------------")
    print(f"Total number of iterations : {NBR_ITERATION}")
    print(f"{(time.time() - start_time)} seconds spent for finding solution")
    print("-----------------")
    bbopt.summarize()
    print(f"Fitness history : {bbopt.history['fitness']}")
    cbb.save_experiment(filename = "flavor_optim", save=True)
