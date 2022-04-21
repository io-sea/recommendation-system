import simpy
from loguru import logger
import numpy as np
import pandas as pd
import math
from cluster import Cluster, Tier, bandwidth_share_model, compute_share_model, get_tier, convert_size
from phase import DelayPhase, ComputePhase, IOPhase, name_app
import copy
import time
import analytics
from application import Application

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
    def __init__(self):
        self.env = simpy.Environment()
        self.data = simpy.Store(self.env)

        self.nvram_bandwidth = {'read':  {'seq': 780, 'rand': 760},
                                'write': {'seq': 515, 'rand': 505}}
        self.ssd_bandwidth = {'read':  {'seq': 210, 'rand': 190},
                              'write': {'seq': 100, 'rand': 100}}

        self.ssd_tier = Tier(self.env, 'SSD', bandwidth=self.ssd_bandwidth, capacity=200e9)
        self.nvram_tier = Tier(self.env, 'NVRAM', bandwidth=self.nvram_bandwidth, capacity=80e9)
        self.cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=5,
                               tiers=[self.nvram_tier, self.ssd_tier])

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
        self.ios = self.get_io_nbr()
        self.n_tiers = len(self.cluster.tiers)
        self.parameter_space = np.array([np.arange(0, self.n_tiers, 1)]*sum(self.ios))

    def get_io_nbr(self):
        io_app = []
        for app in self.apps:
            io_app.append(len([io for io in app.read if io > 0]) +
                          len([io for io in app.write if io > 0]))
        return io_app

    def compute(self, placement=None):  # np.array([[0, 1], [0, 1]])
        self.__init__()  # https://stackoverflow.com/questions/45061369/simpy-how-to-run-a-simulation-multiple-times
        start_index = 0
        print(placement)
        for i_app, app in enumerate(self.apps):
            place_tier = placement[start_index: start_index + self.ios[i_app]]
            start_index = self.ios[i_app]
            self.env.process(app.run(self.cluster, tiers=place_tier))
        # run the simulation
        self.env.run()
        return app.get_fitness()


if __name__ == '__main__':
    logger.remove()
    cbb = ClusterBlackBox()
    PARAMETER_SPACE = cbb.parameter_space
    # combinations are self.n_tiers ** sum(self.ios)
    NBR_ITERATION = cbb.n_tiers ** sum(cbb.ios)

    np.random.seed(5)
    bbopt = BBOptimizer(black_box=cbb,
                        heuristic="surrogate_model",
                        max_iteration=NBR_ITERATION,
                        initial_sample_size=5,
                        parameter_space=PARAMETER_SPACE,
                        next_parameter_strategy=expected_improvement,
                        regression_model=GaussianProcessRegressor)
    start_time = time.time()
    bbopt.optimize()
    print("-----------------")
    print(NBR_ITERATION)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("-----------------")
    bbopt.summarize()
    print(bbopt.history["fitness"])
