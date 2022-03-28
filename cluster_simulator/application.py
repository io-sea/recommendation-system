import simpy
from cluster import Cluster


class Application:
    def __init__(self, compute=[0, 10], read=[1e9, 0], write=[0, 5e9]):
        self.compute = compute
        self.read = read
        self.write = write

    def run(self, env, cluster):
