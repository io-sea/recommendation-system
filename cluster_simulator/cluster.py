import simpy

class Cluster:
    def __init__(self, env, compute_nodes = 1, storage_capacity=1e10, storage_speed = 2e6):
        
        self.compute_nodes = simpy.Resource(env, capacity=compute_nodes)
        self.storage_capacity = simpy.Container(env, init=0, capacity=storage_capacity)
        self.storage_speed = simpy.Container(env, init=storage_speed, capacity=storage_speed)
        
        
        
def run_compute_phase(cluster, env, duration):
    """running a compute phase by exploiting a compute node"""
    
    
def run(cluster, env):
    # Simulate the following events
    # Compute time of 10, exploiting one compute node out of 2 in the cluster
    # Writing 5e9 into the storage at full available speed
    with cluster.compute_nodes.request() as req:
        yield req
        print(f"Start computing phase at {env.now}")
        yield env.timeout(10)
        print(f"End computing phase at {env.now}")           
    
    with cluster.compute_nodes.request() as req:
        yield req            
        print(f"Start requesting bandwidth for I/O at {env.now}")
        bandwidth_share = cluster.storage_speed.capacity
        print(f"Exploiting bandwidth for I/O : {bandwidth_share} out of {cluster.storage_speed.capacity}")
        yield cluster.storage_speed.get(bandwidth_share)  
        print(f"Start I/O writing phase at {env.now} for 5e9")      
        io_time = 5e9/bandwidth_share/1000
        print(f"I/O writing phase duration is {io_time}")
        yield cluster.storage_capacity.put(5e9)
        yield env.timeout(io_time)
        yield cluster.storage_speed.put(bandwidth_share)
        print(f"Finish I/O writing phase at {env.now} for 5e9") 
        print(f"Storage State at {env.now} is {cluster.storage_capacity.level} out of {cluster.storage_capacity.capacity}") 


if __name__ == '__main__':
    env = simpy.Environment()
    cluster = Cluster(env)
    env.process(run(cluster, env))
    env.run(until=3500)
    # print(cluster.storage_speed.capacity)
    # print(cluster.storage_speed.level)
    # print(cluster.storage_speed.get_queue)
    # https://stackoverflow.com/questions/48738371/
    # https://simpy.readthedocs.io/en/latest/topical_guides/resources.html#res-type-container
    # simpy-requesting-multiple-nonspecific-resources-and-order-of-requests
    
        
    
    #env.process(car(env))
    #env.run(until=15)