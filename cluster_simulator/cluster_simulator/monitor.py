import simpy
from functools import partial, wraps
from loguru import logger
from phase import IOPhase
from cluster_simulator.phase import IOPhase


# def check_bandwidth(env, bandwidth):
#     """Checks running IO when bandwidth occupation changes. IOs should be interrupted on release or request of a bandwidth slot.
#     """
#     logger.info("checking...")
#     for io_event in IO.current_ios:
#         if not io_event.triggered:  # capture the IOs not finished
#             logger.info(f"still running IO : {io_event}")


# class MonitorResource(simpy.Resource):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.env = args[0]

#         #self.data = []

#     def request(self, *args, **kwargs):

#         ret = super().request(*args, **kwargs)
#         check_bandwidth(self.env, self)
#         # logger.info(f"[req]Currently used resources at {self.env.now}: {self.count} out of {self.capacity}")
#         #self.data.append((self._env.now, self.count))
#         return ret

#     def release(self, *args, **kwargs):

#         ret = super().release(*args, **kwargs)
#         logger.info(f"[release]Currently used resources at {self.env.now}: {self.count} out of {self.capacity}")
#         for user in self.users:
#             logger.info(f"User: {user} using resource since {user.usage_since}")
#         #self.data.append((self._env.now, self.count))
#         return ret

# def trace(env, callback):
#     """Replace the ``step()`` method of *env* with a tracing function
#     that calls *callbacks* with an events time, priority, ID and its
#     instance just before it is processed.

#     """
#     def get_wrapper(env_step, callback):
#         """Generate the wrapper for env.step()."""
#         @wraps(env_step)
#         def tracing_step():
#             """Call *callback* for the next event if one exist before
#             calling ``env.step()``."""
#             if len(env._queue):
#                 t, prio, eid, event = env._queue[0]
#                 callback(t, prio, eid, event)
#             return env_step()
#         return tracing_step

#     env.step = get_wrapper(env.step, callback)

#     def monitor(data, t, prio, eid, event):
#         data.append((t, eid, type(event)))

#     def test_process(env):
#         yield env.timeout(1)


# def monitor(data, resource):
#     """This is our monitoring callback."""
#     item = (
#         resource._env.now,  # The current simulation time
#         resource.count,  # The number of users
#     )
#     data.append(item)


# def test_process(env, res):
#     with res.request() as req:
#         yield req
#         yield env.timeout(1)


if __name__ == '__main__':
    #     env = simpy.Environment()
    #     res = simpy.Resource(env, capacity=1)
    #     data = []
    #     # Bind *data* as first argument to monitor()
    #     # see https://docs.python.org/3/library/functools.html#functools.partial
    #     monitor = partial(monitor, data)
    #     patch_resource(res, post=monitor)  # Patches (only) this resource instance

    #     p = env.process(test_process(env, res))
    #     env.run(p)
    #     print(data)
    name = ''
    if not name:
        print("dont see")
