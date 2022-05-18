"""
Simulation of a dynamic server pool

Server pool starts empty and servers are added as needed, but there is a delay
simulating start up time before the server is available to fulfill a resource request
After a server is started a check is made to see if the server is still needed before
it is added to the resouce pool

Programmer: Matt
    Wrote original version

Programmer: Michael R. Gibbs
    Added server check for dynamicaly adding servers
    servers are returned to resouce pool only if needed (get queue size > 0)
    
    https://stackoverflow.com/questions/67757880/interrupt-an-earlier-timeout-event-in-simpy
    https://stackoverflow.com/questions/41789294/interrupt-conditional-event-in-simpy
    Research keywords: "Interrupting, and yielding processes using resources"
"""

import simpy
import numpy as np

LAM = 8  # arival rate of jobs
MU = 2  # service rate
ALPHA = 12  # set up rate
NUM_SERVERS = 5
MAX_NUM_JOB = 50  # 10000000000
UNTIL = 10

server_cnt = 0
job_cnt = 0
start_up_list = []


def generate_interarrival():
    return np.random.exponential(1/LAM)


def generate_service():
    return np.random.exponential(1/MU)


def switch_on():
    return np.random.exponential(1/ALPHA)


def return_server(env, servers):
    """
    checks if the server is still needed,
    if so add back to the resource pool so waiting request can be filled
    else, do not add back to resource pool simulating shutdown
    """

    global server_cnt, start_up_list, job_cnt

    if len(servers.get_queue) > 0:
        # server is still needed
        yield servers.put(1)
        print('{0:.5f}'.format(env.now), "queuing server --")

        if server_cnt > job_cnt:
            # have a extra server, try to kill starting up server

            # first clean up events that have already happend
            i = len(start_up_list)-1
            while i >= 0:
                e = start_up_list[i]
                if e.triggered:
                    start_up_list.pop(i)
                i -= 1

            # kill last added startup process hoping that is the one with longest time before start up finishes
            if len(start_up_list) > 0:
                e = start_up_list.pop()
                e.interrupt()
                print('{0:.5f}'.format(env.now), "killing start up server --------------------------------")
    else:
        print('{0:.5f}'.format(env.now), "shutting down server --")
        server_cnt -= 1


def check_servers(env, servers):
    """
    Checks the server pool to see if the pool has any avalable servers
    if not then add a server, (there will be a delay before added server becomes available)

    after the start up delay, check again to see if the server is still needed

    Call this without a yield so it does not block if a server is added
    """

    global server_cnt

    print('{0:.5f}'.format(env.now), "checking server pool", "requests:", len(servers.get_queue), "idel:", servers.level, "servers:", server_cnt)

    if len(servers.get_queue) >= servers.level and server_cnt < NUM_SERVERS:
        # will need another server
        server_cnt += 1
        d = switch_on()
        startT = env.now + d

        print('{0:.5f}'.format(env.now), "adding a server at " + '{0:.5f}'.format(startT) + " --")

        try:  # catch interrupts exceptions
            # start up
            yield env.timeout(d)  # switch on time

            # check if server is still needed
            if len(servers.get_queue) > 0:
                # still need it so add
                yield servers.put(1)
                print('{0:.5f}'.format(env.now), "added a server--")
            else:
                print('{0:.5f}'.format(env.now), "server not needed, not added--")
                server_cnt -= 1
        except:
            server_cnt -= 1
            print('{0:.5f}'.format(env.now), "server starting at " + '{0:.5f}'.format(startT) + " has been killed --")


class Generate_Job():
    def arriving_job(env, servers):
        global num_current_jobs, num_server_on, leaving_time_list
        for i in range(MAX_NUM_JOB):
            job = Job(name="Job%01d" % (i))
            yield env.timeout(generate_interarrival())
            print('{0:.5f}'.format(env.now), job.name, "arrives")

            env.process(job.handling(env, servers))


class Room:                             # A room containing servers (resource)
    def __init__(self, env):
        self.computer = simpy.Container(env, capacity=10000, init=0)


class Job(object):
    def __init__(self, name):
        self.name = name

    def handling(self, env, servers):

        global start_up_list, job_cnt

        # added a check to see if a resource pool needs another server.
        job_cnt += 1

        start_evt = env.process(check_servers(env, servers.computer))
        start_up_list.append(start_evt)
        print('{0:.5f}'.format(env.now), self.name, "requesting a server--")

        with servers.computer.get(1) as req:

            yield req
            # if the queue is empty then the req is never filled and the next lines are never called
            # need to do this before the rescource requests
            #
            # yield env.timeout(switch_on())    #switch on time
            # yield servers.server.put(1)
            print('{0:.5f}'.format(env.now), self.name, "occupies a server--")
            yield env.timeout(generate_service())  # service time
            print('{0:.5f}'.format(env.now), self.name, "leaves")

            # containers do not return a resouce at the end of a "with"
            # added a put
            # yield servers.computer.put(1)
            job_cnt -= 1
            yield env.process(return_server(env, servers.computer))


np.random.seed(0)
env = simpy.Environment()
servers = Room(env)
env.process(Generate_Job.arriving_job(env, servers))
env.run(until=UNTIL)
