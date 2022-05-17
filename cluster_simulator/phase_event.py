from simpy.events import AnyOf, AllOf, Event
import simpy
from loguru import logger
from monitor import MonitorResource


class IO:
    def __init__(self, env, name, volume, bandwidth, delay=0):
        self.env = env
        self.name = name
        self.volume = volume
        self.bandwidth = bandwidth
        self.delay = delay
        self.b_usage = {}
        self.last_event = 0
        self.next_event = 0
        self.name = name
        self.volume = volume
        self.bandwidth = bandwidth
        self.delay = delay
        self.concurrent = False
        self.b_usage = dict()
        self.last_event = 0
        self.next_event = 0
        self.process = env.process(self.process_volume())

    def process_volume(self):
        while True:
            initial_bandiwdth = self.bandwidth.count
            processing_io = env.process(self.run())
            yield processing_io
            if self.bandwidth.count != initial_bandiwdth:
                if not processing_io.triggered:
                    processing_io.interrupt("INTERRUPTION HERE")

    def run(self):
        yield self.env.timeout(self.delay)
        # remaining volume of the IO to be conveyed
        volume = self.volume
        # retry IO until its volume is consumed
        while volume > 0:
            with self.bandwidth.request() as req:
                yield req
                self.b_usage[self.env.now] = round(100/self.bandwidth.count, 2)
                # try exhausting IO volume
                # update bandwidth usage
                #available_bandwidth = 1/(self.bandwidth.count+len(self.bandwidth.queue))
                available_bandwidth = 1/self.bandwidth.count

                self.next_event = self.env.peek()

                # take the smallest step, step_duration must be > 0
                if 0 < self.next_event - self.last_event < volume/available_bandwidth:
                    step_duration = self.next_event - self.last_event
                else:
                    step_duration = volume/available_bandwidth
                # print(f"{self.name}[at {env.now}](step_duration={step_duration}) | last_event = {self.last_event} | next_event = {self.next_event} | peek event = {self.env.peek()}")

                start = self.env.now
                try:
                    yield self.env.timeout(step_duration)
                    self.last_event += step_duration
                    volume -= step_duration * available_bandwidth
                    logger.info(f"[{self.name}](step) time "
                                f"= {start}-->{start+step_duration} | "
                                f"remaining volume = {volume} | "
                                f"available_bandwidth : {available_bandwidth} ")
                    self.b_usage[env.now] = round(100/self.bandwidth.count, 2)
                except simpy.Interrupt as interrupt:
                    logger.info(f"[{self.name}][Interruption] at {self.env.now} with {interrupt.cause} since = {interrupt.cause.usage_since}")
                    self.b_usage[env.now] = round(100/self.bandwidth.count, 2)
                    # if interrupt is a deadline interrupt, then we need to
                    # consume the remaining volume : update bandwidth and exhaust volume


if __name__ == '__main__':
    env = simpy.Environment()
    env.durations = []
    bandwidth = MonitorResource(env, capacity=10)
    IOs = [IO(env, name=str(i+1), volume=2-i,
              bandwidth=bandwidth, delay=i*0) for i in range(2)]

    env.run(until=10)
    for io in IOs:
        print(f"app: {io.name} | bandwidth usage: {io.b_usage}")
