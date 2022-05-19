from simpy.events import AnyOf, AllOf, Event
import simpy
from loguru import logger


class MonitorResource(simpy.Resource):
    """Subclassing simpy Resource to introduce the ability to check_bandwidth when resource is requested or released."""

    def __init__(self, *args, **kwargs):
        """Init method using parent init method."""
        super().__init__(*args, **kwargs)
        self.env = args[0]

    def request(self, *args, **kwargs):
        """On request method, cehck_bandwidth using parent request method."""
        self.check_bandwidth()
        return super().request(*args, **kwargs)

    def release(self, *args, **kwargs):
        """On release method, cehck_bandwidth using parent release method."""
        self.check_bandwidth()
        return super().release(*args, **kwargs)

    def check_bandwidth(self):
        """Checks running IO when bandwidth occupation changes. IOs should be interrupted on release or request of a bandwidth slot.
        """
        for io_event in IO.current_ios:
            if not io_event.processed and io_event.triggered and io_event.is_alive:
                # capture the IOs not finished, but triggered and alive
                print(io_event.is_alive)
                print(io_event.value)
                io_event.interrupt('updating bandwidth')


class IO:
    """Class I/O that manages the IO properties and processing."""
    current_ios = []

    def __init__(self, env, name, volume, bandwidth, delay=0):
        self.env = env
        self.name = name
        self.volume = volume
        self.bandwidth = bandwidth  # a monitored resource
        self.delay = delay
        self.last_event = 0
        self.next_event = 0
        self.process = env.process(self.run())

    def process_volume(self, step_duration, volume):
        try:
            available_bandwidth = 1/self.bandwidth.count
            start = self.env.now
            yield env.timeout(step_duration)
            self.last_event += step_duration
            volume -= step_duration * available_bandwidth

            logger.info(f"[{self.name}](step, at {self.env.now}) | "
                        f"time : {start}-->{start+step_duration} | "
                        f"remaining volume = {volume} | "
                        f"available_bandwidth : {available_bandwidth} ")
        # except simpy.Interrupt as interrupt:
        except simpy.exceptions.Interrupt as interrupt:

            end_time = self.env.now
            step_duration = end_time - start
            volume -= step_duration * available_bandwidth
            logger.info(f"Interrupted at {self.env.now} after usage of {self.env.now - start}s | volume = {volume} | step_duration = {step_duration}")
            logger.info(f"[{self.name}](step, at {end_time}) | "
                        f"time : {start}-->{start + step_duration} | "
                        f"remaining volume = {volume} | "
                        f"available_bandwidth : {available_bandwidth} ")
        return volume

    def run(self):
        yield self.env.timeout(self.delay)
        # remaining volume of the IO to be conveyed
        volume = self.volume
        # retry IO until its volume is consumed
        while volume > 0:
            with self.bandwidth.request() as req:
                yield req

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
                # print(step_duration)
                # print(f"{self.name}[at {env.now}](step_duration={step_duration}) | last_event = {self.last_event} | next_event = {self.next_event} | peek event = {self.env.peek()}")
                io_event = self.env.process(self.process_volume(step_duration, volume))
                IO.current_ios.append(io_event)
                volume = yield io_event
                # if new_volume == volume:
                #     logger.info(f"new_volume = {new_volume} | volume = {volume}")
                # volume = new_volume


if __name__ == '__main__':
    env = simpy.Environment()
    env.durations = []
    bandwidth = MonitorResource(env, capacity=10)
    # IOs = [IO(env, name=str(i+1), volume=2-i,
    #           bandwidth=bandwidth, delay=i*0) for i in range(2)]
    IOs = [IO(env, name=str(i+1), volume=1+i,
              bandwidth=bandwidth, delay=i*0) for i in range(2)]

    env.run()
