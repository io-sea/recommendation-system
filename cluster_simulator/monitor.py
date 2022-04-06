import simpy


class MonitorResource(simpy.Container):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []

    def get(self, *args, **kwargs):
        self.data.append((self._env.now, args[0]))
        return super().get(*args, **kwargs)

    def put(self, *args, **kwargs):
        self.data.append((self._env.now, args[0]))
        return super().put(*args, **kwargs)
