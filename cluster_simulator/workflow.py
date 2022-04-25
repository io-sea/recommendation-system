import networkx as nx
from phase import DelayPhase, ComputePhase, IOPhase, name_app


compute = [0, 10]
read = [1e9, 0]
write = [0, 5e9]
delay = 5
delay_phase = DelayPhase(delay)

compute_phase = ComputePhase(compute[1]-compute[0], cores=3)

read_io = IOPhase(cores=1, operation="read", volume=1e9,
                  pattern=1)

write_io = IOPhase(cores=1, operation="write", volume=3e9,
                   pattern=1)

app_graph = nx.DiGraph()

app_graph.add_nodes_from([read_io, write_io])
app_graph.add_edge(read_io, write_io, object=compute_phase, weight=compute_phase.duration)
