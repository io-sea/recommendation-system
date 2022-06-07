from cluster_simulator.cluster import Cluster, Tier, bandwidth_share_model, compute_share_model, get_tier, convert_size
from cluster_simulator.phase import DelayPhase, ComputePhase, IOPhase
from cluster_simulator.application import Application
import simpy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from itertools import groupby
from operator import itemgetter
from loguru import logger
import itertools


def accumulate_intervals(x_phase, y_phase):
    points = sorted(list(set(sorted(list(itertools.chain.from_iterable(x_phase))))))

    y = []
    for point in points:
        # init value for point
        v = 0
        for index, interval in enumerate(x_phase):
            if point >= interval[0] and point < interval[1]:
                v += y_phase[index]
        y.append(v)
    return points, y


def display_apps(data, width=800, height=600):

    DEFAULT_COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                      'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                      'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                      'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                      'rgb(188, 189, 34)', 'rgb(23, 190, 207)']
    # sort by app key
    items = sorted(data.items, key=itemgetter('app'))
    # list of apps
    apps = list(set(list(data['app'] for data in data.items)))

    fig = make_subplots(rows=len(apps), cols=1, shared_xaxes=True,
                        vertical_spacing=0.2, subplot_titles=apps)
    # iterate on apps
    i = 0
    for app, app_elements in groupby(items,  key=itemgetter('app')):
        # print(f"----------{app}----------")
        # app_color = DEFAULT_COLORS[i]
        offset = 0
        x_app = []
        y_app = []
        text = []

        for phase in app_elements:
            x_app.append(phase["t_start"])
            x_app.append(phase["t_end"])
            y_app.append(phase["bandwidth"])
            y_app.append(phase["bandwidth"])
            placement = "|to/from:"+phase["data_placement"]["placement"] if phase["data_placement"] else ''

            text.append(phase["type"].upper() + placement + "|volume="+convert_size(phase["volume"]))
            text.append(phase["type"].upper() + placement + "|volume="+convert_size(phase["volume"]))

        fig.append_trace(go.Scatter(x=np.array(x_app), y=np.array(y_app),
                                    text=text,
                                    textposition="top center",
                                    name=app, line_shape='hv', showlegend=False), row=i+1, col=1)
        i += 1
        fig.update_xaxes(title_text="time in s")
        fig.update_yaxes(title_text="bandwidth in MB/s")
        fig.update_layout(width=width, height=height, title_text=f"Stacked Volume Time Series for apps:{apps}")
    return fig


def accumulate_intervals(x_phase, y_phase):
    points = sorted(list(set(sorted(list(itertools.chain.from_iterable(x_phase))))))

    y = []
    for point in points:
        # init value for point
        v = 0
        for index, interval in enumerate(x_phase):
            # if intereset point is in interval
            if point >= interval[0] and point < interval[1]:
                # cumulate value from interval having intersections
                v += y_phase[index]
        y.append(v)
    return points, y


def display_cluster(data, cluster, width=800, height=600):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1)
    # iterate on phases
    i = 0
    x_phase = []
    y_phase = []
    x_tiers = []
    capacities = dict()
    text = []
    cpu_phase = []
    storage = dict()
    buffer_storage = dict()
    for phase in data.items:
        # sample_item = {'app': 'B8', 'type': 'read', 'cpu_usage': 1, 't_start': 0, 't_end': 4.761904761904762, 'bandwidth': 210.0, 'phase_duration': 4.761904761904762, 'volume': 1000000000.0, 'tiers': ['SSD', 'NVRAM'], 'data_placement': {'placement': 'SSD'}, 'tier_level': {'SSD': 1000000000.0, 'NVRAM': 0}}

        x_phase.append([phase["t_start"], phase["t_end"]])
        y_phase.append(phase["cpu_usage"])
        x_tiers.append(phase["t_start"])
        x_tiers.append(phase["t_end"])

        tier_indication = "|" + phase["data_placement"]["placement"] if phase["data_placement"] else ''

        text.append(phase["app"] + "|" + phase["type"].capitalize() + tier_indication + " | +volume="+convert_size(phase["volume"]))
        text.append(phase["app"] + "|" + phase["type"].capitalize() + tier_indication + " | +volume="+convert_size(phase["volume"]))

        for tier in phase["tiers"]:
            storage.setdefault(tier, []).append(phase["tier_level"][tier])
            storage.setdefault(tier, []).append(phase["tier_level"][tier])

    # trace CPU usage
    points, values = accumulate_intervals(x_phase, y_phase)
    fig.append_trace(go.Scatter(x=np.array(points), y=np.array(values),
                                line_shape='hv', showlegend=False), row=1, col=1)
    fig['layout']['yaxis']['title'] = 'CPU usage'

    for tier in storage.keys():
        i += 1
        # retrieve capacities
        tier_capacity = [cluster.tiers[j].capacity.capacity for j, ctier in enumerate(cluster.tiers) if ctier.name == tier][0]

        fig.append_trace(go.Scatter(x=np.array(x_tiers),
                                    y=100*np.array(storage[tier])/tier_capacity,
                                    text=text,
                                    textposition="top center",
                                    name=tier, line_shape='linear', showlegend=False), row=i+1, col=1)
        fig['layout']['yaxis' + str(i+1)]['title'] = tier + ' usage in %'

        # fig.append_trace(go.Scatter(x=np.array([x_tiers[0], x_tiers[-1]]),
        #                             y=np.array([cluster_tier.capacity.capacity]*2),
        #                             text=["Maximum Tier Capacity=" + str(cluster_tier.capacity.capacity)]*2, line_shape='hv', showlegend=False, line=dict(color='red', width=3, dash='dot')), row=i+1, col=1)

    # for tier in storage.keys():
    #     i += 1
    #     for index, abscissa in enumerate(x_phase):
    #         fig.append_trace(go.Scatter(x=np.array(x_phase[index]),
    #                                     y=np.array(storage[tier][index]),
    #                                     line_shape='linear', showlegend=False),
    #                                     row=1+i, col=1)

    # fig.append_trace(go.Scatter(x=np.array(points), y=np.array([cluster.compute_cores]),
    #                              line_shape='hv', showlegend=False), row=1, col=1)
    fig.append_trace(go.Scatter(x=np.array([points[0], points[-1]]), y=np.array([cluster.compute_cores.capacity]*2), text=["Maximum available cores in cluster=" +
                                                                                                                           str(cluster.compute_cores.capacity)]*2, line_shape='hv', showlegend=False, line=dict(color='red', width=3, dash='dot')), row=1, col=1)

    # fig.update_xaxes(title_text="time in s")
    # fig.update_yaxes(title_text="Units of used cores")

    fig.update_layout(height=height, width=width, title_text="State of the Cluster")
    return fig


def display_run(data, cluster, width=800, height=600):

    apps = list(set(list(data['app'] for data in data.items)))
    # sort by app key
    items = sorted(data.items, key=itemgetter('app'))
    # list of apps
    list_apps = [app for app, _ in groupby(items,  key=itemgetter('app'))]
    # list Burst Buffers
    bb_tier = [cluster.ephemeral_tier.name + " ("+convert_size(cluster.ephemeral_tier.capacity.capacity)+")"] if cluster.ephemeral_tier else []

    subplots_list = list_apps + bb_tier + [tier.name + " ("+convert_size(tier.capacity.capacity)+")" for tier in cluster.tiers] + ["CPU Cores"]

    fig = make_subplots(rows=len(subplots_list), cols=1, shared_xaxes=True,
                        # vertical_spacing=0.2,
                        subplot_titles=subplots_list)

    # iterate on apps to plot their dataflows
    i = 1
    for app, app_elements in groupby(items,  key=itemgetter('app')):
        # print(f"----------{app}----------")
        # app_color = DEFAULT_COLORS[i]
        offset = 0

        x_app = []
        y_app = []
        text = []

        for phase in app_elements:
            x_app.append(phase["t_start"])
            x_app.append(phase["t_end"])
            y_app.append(phase["bandwidth"])
            y_app.append(phase["bandwidth"])
            placement = "|to/from:"+phase["data_placement"]["placement"] if phase["data_placement"] else ''

            text.append(phase["type"].upper() + placement + "|volume="+convert_size(phase["volume"]))
            text.append(phase["type"].upper() + placement + "|volume="+convert_size(phase["volume"]))

        fig.append_trace(go.Scatter(x=np.array(x_app), y=np.array(y_app),
                                    text=text,
                                    textposition="top center",
                                    name=app, line_shape='hvh', showlegend=False), row=i, col=1)
        fig['layout']['yaxis'+str(i)]['title'] = 'dataflow in MB/s'
        fig.update_xaxes(title_text="time in s")
        i += 1

    x_phase = []
    y_phase = []
    x_tiers = [0]
    capacities = dict()
    text = []
    cpu_phase = []
    buffer_storage = dict()
    storage = dict()
    for phase in data.items:

        x_phase.append([phase["t_start"], phase["t_end"]])
        y_phase.append(phase["cpu_usage"])
        x_tiers.append(phase["t_end"])

        tier_indication = "|" + phase["data_placement"]["placement"] if phase["data_placement"] else ''

        text.append(phase["app"] + "|" + phase["type"].capitalize() + tier_indication + " | +volume="+convert_size(phase["volume"]))

        # feeding tiers level
        for tier in phase["tiers"]:
            storage.setdefault(tier, [0]).append(phase["tier_level"][tier])

        # feeding buffer level
        if cluster.ephemeral_tier:
            bb_tier = cluster.ephemeral_tier.name
            if bb_tier+"_level" in phase:
                buffer_storage.setdefault(bb_tier, []).append(phase[bb_tier+"_level"])

    # Burst Buffer level tracing
    for bb in buffer_storage.keys():
        bb_capacity = cluster.ephemeral_tier.capacity.capacity
        fig.append_trace(go.Scatter(x=np.array(x_tiers),
                                    y=100*np.array(buffer_storage[bb])/bb_capacity,
                                    text=text,
                                    textposition="top center",
                                    name=bb, line_shape='vh', showlegend=False), row=i, col=1)
        fig['layout']['yaxis' + str(i)]['title'] = bb + ' usage in %'
        i += 1

    for tier in storage.keys():
        # retrieve capacities
        tier_capacity = [cluster.tiers[j].capacity.capacity for j, ctier in enumerate(cluster.tiers) if ctier.name == tier][0]
        fig.append_trace(go.Scatter(x=np.array(x_tiers),
                                    y=100*np.array(storage[tier])/tier_capacity,
                                    text=text,
                                    textposition="top center",
                                    name=tier, line_shape='vh', showlegend=False), row=i, col=1)
        fig['layout']['yaxis' + str(i)]['title'] = tier + ' usage in %'

        i += 1

    # CPU tracing

    points, values = accumulate_intervals(x_phase, y_phase)
    fig.append_trace(go.Scatter(x=np.array(points), y=np.array(values),
                                line_shape='hv', showlegend=False), row=i, col=1)
    fig.append_trace(go.Scatter(x=np.array([points[0], points[-1]]), y=np.array([cluster.compute_cores.capacity]*2), text=["Maximum available cores in cluster=" +
                                                                                                                           str(cluster.compute_cores.capacity)]*2, line_shape='hv', showlegend=False,
                                line=dict(color='red', width=3, dash='dot')), row=i, col=1)
    fig['layout']['yaxis'+str(i)]['title'] = 'CPU usage'
    i += 1

    fig.update_layout(width=width, height=height, title_text="State of the Cluster")

    return fig

    """
    env = simpy.Environment()
data = simpy.Store(env)
nvram_bandwidth = {'read':  {'seq': 780, 'rand': 760},
                           'write': {'seq': 515, 'rand': 505}}
ssd_bandwidth = {'read':  {'seq': 210, 'rand': 190},
                    'write': {'seq': 100, 'rand': 100}}

ssd_tier = Tier(env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
nvram_tier = Tier(env, 'NVRAM', bandwidth=nvram_bandwidth, capacity=80e9)
cluster = Cluster(env, tiers=[ssd_tier, nvram_tier])
app1 = Application(env, name="app1", compute=[0, 10, 25], read=[1e9, 0, 2e9], write=[0, 5e9, 0], data=data)
app2 = Application(env, name="app2", compute=[0, 15], read=[0, 0], write=[0, 0], data=data)
env.process(app1.run(cluster, tiers=[0, 1, 0]))
env.process(app2.run(cluster, tiers=[0, 1]))
env.run()


# sort by app key
items = sorted(data.items, key=itemgetter('app'))
# list of apps
apps = list(set(list(data['app'] for data in data.items)))

fig = make_subplots(rows=len(apps), cols=1, shared_xaxes=True,
                    vertical_spacing=0.2, subplot_titles=apps)
# iterate on apps
i = 0
for app, app_elements in groupby(items,  key=itemgetter('app')):
    print(f"----------{app}----------")
    app_color = DEFAULT_COLORS[i]
    offset = 0


    x_app = []
    y_app = []
    text = []

    for phase in app_elements:
        x_app.append(phase["t_start"])
        x_app.append(phase["t_end"])
        y_app.append(phase["bandwidth"])
        y_app.append(phase["bandwidth"])
        placement = " | to/from:"+phase["data_placement"]["placement"] if phase["data_placement"] else ''

        text.append(phase["type"].capitalize() + " | "+ str(phase["cpu_usage"])+"cores" + placement + " | volume="+convert_size(phase["volume"]))
        text.append(phase["type"].capitalize() + " | "+ str(phase["cpu_usage"])+"cores" + placement + " | volume="+convert_size(phase["volume"]))


    fig.append_trace(go.Scatter(x=np.array(x_app), y=np.array(y_app),
                                text=text,
                                textposition="top center",
                                name=app, line_shape='hv', showlegend=False), row=i+1, col=1)
    i += 1
    fig.update_xaxes(title_text="time in s")
    fig.update_yaxes(title_text="bandwidth in MB/s")
    fig.update_layout(title_text=f"Stacked Volume Time Series for apps:{apps}")
fig.show()

# text=[phase["type"] + "from/to" + phase["data_placement"]["placement"] + "volume="+convert_size(phase["volume"])],



    """
