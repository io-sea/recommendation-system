from cluster_simulator.cluster import Cluster, Tier, bandwidth_share_model, compute_share_model, get_tier, convert_size
from cluster_simulator.phase import DelayPhase, ComputePhase, IOPhase
from cluster_simulator.application import Application
import simpy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as colors
import numpy as np
from itertools import groupby
from operator import itemgetter
import itertools
from collections import deque
import matplotlib.pyplot as plt

DEFAULT_COLORS = [(31, 119, 180), (255, 127, 14),
                      (44, 160, 44), (214, 39, 40),
                      (148, 103, 189), (140, 86, 75),
                      (227, 119, 194), (127, 127, 127),
                      (188, 189, 34), (23, 190, 207)]

def accumulate_intervals(x_phase, y_phase):
    """Superimpose all points intervall within the boundary"""
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
        #app_color = DEFAULT_COLORS[i]
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

        fig.add_trace(go.Scatter(x=np.array(x_app), y=np.array(y_app),
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
        # | Monitoring| app: P6 | type: movement | cpu_usage: 1 | t_start: 281.0 | t_end: 287.5 | bandwidth_concurrency: 1 | bandwidth: 40.0 MB/s | phase_duration: 6.5 | volume: 260.0 MB | tiers: ['HDD'] | data_placement: {'placement': 'HDD', 'source': 'BB'} | init_level: {'HDD': 11740000000.0, 'BB': 8490000000.0} | tier_level: {'HDD': 12000000000.0, 'BB': 8490000000.0} | BB_level: 8490000000.0

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
    fig.add_trace(go.Scatter(x=np.array(points), y=np.array(values),
                                line_shape='hv', showlegend=False), row=1, col=1)
    fig['layout']['yaxis']['title'] = 'CPU usage'

    for tier in storage.keys():
        i += 1
        # retrieve capacities
        tier_capacity = [cluster.tiers[j].capacity.capacity for j, ctier in enumerate(cluster.tiers) if ctier.name == tier][0]
        print(np.array(x_tiers))
        print(np.array(storage[tier])/tier_capacity)
        fig.add_trace(go.Scatter(x=np.array(x_tiers),
                                    y=100*np.array(storage[tier])/tier_capacity,
                                    text=text,
                                    textposition="top center",
                                    name=tier, line_shape='linear', showlegend=False), row=i+1, col=1)
        fig['layout']['yaxis' + str(i+1)]['title'] = tier + ' usage in %'

        # fig.add_trace(go.Scatter(x=np.array([x_tiers[0], x_tiers[-1]]),
        #                             y=np.array([cluster_tier.capacity.capacity]*2),
        #                             text=["Maximum Tier Capacity=" + str(cluster_tier.capacity.capacity)]*2, line_shape='hv', showlegend=False, line=dict(color='red', width=3, dash='dot')), row=i+1, col=1)

    # for tier in storage.keys():
    #     i += 1
    #     for index, abscissa in enumerate(x_phase):
    #         fig.add_trace(go.Scatter(x=np.array(x_phase[index]),
    #                                     y=np.array(storage[tier][index]),
    #                                     line_shape='linear', showlegend=False),
    #                                     row=1+i, col=1)

    # fig.add_trace(go.Scatter(x=np.array(points), y=np.array([cluster.compute_cores]),
    #                              line_shape='hv', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.array([points[0], points[-1]]), y=np.array([cluster.compute_cores.capacity]*2), text=["Maximum available cores in cluster=" +
                                                                                                                           str(cluster.compute_cores.capacity)]*2, line_shape='hv', showlegend=False, line=dict(color='red', width=3, dash='dot')), row=1, col=1)

    # fig.update_xaxes(title_text="time in s")
    # fig.update_yaxes(title_text="Units of used cores")

    fig.update_layout(height=height, width=width, title_text="State of the Cluster")
    return fig


def display_run(data, cluster, width=800, height=600, nbr_points=None):
    # | Monitoring| app: P6 | type: movement | cpu_usage: 1 | t_start: 281.0 | t_end: 287.5 | bandwidth_concurrency: 1 | bandwidth: 40.0 MB/s | phase_duration: 6.5 | volume: 260.0 MB | tiers: ['HDD'] | data_placement: {'placement': 'HDD', 'source': 'BB'} | init_level: {'HDD': 11740000000.0, 'BB': 8490000000.0} | tier_level: {'HDD': 12000000000.0, 'BB': 8490000000.0} | BB_level: 8490000000.0
    # | Monitoring| app: N6 | type: eviction | cpu_usage: 1 | t_start: 93.5 | t_end: 93.5 | bandwidth_concurrency: 3 | bandwidth: inf MB/s | phase_duration: 0 | volume: 740.0 MB | tiers: ['HDD'] | data_placement: {'placement': 'BB'} | init_level: {'HDD': 2740000000.0, 'BB': 10000000000.0} | tier_level: {'HDD': 2740000000.0, 'BB': 9260000000.0} | BB_level: 9260000000.0

    jet_colors = colors.PLOTLY_SCALES['Viridis']
    apps = list(set(list(data['app'] for data in data.items)))
    # sort by app key
    items = sorted(data.items, key=itemgetter('app'))
    # list of apps
    list_apps = ["app#"+app for app, _ in groupby(items,  key=itemgetter('app'))]
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

        app_time = []
        phase_bw_read = []
        phase_bw_write = []
        app_bw_read = []
        app_bw_write = []
        t_starts = []
        t_ends = []

        text = []

        x_mvt = []
        y_mvt = []
        text_mvt = []

        for phase in app_elements:
            if phase['type'] not in ['movement', 'eviction']:
                t_starts.append(phase["t_start"])
                t_ends.append(phase["t_end"])

                if phase['type'] == 'compute':
                    # remove intermediate 0 padding if not final
                    app_bw_read.append(0)
                    app_bw_write.append(0)

                elif phase["type"] == "read":
                    app_bw_read.append(phase["bandwidth"])
                    app_bw_write.append(0)

                elif phase["type"]  == "write":
                    app_bw_read.append(0)
                    app_bw_write.append(phase["bandwidth"])

                else:
                    app_bw_read.append(0)
                    app_bw_write.append(0)


                placement = "|on:"+phase["data_placement"]["placement"] if phase["data_placement"] else ''

                text.append(phase["type"].upper() + placement + "|volume="+convert_size(phase["volume"]))
                text.append(phase["type"].upper() + placement + "|volume="+convert_size(phase["volume"]))

            if phase["type"] in ['movement']:
                x_mvt.append(phase["t_start"])
                x_mvt.append(phase["t_end"])
                y_mvt.append(phase["bandwidth"])
                y_mvt.append(phase["bandwidth"])
                placement = "|in:"+phase["data_placement"]["placement"] if phase["data_placement"] else ''
                source = ""
                if "data_placement" in phase and "source" in phase["data_placement"]:
                    source = "|from:"+phase["data_placement"]["source"]

                text_mvt.append(phase["type"].upper() + placement + source + "|volume="+convert_size(phase["volume"]))
                text_mvt.append(phase["type"].upper() + placement + "|volume="+convert_size(phase["volume"]))

        # do the interpolation
        if not nbr_points:
            N = 1 + int(max(t_ends) - min(t_starts))
        else:
            N = nbr_points
        app_time = np.linspace(min(t_starts), max(t_ends), N).tolist()
        app_bw_read = interpolate_signal_from_simulation(app_time, t_starts, t_ends, app_bw_read)
        app_bw_write = interpolate_signal_from_simulation(app_time, t_starts, t_ends, app_bw_write)
        # plot the app phases
        fig.add_trace(go.Scatter(x=np.array(app_time), y=np.array(app_bw_read),
                                    text=text,
                                    textposition="top center",
                                    name="app#"+app+"#read", line_shape='linear'),
                                    #line=dict(color=jet_colors[i][1])),
                                    row=i, col=1)
        fig.add_trace(go.Scatter(x=np.array(app_time), y=np.array(app_bw_write),
                                    text=text,
                                    textposition="top center",
                                    name="app#"+app+"#write", line_shape='linear'),
                                    #line=dict(color=jet_colors[i][1])),
                                    row=i, col=1)
        fig.add_trace(go.Scatter(x=np.array(x_mvt), y=np.array(y_mvt),
                                    text=text_mvt,
                                    textposition="top center",
                                    name=app + " mvt", #line_shape='linear',
                                    line={'shape':'linear','dash': 'dot'}), row=i, col=1)
        fig['layout']['yaxis'+str(i)]['title'] = 'dataflow in MB/s'
        fig.update_xaxes(title_text="time in s")
        i += 1

    x_phase = []
    y_phase = []
    x_tiers = [0]
    x_evict = []
    capacities = dict()
    text = []
    text_evict = []
    cpu_phase = []
    buffer_storage = dict()
    storage = dict()
    bb_levels = dict()
    storage_levels = dict()
    eviction_levels = dict()
    for phase in data.items:

        x_phase.append([phase["t_start"], phase["t_end"]])
        y_phase.append(phase["cpu_usage"])
        x_tiers.append(phase["t_start"])
        x_tiers.append(phase["t_end"])

        # feeding eviction dict
        if phase["type"] == "eviction":
            if "source" in phase["data_placement"]:
                tier = phase["data_placement"]["source"]
                eviction_levels.setdefault(tier, []).append([phase["init_level"][tier], phase["tier_level"][tier]])
                x_evict.append([phase["t_start"], phase["t_end"]])
                text_evict.append(phase["type"].upper() + "|IN:"+tier + "|volume="+convert_size(phase["init_level"][tier]-phase["tier_level"][tier]))
                text_evict.append(phase["type"].upper() + "|IN:"+tier + "|volume="+convert_size(phase["init_level"][tier]-phase["tier_level"][tier]))

        tier_indication = "|" + phase["data_placement"]["placement"] if phase["data_placement"] else ''

        text.append(phase["app"] + "|" + phase["type"].capitalize() + tier_indication + " | +volume="+convert_size(phase["volume"]))

        # feeding tiers level
        for tier in phase["tiers"]:
            if tier not in storage_levels.keys():
                storage_levels[tier] = dict()
            storage_levels[tier][phase["t_start"]] = phase["init_level"][tier]
            storage_levels[tier][phase["t_end"]] = phase["tier_level"][tier]
            storage.setdefault(tier, []).append(phase["init_level"][tier])
            storage.setdefault(tier, []).append(phase["tier_level"][tier])

        # feeding buffer level
        if cluster.ephemeral_tier:
            bb_tier = cluster.ephemeral_tier.name
            if bb_tier+"_level" in phase:
                bb_levels[phase["t_start"]] = phase["init_level"][bb_tier]
                bb_levels[phase["t_end"]] = phase["tier_level"][bb_tier]
                buffer_storage.setdefault(bb_tier, []).append([phase["init_level"][bb_tier], phase["tier_level"][bb_tier]])

    # Burst Buffer level tracing
    for bb in buffer_storage.keys():
        bb_capacity = cluster.ephemeral_tier.capacity.capacity
        # for segment, level in zip(x_phase, buffer_storage[bb]):
        #     fig.add_trace(go.Scatter(x=np.array(segment),
        #                                 y=level,
        #                                 text=text,
        #                                 textposition="top center",
        #                                 name=bb, line_shape='linear', showlegend=False), row=i, col=1)
        fig.add_trace(go.Scatter(x=np.array(list(bb_levels.keys())),
                                    y=100*bb_capacity*np.array(list(bb_levels.values()))/bb_capacity/100,
                                    text=text,
                                    textposition="top center",
                                    name=bb, line_shape='linear', showlegend=False), row=i, col=1)

        if bb in eviction_levels:
            for segment, level in zip(x_evict, eviction_levels[bb]):
                # fig.add_trace(go.Scatter(x=np.array(segment),
                #                             y=level,
                #                             text=text,
                #                             textposition="top center",
                #                             name=bb, line=dict(color='red', width=3, dash='dot'), showlegend=False), row=i, col=1)
                fig.add_trace(go.Scatter(x=np.array(segment), y=np.array(level),
                                            text=text_evict,
                                            textposition="top center",
                                            name=" Eviction",
                                            line=dict(shape='linear', color="black"),
                                            showlegend=False), row=i, col=1)
                # fig.add_shape(type="line", x0=segment[0], y0=level[0], x1=segment[1], y1=level[1],
                #               line=dict(width=3, color="black"),
                #               row=i, col=1)
                # fig.add_annotation(ax=segment[0],
                #                    ay=level[0],
                #                    x=segment[1],
                #                    y=level[1],
                #                    xref='x',

                #                    yref='y',
                #                    axref='x',
                #                    ayref='y',
                #                    text='',  # if you want only the arrow
                #                    showarrow=True,
                #                    arrowhead=3,
                #                    arrowsize=1,
                #                    arrowwidth=1,
                #                    arrowcolor='black', row=i, col=1)

        # fig.add_trace(go.Scatter(x=np.array(list(bb_levels.keys())),
        #                             y=100*bb_capacity*np.array(list(bb_levels.values()))/bb_capacity/100,
        #                             text=text,
        #                             textposition="top center",
        #                             name=bb, line_shape='linear', showlegend=False), row=i, col=1)
        # fig.add_trace(go.Scatter(x=np.array(x_tiers),
        #                             y=100*np.array(buffer_storage[bb])/bb_capacity,
        #                             text=text,
        #                             textposition="top center",
        #                             name=bb, line_shape='linear', showlegend=False), row=i, col=1)
        fig['layout']['yaxis' + str(i)]['title'] = bb + ' usage in Bytes'
        i += 1

    for tier in storage.keys():
        # retrieve capacities
        tier_capacity = [cluster.tiers[j].capacity.capacity for j, ctier in enumerate(cluster.tiers) if ctier.name == tier][0]
        fig.add_trace(go.Scatter(x=np.array(list(storage_levels[tier].keys())),
                                    y=100*tier_capacity*np.array(list(storage_levels[tier].values()))/tier_capacity/100,
                                    text=text,
                                    textposition="top center",
                                    name=tier, line_shape='linear', showlegend=False), row=i, col=1)
        # fig.add_trace(go.Scatter(x=np.array(x_tiers),
        #                             y=100*np.array(storage[tier])/tier_capacity,
        #                             text=text,
        #                             textposition="top center",
        #                             name=tier, line_shape='linear', showlegend=False), row=i, col=1)
        fig['layout']['yaxis' + str(i)]['title'] = tier + ' usage in Bytes'

        i += 1

    # CPU tracing
    points, values = accumulate_intervals(x_phase, y_phase)
    fig.add_trace(go.Scatter(x=np.array(points), y=np.array(values),
                                line_shape='hv', showlegend=False), row=i, col=1)

    fig.add_trace(go.Scatter(x=np.array([points[0], points[-1]]), y=np.array([cluster.compute_cores.capacity]*2), text=["Maximum available cores in cluster=" + str(cluster.compute_cores.capacity)]*2, line_shape='hv', showlegend=False,
                                line=dict(color='red', width=3, dash='dot')), row=i, col=1)
    fig['layout']['yaxis'+str(i)]['title'] = 'CPU usage'
    i += 1

    fig.update_layout(width=width, height=height, title_text="State of the Cluster")



    return fig


def get_execution_signal(data):
    """Extract from data structure generated by an App execution the subsequent apps' signal in order to be able to get traces independantely."""

    # sample_item = {'app': 'B8', 'type': 'read', 'cpu_usage': 1, 't_start': 0, 't_end': 4.761904761904762, 'bandwidth': 210.0, 'phase_duration': 4.761904761904762, 'volume': 1000000000.0, 'tiers': ['SSD', 'NVRAM'], 'data_placement': {'placement': 'SSD'}, 'tier_level': {'SSD': 1000000000.0, 'NVRAM': 0}}


    # # Get list of present apps in data content
    items = sorted(data.items, key=itemgetter('app'))
    list_apps = [app for app, _ in groupby(items,  key=itemgetter('app'))]

    output_dict = dict()
    # iterate on apps
    for app, app_elements in groupby(items,  key=itemgetter('app')):
        app_time = [0]
        app_bw_read = [0]
        app_bw_write = [0]
        # iterate on phase (app elements/events)
        for phase in app_elements:
            if phase['type'] in ['compute']:
                app_time.append(phase["t_end"])
                app_bw_read.append(0)
                app_bw_write.append(0)
            if phase["type"] in ["read"]:
                app_time.append(phase["t_end"])
                app_bw_read.append(phase["bandwidth"])
                app_bw_write.append(0)
            if phase["type"] in ["write"]:
                app_time.append(phase["t_end"])
                app_bw_write.append(phase["bandwidth"])
                app_bw_read.append(0)

        output_dict[app] = {}
        output_dict[app]["time"] = app_time # if app_time[0]==0 else app_time.insert(0, 0)
        output_dict[app]["read_bw"] = app_bw_read
        output_dict[app]["write_bw"] = app_bw_write

    return output_dict


def get_execution_signal_2(data):
    """Extract from data structure generated by an App execution the subsequent apps' signal in order to be able to get traces independantely."""

    # sample_item = {'app': 'B8', 'type': 'read', 'cpu_usage': 1, 't_start': 0, 't_end': 4.761904761904762, 'bandwidth': 210.0, 'phase_duration': 4.761904761904762, 'volume': 1000000000.0, 'tiers': ['SSD', 'NVRAM'], 'data_placement': {'placement': 'SSD'}, 'tier_level': {'SSD': 1000000000.0, 'NVRAM': 0}}


    # # Get list of present apps in data content
    items = sorted(data.items, key=itemgetter('app'))
    list_apps = [app for app, _ in groupby(items,  key=itemgetter('app'))]


    output_dict = dict()
    # iterate on apps
    for app, app_elements in groupby(items,  key=itemgetter('app')):

        app_time = []
        app_bw_read = []
        app_bw_write = []
        compute_excess = 0
        phase_sequence = deque(["", "", ""])
        # iterate on phase (app elements/events)
        for phase in app_elements:
            phase_timestamps = list(range(int(phase["t_start"] - compute_excess), int(phase["t_end"] - compute_excess)))

            if phase['type'] in ['compute']:
                # remove intermediate 0 padding if not final
                phase_sequence.append('co')
                phase_sequence.popleft()
                phase_bw = [0]*len(phase_timestamps)
                app_time += phase_timestamps
                app_bw_read += phase_bw
                app_bw_write += phase_bw

            if phase["type"] in ["read", "write"]:
                phase_sequence.append('io')
                phase_sequence.popleft()
                if list(phase_sequence) == ['io', 'co', 'io']:
                    compute_excess += 1
                    phase_timestamps = list(range(int(phase["t_start"] - compute_excess), int(phase["t_end"] - compute_excess)))
                    app_time.pop()
                    app_bw_read.pop()
                    app_bw_write.pop()

            if phase["type"] in ["read"]:
                app_time += phase_timestamps
                app_bw_read += [phase["bandwidth"]]*len(phase_timestamps)
                app_bw_write += [0]*len(phase_timestamps)

            if phase["type"] in ["write"]:
                app_time += phase_timestamps
                app_bw_read += [0]*len(phase_timestamps)
                app_bw_write += [phase["bandwidth"]]*len(phase_timestamps)
        output_dict[app] = {}
        output_dict[app]["time"] = app_time # if app_time[0]==0 else app_time.insert(0, 0)
        output_dict[app]["read_bw"] = app_bw_read
        output_dict[app]["write_bw"] = app_bw_write

    return output_dict

def plot_simple_signal(x, read_signal, write_signal):
    plt.plot(x, read_signal, label="read signal")
    plt.plot(x, write_signal, label="write signal")
    plt.grid(True)
    plt.legend()
    plt.title(f"timeseries for job signals")
    plt.show()

def interpolate_signal_from_simulation(x, start_points, end_points, bw_values):
    """Transfroms a set of interval start points and end points and multidimensional values into n-dimensional discrete signal by taking relevant value inside intervals.
    If intervals are redundant, only max value is retained.

    Args:
        x (np.array): array with values for the n-dimensional signal.
        start_points (list): list of starting points for the intervals
        end_points (list): list of ending points for the intervals
        bw_values (list): list of 1-dim signal values between interval boundaries.

    Returns:
        result (np.array): array with interpolated values from bw_values within boundaries.

    Example of usage:
        t_starts = [0, 0, 20]
        t_ends = [20, 20, 30]

        read_bw = [105, 0, 0]
        write_bw = [0, 50, 100]

        result_read = fill_list(x, t_starts, t_ends, read_bw)
        result_write = fill_list(x, t_starts, t_ends, write_bw)

    """

    assert len(start_points) == len(end_points) == len(bw_values), f"start_points: {len(start_points)}, end_points: {len(end_points)} and bw_values: {len(bw_values)} must have same length"

    last_start = 0
    last_end = 0
    max_bw = 0
    result = [0] * len(x)
    for ix, point in enumerate(x):
        #print(f"{point} -> len(result) = {len(result)}")
        for i, (start, end) in enumerate(zip(start_points, end_points)):
            # print(f"scanning interval: {i}")
            # if interval has been seen before and point falls inside
            if last_start == start and last_end == end and start <= point < end:
                if bw_values[i] > max_bw:
                    max_bw = bw_values[i]
                    result.insert(ix, bw_values[i])
                    #print(f"Point {point}: {start}->{end}: interval number: {i} | value: {bw_values[i]}")
                    result.pop()
            else:
                max_bw = 0
                last_start = start
                last_end = end
                if start <= point < end and bw_values[i] > 0:
                    max_bw = bw_values[i]
                    result.insert(ix, bw_values[i])
                    #print(f"Point {point}: {start}->{end}: interval number: {i} | value: {bw_values[i]}")
                    result.pop()

    return result

def get_execution_signal_3(data, nbr_points=None):
    """Extract from data structure generated by an App execution the subsequent apps' signal in order to be able to get traces independantely."""

    # sample_item = {'app': 'B8', 'type': 'read', 'cpu_usage': 1, 't_start': 0, 't_end': 4.761904761904762, 'bandwidth': 210.0, 'phase_duration': 4.761904761904762, 'volume': 1000000000.0, 'tiers': ['SSD', 'NVRAM'], 'data_placement': {'placement': 'SSD'}, 'tier_level': {'SSD': 1000000000.0, 'NVRAM': 0}}


    # # Get list of present apps in data content
    items = sorted(data.items, key=itemgetter('app'))
    list_apps = [app for app, _ in groupby(items,  key=itemgetter('app'))]

    result = []
    # iterate on apps
    for app, app_elements in groupby(items,  key=itemgetter('app')):
        output_dict = {app: {}}
        app_time = []
        phase_bw_read = []
        phase_bw_write = []
        app_bw_read = []
        app_bw_write = []
        t_starts = []
        t_ends = []

        for phase in app_elements:
            t_starts.append(phase["t_start"])
            t_ends.append(phase["t_end"])

            if phase['type'] in ['compute']:
                # remove intermediate 0 padding if not final
                app_bw_read.append(0)
                app_bw_write.append(0)

            if phase["type"] in ["read"]:
                app_bw_read.append(phase["bandwidth"])
                app_bw_write.append(0)

            if phase["type"] in ["write"]:
                app_bw_read.append(0)
                app_bw_write.append(phase["bandwidth"])

        # do the interpolation
        if not nbr_points:
            N = 1 + int(max(t_ends) - min(t_starts))
        else:
            N = nbr_points
        app_time = np.linspace(min(t_starts), max(t_ends), N).tolist()
        app_bw_read = interpolate_signal_from_simulation(app_time, t_starts, t_ends, app_bw_read)
        app_bw_write = interpolate_signal_from_simulation(app_time, t_starts, t_ends, app_bw_write)

        output_dict[app]["time"] = app_time # if app_time[0]==0 else app_time.insert(0, 0)
        output_dict[app]["read_bw"] = app_bw_read
        output_dict[app]["write_bw"] = app_bw_write

    return output_dict

def display_run_with_signal(data, cluster, app_signal, width=1200, height=600):
    # | Monitoring| app: P6 | type: movement | cpu_usage: 1 | t_start: 281.0 | t_end: 287.5 | bandwidth_concurrency: 1 | bandwidth: 40.0 MB/s | phase_duration: 6.5 | volume: 260.0 MB | tiers: ['HDD'] | data_placement: {'placement': 'HDD', 'source': 'BB'} | init_level: {'HDD': 11740000000.0, 'BB': 8490000000.0} | tier_level: {'HDD': 12000000000.0, 'BB': 8490000000.0} | BB_level: 8490000000.0
    # | Monitoring| app: N6 | type: eviction | cpu_usage: 1 | t_start: 93.5 | t_end: 93.5 | bandwidth_concurrency: 3 | bandwidth: inf MB/s | phase_duration: 0 | volume: 740.0 MB | tiers: ['HDD'] | data_placement: {'placement': 'BB'} | init_level: {'HDD': 2740000000.0, 'BB': 10000000000.0} | tier_level: {'HDD': 2740000000.0, 'BB': 9260000000.0} | BB_level: 9260000000.0
    timestamps, read_signal, write_signal = app_signal

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

        x_mvt = []
        y_mvt = []
        text_mvt = []

        for phase in app_elements:
            if phase['type'] not in ['movement']:
                x_app.append(phase["t_start"])
                x_app.append(phase["t_end"])
                y_app.append(phase["bandwidth"])
                y_app.append(phase["bandwidth"])
                placement = "|to/from:"+phase["data_placement"]["placement"] if phase["data_placement"] else ''

                text.append(phase["type"].upper() + placement + "|volume="+convert_size(phase["volume"]))
                text.append(phase["type"].upper() + placement + "|volume="+convert_size(phase["volume"]))

            if phase["type"] in ['movement']:
                x_mvt.append(phase["t_start"])
                x_mvt.append(phase["t_end"])
                y_mvt.append(phase["bandwidth"])
                y_mvt.append(phase["bandwidth"])
                placement = "|in:"+phase["data_placement"]["placement"] if phase["data_placement"] else ''
                source = ""
                if "data_placement" in phase and "source" in phase["data_placement"]:
                    source = "|from:"+phase["data_placement"]["source"]

                text_mvt.append(phase["type"].upper() + placement + source + "|volume="+convert_size(phase["volume"]))
                text_mvt.append(phase["type"].upper() + placement + "|volume="+convert_size(phase["volume"]))

                # plot the app phases
        print(f"x_app = {x_app}")
        fig.add_trace(go.Scatter(x=np.array(x_app), y=np.array(y_app),
                                    text=text,
                                    textposition="top center", mode='lines+markers',
                                    name="App Modeled by Execution Sim "+app, line_shape='linear'), row=i, col=1)
        # plot original signals
        fig.add_trace(go.Scatter(x=timestamps, y=read_signal/5/1e6, line_dash='dash',
                                    name="read signal from original data", line_shape='linear',
                                    mode='lines+markers'), row=i, col=1)
        fig.add_trace(go.Scatter(x=timestamps, y=write_signal, line_dash='dash',
                                    name="write signal from original data", line_shape='linear',
                                    mode='lines+markers'), row=i, col=1)
        fig.add_trace(go.Scatter(x=np.array(x_mvt), y=np.array(y_mvt),
                                    text=text_mvt,
                                    textposition="top center",
                                    name=app + " mvt", line_shape='linear', line={'dash': 'dot'}), row=i, col=1)
        fig['layout']['yaxis'+str(i)]['title'] = 'dataflow in MB/s'
        fig.update_xaxes(title_text="time in s")
        i += 1
    x_phase = []
    y_phase = []
    x_tiers = [0]
    x_evict = []
    capacities = dict()
    text = []
    text_evict = []
    cpu_phase = []
    buffer_storage = dict()
    storage = dict()
    bb_levels = dict()
    storage_levels = dict()
    eviction_levels = dict()
    for phase in data.items:

        x_phase.append([phase["t_start"], phase["t_end"]])
        y_phase.append(phase["cpu_usage"])
        x_tiers.append(phase["t_start"])
        x_tiers.append(phase["t_end"])

        # feeding eviction dict
        if phase["type"] == "eviction":
            if "source" in phase["data_placement"]:
                tier = phase["data_placement"]["source"]
                eviction_levels.setdefault(tier, []).append([phase["init_level"][tier], phase["tier_level"][tier]])
                x_evict.append([phase["t_start"], phase["t_end"]])
                text_evict.append(phase["type"].upper() + "|IN:"+tier + "|volume="+convert_size(phase["init_level"][tier]-phase["tier_level"][tier]))
                text_evict.append(phase["type"].upper() + "|IN:"+tier + "|volume="+convert_size(phase["init_level"][tier]-phase["tier_level"][tier]))

        tier_indication = "|" + phase["data_placement"]["placement"] if phase["data_placement"] else ''

        text.append(phase["app"] + "|" + phase["type"].capitalize() + tier_indication + " | +volume="+convert_size(phase["volume"]))

        # feeding tiers level
        for tier in phase["tiers"]:
            if tier not in storage_levels.keys():
                storage_levels[tier] = dict()
            storage_levels[tier][phase["t_start"]] = phase["init_level"][tier]
            storage_levels[tier][phase["t_end"]] = phase["tier_level"][tier]
            storage.setdefault(tier, []).append(phase["init_level"][tier])
            storage.setdefault(tier, []).append(phase["tier_level"][tier])

        # feeding buffer level
        if cluster.ephemeral_tier:
            bb_tier = cluster.ephemeral_tier.name
            if bb_tier+"_level" in phase:
                bb_levels[phase["t_start"]] = phase["init_level"][bb_tier]
                bb_levels[phase["t_end"]] = phase["tier_level"][bb_tier]
                buffer_storage.setdefault(bb_tier, []).append([phase["init_level"][bb_tier], phase["tier_level"][bb_tier]])

    # Burst Buffer level tracing
    for bb in buffer_storage.keys():
        bb_capacity = cluster.ephemeral_tier.capacity.capacity
        # for segment, level in zip(x_phase, buffer_storage[bb]):
        #     fig.add_trace(go.Scatter(x=np.array(segment),
        #                                 y=level,
        #                                 text=text,
        #                                 textposition="top center",
        #                                 name=bb, line_shape='linear', showlegend=False), row=i, col=1)
        fig.add_trace(go.Scatter(x=np.array(list(bb_levels.keys())),
                                    y=100*bb_capacity*np.array(list(bb_levels.values()))/bb_capacity/100,
                                    text=text,
                                    textposition="top center",
                                    name=bb, line_shape='linear', showlegend=False), row=i, col=1)

        if bb in eviction_levels:
            for segment, level in zip(x_evict, eviction_levels[bb]):
                # fig.add_trace(go.Scatter(x=np.array(segment),
                #                             y=level,
                #                             text=text,
                #                             textposition="top center",
                #                             name=bb, line=dict(color='red', width=3, dash='dot'), showlegend=False), row=i, col=1)
                fig.add_trace(go.Scatter(x=np.array(segment), y=np.array(level),
                                            text=text_evict,
                                            textposition="top center",
                                            name=" Eviction",
                                            line=dict(shape='linear', color="black"),
                                            showlegend=False), row=i, col=1)
                # fig.add_shape(type="line", x0=segment[0], y0=level[0], x1=segment[1], y1=level[1],
                #               line=dict(width=3, color="black"),
                #               row=i, col=1)
                # fig.add_annotation(ax=segment[0],
                #                    ay=level[0],
                #                    x=segment[1],
                #                    y=level[1],
                #                    xref='x',

                #                    yref='y',
                #                    axref='x',
                #                    ayref='y',
                #                    text='',  # if you want only the arrow
                #                    showarrow=True,
                #                    arrowhead=3,
                #                    arrowsize=1,
                #                    arrowwidth=1,
                #                    arrowcolor='black', row=i, col=1)

        # fig.add_trace(go.Scatter(x=np.array(list(bb_levels.keys())),
        #                             y=100*bb_capacity*np.array(list(bb_levels.values()))/bb_capacity/100,
        #                             text=text,
        #                             textposition="top center",
        #                             name=bb, line_shape='linear', showlegend=False), row=i, col=1)
        # fig.add_trace(go.Scatter(x=np.array(x_tiers),
        #                             y=100*np.array(buffer_storage[bb])/bb_capacity,
        #                             text=text,
        #                             textposition="top center",
        #                             name=bb, line_shape='linear', showlegend=False), row=i, col=1)
        fig['layout']['yaxis' + str(i)]['title'] = bb + ' usage in Bytes'
        i += 1

    for tier in storage.keys():
        # retrieve capacities
        tier_capacity = [cluster.tiers[j].capacity.capacity for j, ctier in enumerate(cluster.tiers) if ctier.name == tier][0]
        fig.add_trace(go.Scatter(x=np.array(list(storage_levels[tier].keys())),
                                    y=100*tier_capacity*np.array(list(storage_levels[tier].values()))/tier_capacity/100,
                                    text=text,
                                    textposition="top center",
                                    name=tier, line_shape='linear', showlegend=False), row=i, col=1)
        # fig.add_trace(go.Scatter(x=np.array(x_tiers),
        #                             y=100*np.array(storage[tier])/tier_capacity,
        #                             text=text,
        #                             textposition="top center",
        #                             name=tier, line_shape='linear', showlegend=False), row=i, col=1)
        fig['layout']['yaxis' + str(i)]['title'] = tier + ' usage in Bytes'

        i += 1

    # CPU tracing
    points, values = accumulate_intervals(x_phase, y_phase)
    fig.add_trace(go.Scatter(x=np.array(points), y=np.array(values),
                                line_shape='hv', showlegend=False), row=i, col=1)

    fig.add_trace(go.Scatter(x=np.array([points[0], points[-1]]), y=np.array([cluster.compute_cores.capacity]*2), text=["Maximum available cores in cluster=" + str(cluster.compute_cores.capacity)]*2, line_shape='hv', showlegend=False,
                                line=dict(color='red', width=3, dash='dot')), row=i, col=1)
    fig['layout']['yaxis'+str(i)]['title'] = 'CPU usage'
    i += 1

    fig.update_layout(width=width, height=height, title_text="State of the Cluster")

    return fig



def display_original_sim_signals(simulated, originals, str_org="Original",
                                 str_sim="Simulated", width=1200, height=600):

    timestamps, original_read, original_write = originals
    time, read_bw, write_bw = simulated

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        # vertical_spacing=0.2,
                        subplot_titles=["On read/write instant bandwidths",
                                        "On read/write cumulated volumes"])
    # plot original signals
    fig.add_trace(go.Scatter(x=np.array(timestamps), y=np.array(original_read),
                                # text=text,
                                # legendgroup="Timeseries",
                                textposition="top center", mode='lines+markers',
                                marker_size = 5,
                                name=f"{str_org} I/O Read Volume Signal (IOI)", line_shape='linear',
                                line_color = "blue"), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.array(timestamps), y=np.array(original_write),
                                # text=text,
                                # legendgroup="Timeseries",
                                textposition="top center", mode='lines+markers',
                                marker_size = 5,
                                name=f"{str_org} I/O Written Volume Signal (IOI)", line_shape='linear',
                                line_color="green"), row=1, col=1)

    # plot simulated signals
    fig.add_trace(go.Scatter(x=np.array(time), y=np.array(read_bw),
                                # text=text,
                                # legendgroup="Timeseries",
                                textposition="top center", mode='lines',
                                name=f"{str_sim} I/O Read Volume Signal", line_shape='linear',
                                line={'dash': 'dot'},
                                line_color = "blue"), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.array(time), y=np.array(write_bw),
                                # text=text,
                                # legendgroup="Timeseries",
                                textposition="top center", mode='lines',
                                name=f"{str_sim} I/O Written Volume Signal", line_shape='linear',
                                line={'dash': 'dot'},
                                line_color="green"), row=1, col=1)

    # plot cumulated original signals
    fig.add_trace(go.Scatter(x=np.array(timestamps), y=np.cumsum(np.array(original_read)),
                                # text=text,
                                # legendgroup="Cumulated Timeseries",
                                textposition="top center", mode='lines+markers',
                                marker_size = 5,
                                name=f"Cumulated {str_org} I/O Read Volume Signal (IOI)", line_shape='linear',
                                line_color = "blue"), row=2, col=1)
    fig.add_trace(go.Scatter(x=np.array(timestamps), y=np.cumsum(np.array(original_write)),
                                # text=text,
                                # legendgroup="Cumulated Timeseries",
                                textposition="top center", mode='lines+markers',
                                marker_size = 5,
                                name=f"Cumulated {str_org} I/O Written Volume Signal (IOI)", line_shape='linear',
                                line_color="green"), row=2, col=1)

    # plot cumulated simulated signals
    fig.add_trace(go.Scatter(x=np.array(time), y=np.cumsum(np.array(read_bw)),
                                # text=text,
                                # legendgroup="Cumulated Timeseries",
                                textposition="top center", mode='lines',
                                name=f"{str_sim} I/O Read Cumulative Volume Signal", line_shape='linear',
                                line={'dash': 'dot'},
                                line_color = "blue"), row=2, col=1)
    fig.add_trace(go.Scatter(x=np.array(time), y=np.cumsum(np.array(write_bw)),
                                # text=text,
                                # legendgroup="Cumulated Timeseries",
                                textposition="top center", mode='lines',
                                name=f"{str_sim} I/O Written Cumulative Volume Signal", line_shape='linear',
                                line={'dash': 'dot'},
                                line_color="green"), row=2, col=1)

    fig['layout']['yaxis']['title'] = 'dataflow in MB/s'
    fig.update_xaxes(title_text="time (1pt = 5s)")


    fig.update_layout(width=width, height=height,
                      autosize=False,
                      legend=dict(
                          font=dict(size=8),
                          tracegroupgap = 2*180,
                          orientation="v"
                      ),
                      title_text="Comparing simulated and IOI signals")

    return fig


def compare_original_sim_signals(simulated, originals, str_org="Original",
                                 str_sim="Simulated", width=1200, height=600):

    timestamps, original_read, original_write = originals
    time, read_bw, write_bw = simulated

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        # vertical_spacing=0.2,
                        subplot_titles=["On read/write instant bandwidths",
                                        "On read/write cumulated volumes"])
    # plot original signals
    fig.add_trace(go.Scatter(x=np.array(timestamps), y=np.array(original_read),
                                # text=text,
                                # legendgroup="Timeseries",
                                textposition="top center", mode='lines+markers',
                                marker_size = 5,
                                name=f"{str_org} I/O Read Volume Signal (IOI)", line_shape='linear',
                                line_color = "blue"), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.array(timestamps), y=np.array(original_write),
                                # text=text,
                                # legendgroup="Timeseries",
                                textposition="top center", mode='lines+markers',
                                marker_size = 5,
                                name=f"{str_org} I/O Written Volume Signal (IOI)", line_shape='linear',
                                line_color="green"), row=1, col=1)

    # plot simulated signals
    fig.add_trace(go.Scatter(x=np.array(time), y=np.array(read_bw),
                                # text=text,
                                # legendgroup="Timeseries",
                                textposition="top center", mode='lines',
                                name=f"{str_sim} I/O Read Volume Signal", line_shape='linear',
                                line={'dash': 'dot'},
                                line_color = "blue"), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.array(time), y=np.array(write_bw),
                                # text=text,
                                # legendgroup="Timeseries",
                                textposition="top center", mode='lines',
                                name=f"{str_sim} I/O Written Volume Signal", line_shape='linear',
                                line={'dash': 'dot'},
                                line_color="green"), row=1, col=1)

    # plot cumulated original signals
    fig.add_trace(go.Scatter(x=np.array(timestamps), y=np.cumsum(np.array(original_read)),
                                # text=text,
                                # legendgroup="Cumulated Timeseries",
                                textposition="top center", mode='lines+markers',
                                marker_size = 5,
                                name=f"Cumulated {str_org} I/O Read Volume Signal (IOI)", line_shape='linear',
                                line_color = "blue"), row=2, col=1)
    fig.add_trace(go.Scatter(x=np.array(timestamps), y=np.cumsum(np.array(original_write)),
                                # text=text,
                                # legendgroup="Cumulated Timeseries",
                                textposition="top center", mode='lines+markers',
                                marker_size = 5,
                                name=f"Cumulated {str_org} I/O Written Volume Signal (IOI)", line_shape='linear',
                                line_color="green"), row=2, col=1)

    # plot cumulated simulated signals
    fig.add_trace(go.Scatter(x=np.array(time), y=np.cumsum(np.array(read_bw)),
                                # text=text,
                                # legendgroup="Cumulated Timeseries",
                                textposition="top center", mode='lines',
                                name=f"{str_sim} I/O Read Cumulative Volume Signal", line_shape='linear',
                                line={'dash': 'dot'},
                                line_color = "blue"), row=2, col=1)
    fig.add_trace(go.Scatter(x=np.array(time), y=np.cumsum(np.array(write_bw)),
                                # text=text,
                                # legendgroup="Cumulated Timeseries",
                                textposition="top center", mode='lines',
                                name=f"{str_sim} I/O Written Cumulative Volume Signal", line_shape='linear',
                                line={'dash': 'dot'},
                                line_color="green"), row=2, col=1)

    fig['layout']['yaxis']['title'] = 'dataflow in MB/s'
    fig.update_xaxes(title_text="time (1pt = 5s)")


    fig.update_layout(width=width, height=height,
                      autosize=False,
                      legend=dict(
                          font=dict(size=8),
                          tracegroupgap = 2*180,
                          orientation="v"
                      ),
                      title_text="Comparing simulated and IOI signals")

    return fig
