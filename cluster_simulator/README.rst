==========
Quickstart
==========
It is a part of the Recommandation System called **Execution Simulator**. The big picture of the Recommandation System is shown below.

.. image:: ../notebooks/recommendation_system_diagram.png



A package that simulates  I/O application behavior in multi-tiered HPC system.

Features
--------

- Encodes a simple representation of an application as a sequence of I/O dominant and compute dominant phases.
- Storage tiers have their own performances and capacities that are shared resources between phases and applications.
- Support ephemeral tiers that lasts only during the application runtime.

Examples
========

Simple sequential application
-----------------------------

.. code-block:: python
    :caption: initial imports

    import simpy
    import time
    from cluster_simulator.cluster import Cluster, Tier, bandwidth_share_model, compute_share_model, get_tier, convert_size
    from cluster_simulator.phase import DelayPhase, ComputePhase, IOPhase
    from cluster_simulator.application import Application
    from cluster_simulator.analytics import display_run

.. code-block:: python
    :caption: application formalism

    # preparing execution environment variables
    env = simpy.Environment()
    data = simpy.Store(env)
    # app1 : read 3GB -> compute 15 seconds duration for 1 -> write 7GB
    app1 = Application( env, name="app1", # name of the app in the display
                        compute=[0, 15],  # two events, first at 0 and second at 15, and compute between them
                        read=[3e9, 0],    # read 3GB at 0, before compute phase, at the end do nothing (0)
                        write=[0, 5e9],  # write 5GB at first event, and 10GB at the second, after compute phase
                        data=data)        # collected data for monitoring

.. code-block:: python
    :caption: preparing the cluster compute and storage tiers facilities

    ssd_bandwidth =   {'read':  {'seq': 210, 'rand': 190}, # throughput for read ops in MB/s
                       'write': {'seq': 100, 'rand': 100}} # for read/write random/sequential I/O

    # register the tier with a name and a capacity
    ssd_tier = Tier(env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
    hdd_bandwidth = {'read':  {'seq': 80, 'rand': 80},
                     'write': {'seq': 40, 'rand': 40}}

    # register the tier with a name and a capacity
    hdd_tier = Tier(self.env, 'HDD', bandwidth=hdd_bandwidth, capacity=1e12)

    # register the cluster by completing the compute characteristics
    cluster = Cluster(env, compute_nodes=3,     # number of physical nodes
                           cores_per_node=2,    # available cores per node
                           tiers=[hdd_tier, ssd_tier]) # associate storage tiers to the cluster

.. code-block:: python
    :caption: running the application and get traces

    # placement list indicated where each I/O run in the tiers hierarchy
    env.process(app1.run(cluster, placement=[0, 1])) # run I/O n°1 on first tier (HDD), the second on SSD
    env.run()

We get the following (interactive) timeseries plot.
The application lasts 102.5 seconds. The first read I/O conveys 5GB of data from the HDD tier at a rate of 80MB/s. Its duration is 37.5 seconds. Then it is followed by a compute dominant phase of 15 seconds. Finally the write I/O phase happens on the SSD tier in 50 seconds at a 100MB/s rate.

.. raw:: html
    :file: docs/figure.html

Two parallel applications
-------------------------
Now we define two similar applications that will run on the same cluster and place their I/O operations on the same SSD tier.

.. code-block:: python
    :caption: running concurrent applications on the same cluster

    app1 = Application(env, name="app1", read=[4e9, 0], compute=[0, 10],  write=[0, 10e9],
                   data=data)
    app2 = Application(env, name="app2", read=[7e9, 0], compute=[0, 15],  write=[0, 3e9],
                   data=data)

    env.process(app1.run(cluster, placement=[1, 1])) # both I/O are placed in SSD
    env.process(app2.run(cluster, placement=[1, 1])) # app2 I/O are also in SSD
    env.run()

The two apps share equally an available bandwidth of 210MB/s for reading from SSD. Once app1 finishes the I/O reading at t = 38.09 seconds, it frees the bandwidth for the first reading I/O of the application 2. Hence the throughput reaches 210MB/s between 38.09 and 48.09 seconds. After this interval, the writing I/O of app1 starts while the first reading I/O still not finished, so they will share again available bandwidth.

.. raw:: html
    :file: docs/figure2.html

* TODO

