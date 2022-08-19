==========
Quickstart
==========
It is a part of the Recommandation System called **App Decomposer**. The big picture of the Recommandation System is shown below.

.. image:: ../notebooks/recommendation_system_diagram.png



A package that detect for an instrumented app compute, read I/O and write I/O phases and encodes the sequential behavior as event-based numerical arrays.

Features
--------

- Transform IOI timeseries traces into a sequential representation of an HPC application.
- Transform single dimensional signal into events with associated timestamps, volume and bandwidths


Principle illustration
======================
As seen in Cluster Simulator package, to simulate an execution of an application, one needs to provide a basic formalism to describe the app behavior.
It consists of mainly three arrays of values:

- compute=[0, 15],  two events, first at 0 and second at 15, and compute is supposed by default between each consequent events.
- read=[3e9, 0], read 3GB at 0, before compute phase, at the end do nothing (0)
- write=[0, 5e9], write 5GB at first event, and 10GB at the second, after compute phase

.. code-block:: python
    :caption: application formalism

    # app example : read 3GB -> compute 15 seconds duration for 1 -> write 7GB
    app1 = Application( env, name="app1", # name of the app in the display
    compute=[0, 15],  # two events, first at 0 and second at 15, and compute between them
    read=[3e9, 0],    # read 3GB at 0, before compute phase, at the end do nothing (0)
    write=[0, 5e9],  # write 5GB at first event, and 10GB at the second, after compute phase
    data=data)        # collected data for monitoring

The role of the app decomposer module is to provide a complete application description from signals retrieved by the IO-Instrumentation (IOI) product. The IOI instruments job execution in an HPC cluster and collects many timeseries. Typically, each 5 seconds, IOI records in database the amount of data read by the application (summing all nodes volumes) et save it in the database, as well as the amount of total data written by the application. This will issue two timeseries ``read_volume`` and ``write_volume`` among others.

To illustrate this package ability, we will focus on the following synthetic signal sample.

.. code-block:: python
    :caption: synthetic timeseries

    timestamps = np.arange(6)
    read_signal = np.array([1, 1, 0, 0, 0, 0]) # 1MB read for each 1-value
    write_signal = np.array([0, 0, 0, 0, 1, 1]) # 1MB write for each 1-value

    # get the representation encoding from above timeseries
    jd = JobDecomposer() # init the job decomposer
    compute, reads, writes, read_bw, write_bw = jd.get_job_representation(merge_clusters=True)
    # This is the app encoding representation for Execution Simulator
    print(f"compute={compute}, reads={reads}, read_bw={read_bw}")
    print(f"compute={compute}, writes={writes}, write_bw={write_bw}")

We get the following representation output:

.. code-block:: python
    :caption: representation timeseries

    # two events, first one at 0 and the second one at 4
    # at first event : read 2MB of data at rate 1MB/s
    compute=[0, 4], reads=[2, 0], read_bw=[1.0, 0]
    # then waits 4 timestamps of compute phase for next event
    # at next event : writes 2MB of data at 1MB/s
    compute=[0, 4], writes=[0, 2], write_bw=[0, 1.0]


Now we will feed this representation to the cluster_simulator module:

.. code-block:: python
    :caption: feeding representation to simulator

    app = Application(self.env, name="#read-compute-write",
                        compute=[0, 4],
                        read=[2, 0],
                        write=[0, 2], data=data)

    self.env.process(app.run(cluster, placement=[0, 0, 0, 0, 0, 0]))
    self.env.run()


.. raw:: html
    :file: docs/figure_synthetic_signal.html

Decomposition steps
===================

Here is and example of an application traces collected from IO-Instrumentation database.
We can see on the figure below the collected data volume in MB for read and write accesses of the application.

.. raw:: html
    :file: docs/figure_timeseries_ioi_signal.html

Let's first extract the read signal and decomposes it into phases with I/O or compute phases.
The AppDecomposer will automatically detect I/O activity and spikes by classifying each signal point. If one hovers mouse above a signal point, the class shows up.

.. raw:: html
    :file: docs/decomposing_read_signal.html

Following the upper decomposition we get the event-based representation:

.. topic:: Representation elements

    events = [0, 1, 17, 19, 35, 36, 47, 48, 54],

    these are timestamps at which an I/O event is expected. These timestamps do not take into account the width of I/O spikes, as they are supposed to be infinitely narrow (dirac distribution). Between each two consecutive events, it is supposed that there is a pure compute phase.

    volumes = [22539772, 25123014, 166522759, 27021171, 152762756, 15660482, 130355500, 33094134, 0],

    the volumes are the sum of collected volume for each phase

    bandwidth = [2253977.2, 5024602.8, 16652275.9, 1801411.4, 15276275.6, 1566048.2, 26071100.0, 6618826.8, 0]

    in this example, the sampling period = 5s, so an I/O spike reaching 1MB for two consequent points will collect 2MB in 10s and reaches a bandwidth of 0.2MB/s.

We apply the same process for the write signal.

.. raw:: html
    :file: docs/decomposing_write_signal.html

And its detailed representation:

.. topic:: Representation elements

    events = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 36, 37, 38, 39, 40, 44, 45, 46, 47, 48, 49, 50]

    volumes = [0, 33068580, 73718411, 85482955, 39079032, 19204163, 22885166, 17794806, 12256739, 9045658, 10721933, 91289590, 90606733, 153142483, 17539498, 14972723, 16343069, 9816493, 14389077, 37970011, 8593287, 11060167, 8889613, 20927905, 23732530, 34655496, 45001180, 36068500, 19717086, 24719515, 15545906, 25512431, 12743845, 40467940, 13129635, 46422630, 7074681, 12118344, 0]

    bandwidth = [0, 6613716.0, 14743682.2, 8548295.5, 7815806.4, 3840832.6, 2288516.6, 3558961.2, 2451347.8, 1809131.6, 2144386.6, 9128959.0, 18121346.6, 15314248.3, 3507899.6, 2994544.6, 3268613.8, 1963298.6, 2877815.4, 3797001.1, 1718657.4, 2212033.4, 1777922.6, 2092790.5, 4746506.0, 6931099.2, 9000236.0, 7213700.0, 3943417.2, 2471951.5, 3109181.2, 1700828.7333333334, 2548769.0, 8093588.0, 2625927.0, 4642263.0, 1414936.2, 2423668.8, 0]

As we can see on the ``write`` timeseries, a unique I/O activity can have multiple sub-phases and levels of bandwidths. A user could want to group them into one unique and average checkpoint. The AppDecomposer has a specific ``merge`` option  for this:

.. code-block:: python
    :caption: enabling merge of same class points into unique spike

    # enabling merge of same class points
    write_dec = SignalDecomposer(write_signal, merge=True)
    write_bkps, write_labels = write_dec.decompose()



.. raw:: html
    :file: docs/decomposing_write_signal_with_merge.html

Now we can see that the write signal is decomposed into four distinct big checkpoints, with an averaged bandwidth for each. The checkpoints are separated by 0-class points.

This simplify drastically the representation (for the write signal) which becomes human-readible.

.. topic:: Representation elements

    events = [0, 1, 8, 12, 16, 17]

    volumes = [0, 323257443, 495540649, 224952644, 131957075, 0]

    bandwidth = [0, 5387624.05, 5829889.988235294, 4090048.0727272728, 3770202.1428571427, 0]


In order to validate that the volume conveyed by the application is kept consistent by the decomposition, we plot the cumulative volumes of the decomposition model in line with the cumulative volume of the original signal collected from IO-Instrumentation.


.. raw:: html
    :file: docs/decomposing_cumvol_write_signal_with_merge.html


* TODO

