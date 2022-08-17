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


Synthetic example
=================
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


* TODO

