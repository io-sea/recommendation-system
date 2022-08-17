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


Examples
========
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


* TODO

