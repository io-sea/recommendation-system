Jobmodel
========

Jobmodel build synthetic representation of an HPC job. It uses the collected time-series metrics to 
extract behavioral features of the analyzed job.  
It also proposes a module to compute distance between jobs, using the raw time-series metrics.

It is composed of several modules:
- to compute pairwise distance between jobs as multivariate time-series (_distances.py_),
- to extract features form jobs as multivariate time-series (_job_features.py_),

## Installation
_jobmodel_ respects the standard PyPi procedure for installation. Run in the root directory:

```python setup.py install```

or

```pip install .```

## Quick start

### distances.py

The ```DistanceJobs``` class permits to compute the distance between two jobs (multivariate time-series) from their iterator of connectors.

Using the iterators of connectors build with the ```PoolConnector``` (_ioanalytisctools_ library), you can compute the distance between two jobs as following:
```
from ioanalytisctools.api_connector import PoolConnector
from jobmodel.distances import DistanceJobs, fastdtw_dist

list_jobs = ['job_object_id_1', 'job_object_id_2']
list_ts = ["volume", "processCount"]
pool_cnct = PoolConnector('http://my-machine', 'user_api_token', list_jobs)
ts_pool = pool_cnct.pool_ts_connector(list_ts)

con1 = ts_pool['job_object_id_1']
con2 = ts_pool['job_object_id_2']

dist_ij = DistanceJobs(con1, con2, dist_fct=fastdtw_dist)
distance = dist_ij.dist_two_jobs()
```

Here the distance function usd to compute the distance between each time-series of jobs is DTW (```fastdtw_dist```).
A set of existing distance functions are available in this module.

You can develop your own distance function to be used with the ```DistanceJobs``` class. 
To be compliant with the usage, the distance function should respect the following interface:
```
def dist_custom(ts1, ts2, *args, **kwargs):
    """
    Args:
        ts1 (numpy array): the first time-series array
        ts2 (numpy array): the second time-series array
        *args: additional positional arguments
        **kwargs: additional keyword arguments

    Return:
        a scalar value of distance between ts1 and ts2
    """
    return distance_fct(*args, **kwargs)[0]
```

