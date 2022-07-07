import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ruptures as rpt
import os, random, time
from kneed import KneeLocator

import warnings
warnings.filterwarnings("ignore")

# choose a job
def list_jobs(dataset_path = "/home_nfs/mimounis/iosea-wp3-recommandation-system/dataset_generation/dataset_generation"):
    # /home_nfs/mimounis/iosea-wp3-recommandation-system/dataset_generation/dataset_generation
    job_files = []
    job_ids = []
    dataset = []
    for root, dirs, files in os.walk(dataset_path):
        for csv_file in files:
            if csv_file.endswith(".csv"):
                job_files.append(os.path.join(root, csv_file))
                job_ids.append(csv_file.split("_")[-1].split(".csv")[0])
                dataset.append(os.path.split(root)[-1])
    return job_files, job_ids, dataset

def get_sum_of_cost(algo, n_bkps) -> float:
    """Return the sum of costs for the change points `bkps`"""
    bkps = algo.predict(n_bkps=n_bkps)
    return algo.cost.sum_of_costs(bkps)

def get_optimal_breakpoints(signal, algo=rpt.KernelCPD, kernel="linear"):
    initialized_algo = algo(kernel=kernel, min_size=1).fit(signal)
    n_bkps_max = int((np.sqrt(len(signal))))
    # segment
    _ = initialized_algo.predict(n_bkps_max)
    array_of_n_bkps = np.arange(1, n_bkps_max + 1)
    costs = [get_sum_of_cost(algo=initialized_algo, n_bkps=n_bkps) for n_bkps in array_of_n_bkps]
    elbow = KneeLocator(array_of_n_bkps, costs, S= 1,
                         curve="convex",  direction="decreasing").elbow
    optimal_n_breakpoints = elbow if elbow else n_bkps_max
    optimal_breakpoints = initialized_algo.predict(n_bkps=optimal_n_breakpoints)
    optimal_cost = costs[array_of_n_bkps.tolist().index(optimal_n_breakpoints)]
    
    return optimal_n_breakpoints, optimal_breakpoints, optimal_cost

if __name__ == '__main__':
    
    job_files, job_ids, datasets = list_jobs("/home_nfs/mimounis/iosea-wp3-recommandation-system/dataset_generation/dataset_generation")

    df = pd.DataFrame(columns=["dataset", "jobid", "timeserie", "signal_length", "n_brkpts",
                            "kernel", "computation_time", "loss"])
       

    # algos_names = ["KernelCPD", "Binseg", "Botup", "Wind"]
    
    output_file = os.path.join("/home_nfs/mimounis/iosea-wp3-recommandation-system/job_decomposer/job_decomposer", "kernel_cpd.csv")
    
    
    for job_file, job_id, dataset in zip(job_files, job_ids, datasets):
        df_signal = pd.read_csv(job_file, index_col=0)
        for ts in ["bytesWritten", "bytesRead"]:
            signal = df_signal[[ts]].to_numpy()
            signal_dim = signal.shape[1]

            # define costs
            kernels = ["linear", "rbf"]

            # algos = [rpt.Pelt(custom_cost=cost, min_size=2), rpt.Binseg(custom_cost=cost),
            #          rpt.BottomUp(custom_cost=cost), rpt.Window(custom_cost=cost, width=10)]

            # algos = [rpt.Pelt, rpt.Binseg, rpt.BottomUp, rpt.Window]

            # algos_names = ["Pelt", "Binseg", "Botup", "Wind"]

    
            
            for kernel in kernels:
                print(f"{dataset=} | {job_id=} | {kernel=} | signal length = {len(signal)} points / {len(signal)/12/60} hours")
                try:
                    start_time = time.time()
                    optimal_n_breakpoints, optimal_breakpoints, optimal_cost = get_optimal_breakpoints(signal, kernel=kernel)
                    duration = time.time() - start_time
                    
                    df = df.append({"dataset": dataset,
                                    "jobid": job_id,
                                    "timeserie": ts,
                                    "signal_length": signal.shape[0],
                                    "n_brkpts": optimal_n_breakpoints,
                                    "kernel": kernel,
                                    "loss": optimal_cost,
                                    "computation_time": duration,
                                    },
                                ignore_index=True)
                        
                except:
                    pass
                    
            print(f"updating {output_file}")
            df.to_csv(output_file)
            
                        
    
    #df.to_csv("_".join(algos_names) + "_bench.csv")
    
        
        