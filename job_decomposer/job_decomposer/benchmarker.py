import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ruptures as rpt
import os, random, time

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
    
if __name__ == '__main__':
    
    job_files, job_ids, datasets = list_jobs()


    df = pd.DataFrame(columns=["dataset", "jobid", "timeserie", "signal_length", "n_brkpts",
                            "algo_name", "cost_function", "loss", "penalty", "computation_time"])
    
    

    algos_names = ["Pelt", "Binseg", "Botup", "Wind"]
    
    output_file = os.path.join("/home_nfs/mimounis/iosea-wp3-recommandation-system/job_decomposer/job_decomposer", "_".join(algos_names) + "_bench.csv")
    
    
    for job_file, job_id, dataset in zip(job_files, job_ids, datasets):
        df_signal = pd.read_csv(job_file, index_col=0)
        for ts in ["bytesWritten", "bytesRead"]:
            signal = df_signal[[ts]].to_numpy()
            signal_dim = signal.shape[1]

            # define costs
            costs = [rpt.costs.CostL1(), rpt.costs.CostL2(), rpt.costs.CostNormal(), rpt.costs.CostLinear(),
                    rpt.costs.CostCLinear(), rpt.costs.CostAR(order=3), rpt.costs.CostMl(metric=np.eye(signal_dim)), rpt.costs.CostRank()]
            costs_names = ["L1", "L2", "Gauss", "Linear", "Clinear", "AR", "Mala", "Rank"]

            # algos = [rpt.Pelt(custom_cost=cost, min_size=2), rpt.Binseg(custom_cost=cost),
            #          rpt.BottomUp(custom_cost=cost), rpt.Window(custom_cost=cost, width=10)]

            algos = [rpt.Pelt, rpt.Binseg, rpt.BottomUp, rpt.Window]

            algos_names = ["Pelt", "Binseg", "Botup", "Wind"]

            penalties = np.logspace(-3, 3, num=6).tolist()
    
            for algo, algo_name in zip(algos[0:3], algos_names):
                for cost, costs_name in zip(costs[0:3], costs_names):
                    for pen in penalties:
                        print(f"{job_id=} | {algo_name=} | {costs_name=} | {pen=}")
                        try:
                            start_time = time.time()
                            result = algo(custom_cost=cost, min_size=2).fit_predict(signal, pen)
                            duration = time.time() - start_time
                            n_brkpts = len(result) - 1
                            loss = cost.sum_of_costs(result)
                            df = df.append({"dataset": dataset,
                                            "jobid": job_id,
                                            "timeserie": ts,
                                            "signal_length": signal.shape[0],
                                            "n_brkpts": n_brkpts,
                                            "algo_name": algo_name,
                                            "cost_function": costs_name,
                                            "loss": loss,
                                            "penalty": pen,
                                            "computation_time": duration,
                                            },
                                        ignore_index=True)
                            
                        except:
                            pass
                        
            df.to_csv(output_file)
                        
    
    #df.to_csv("_".join(algos_names) + "_bench.csv")
    
        
        