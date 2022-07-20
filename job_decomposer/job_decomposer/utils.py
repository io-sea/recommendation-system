import matplotlib.pyplot as plt
import pandas as pd
import ruptures as rpt
import pwlf
import numpy as np
from scipy import integrate
import os, random
from kneed import KneeLocator
import numpy as np
import time
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def choose_random_job(dataset_path = "C:\\Users\\a770398\\IO-SEA\\io-sea-3.4-analytics\\dataset_generation\\dataset_generation"):
    job_files = []
    for root, dirs, files in os.walk(dataset_path):
        for csv_file in files:
            if csv_file.endswith(".csv"):
                job_files.append(os.path.join(root, csv_file))
    return random.choice(job_files)


def list_jobs(dataset_path = "C:\\Users\\a770398\\IO-SEA\\io-sea-3.4-analytics\\dataset_generation\\dataset_generation"):
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



def plot_job_phases(x, signal, breakpoints):
    
    x = ((np.array(x) - x[0])/5).tolist()
    plt.rcParams["figure.figsize"] = (12, 5)
    plt.rcParams['figure.facecolor'] = 'gray'
    
    plt.plot(x, signal, lw=2, label="IO")
    for i_brk, brk in enumerate(breakpoints[:-1]): #avoid last point
        if i_brk % 2 == 0: # opening point
            plt.plot(x[brk], signal[brk], '>g')
        else:
            plt.plot(x[brk], signal[brk], '<r')
    plt.grid(True)
    plt.show()
    
def plot_job_phases2(x, signal, breakpoints):
    signal = np.array(signal)
    #x = ((np.array(x) - x[0])/5).tolist()
    fig, ax = rpt.display(signal, breakpoints)
    #ax[0].plot(x, signal, lw=2, label="IO", color='k')
    
    for i_brk, brk in enumerate(breakpoints[:-1]): #avoid last point
        if i_brk % 2 == 0: # opening point
            ax[0].plot([brk], signal[brk], '>g')
        else:
            ax[0].plot([brk], signal[brk], '<r')
    plt.grid(True)
    plt.show()
    
    
def get_df_by_job_id(job_id=1391):
    job_files, job_ids, datasets = list_jobs()
    csv_file = job_files[job_ids.index(str(job_id))]
    print(f"retrieving {csv_file}")
    return pd.read_csv(csv_file, index_col=0)


def plot_bench_decomposition(bench_file="Pelt_Binseg_Botup_Wind_bench.csv", idx=0):
        

    bench_df = pd.read_csv(bench_file, index_col=0)
    if idx == "rnd":
        idx = int(bench_df.sample().index.values[0])

    algos = [rpt.Pelt, rpt.Binseg, rpt.BottomUp, rpt.Window]
    algos_names = ["Pelt", "Binseg", "Botup", "Wind"]
    # define costs
    costs = [rpt.costs.CostL1(), rpt.costs.CostL2(), rpt.costs.CostNormal(), rpt.costs.CostLinear(),
            rpt.costs.CostCLinear(), rpt.costs.CostAR(order=3), rpt.costs.CostMl(metric=np.eye(1)), rpt.costs.CostRank()]
    costs_names = ["L1", "L2", "Gauss", "Linear", "Clinear", "AR", "Mala", "Rank"]

    #df.iloc[idx]["jobid"]
    algo = algos[algos_names.index(bench_df.iloc[idx]["algo_name"])]
    cost = costs[costs_names.index(bench_df.iloc[idx]["cost_function"])]
    penalty = bench_df.iloc[idx]["penalty"]
    timeserie = bench_df.iloc[idx]["timeserie"]

    job_files, job_ids, datasets = list_jobs()
    str_jobid = str(bench_df.iloc[idx]["jobid"])
    print(f"{algo = }")
    print(f"{cost = }")
    csv_file = job_files[job_ids.index(str_jobid)]
    print(f"{csv_file = }")
    df = pd.read_csv(csv_file, index_col=0)

    signal = df[[timeserie]].to_numpy().tolist()
    x = df[["timestamp"]].to_numpy().tolist()
    print(f"length of signal = {len(signal)}")
    breakpoints = algo(custom_cost=cost, min_size=1, jump=1).fit_predict(df[[timeserie]].to_numpy(), pen=penalty)
    loss = cost.sum_of_costs(breakpoints)
    print(f"{penalty = }")
    print(f"{loss = }")
    print(f"number of breakpoints = {len(breakpoints)-1}")
        
    plot_job_phases2(x, signal, breakpoints)
    

def plot_bench_decomposition_cpd(bench_file="kernel_cpd.csv", idx = 0):
    bench_df = pd.read_csv(bench_file, index_col=0)
    
    if idx == "rnd":
        idx = int(bench_df.sample().index.values[0])

    algo = rpt.KernelCPD
    # define costs
    kernels = ["rbf", "linear"]

    kernel = bench_df.iloc[idx]["kernel"]
    n_brkpts = bench_df.iloc[idx]["n_brkpts"]


    job_files, job_ids, datasets = list_jobs()
    str_jobid = str(bench_df.iloc[idx]["jobid"])
    csv_file = job_files[job_ids.index(str_jobid)]
    print(f"{csv_file = }")
    df = pd.read_csv(csv_file, index_col=0)

    timeserie = bench_df.iloc[idx]["timeserie"]
    signal = df[[timeserie]].to_numpy().tolist()
    x = df[["timestamp"]].to_numpy().tolist()
    print(f"length of signal = {len(signal)}")
    breakpoints = algo(kernel=kernel, min_size=1).fit_predict(df[[timeserie]].to_numpy(),
                                                            n_bkps=n_brkpts)
    print(f"{breakpoints = }")
    print(f"number of breakpoints = {len(breakpoints)-1}")
        
    plot_job_phases2(x, signal, breakpoints)
    

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


def get_kmeans_breakpoints(signal, v0_threshold = 0.05):
    """_summary_

    Args:
        signal (ndarray): _description_
    """
    v0_weight = 1
    n_clusters = 2
    while v0_weight > v0_threshold:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(signal)
        v0_indexes = np.where(kmeans.labels_==0, True, False)
        v0_weight = signal[v0_indexes].sum() / signal.sum()
        n_clusters += 1
        if n_clusters >= len(signal):
            # if n_clusters equals signal points, exit the loop
            break
        
    ab = np.arange(len(signal))
    # merge v1+ clusters together
    labels = np.where(kmeans.labels_ > 0, 1, 0)
    return  ab[np.insert(np.where(np.diff(kmeans.labels_)!=0, True, False), 0, False)]



def clustering(signal, n):
    k_means = KMeans(init='k-means++', n_clusters=n, n_init=10)
    t0 = time.time()
    k_means.fit(signal)
    t_batch = time.time() - t0
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    #print(k_means_labels, t_batch)
    return k_means_labels

def find_best_cluster(signal, min_diff=0.01, max_n=8):
    step = 2
    n_clusters = step
    n_best = step
    best_score = -1
    n_samples = len(signal)
    best_preds = np.random.choice(n_clusters, n_samples)  # init with random labels

    while step <= max_n and n_clusters < n_samples:
        preds = clustering(signal, step)
        score = silhouette_score(signal, preds, metric="euclidean")
        #print(score)
        if score > best_score * (1 + min_diff):  # if better score
            best_score = score
            best_preds = preds
            n_best = step
        step += 1
        n_clusters += 1
    
    return best_preds, n_best

def changepoint_detection(changes):
    result=[]
    
    for idx in range(len(changes)-1):
        if changes[idx] != changes[idx+1]:
            #print(idx)
            result.append(idx+1)
    return result


def get_kmeans_original_breakpoints(signal):
    k_means_labels, n = find_best_cluster(signal)
    result = changepoint_detection(k_means_labels)
    return result