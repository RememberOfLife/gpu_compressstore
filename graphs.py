#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

plt.rcParams.update({"font.size": 7})

# column indices
DATA_COL_DATABYTES = 0
DATA_COL_P = 1
DATA_COL_ALGO = 2
DATA_COL_CHUNKLENGTH = 3
DATA_COL_BLOCKS = 4
DATA_COL_THREADS = 5
DATA_COL_TIME = 6
DATA_COL_TROUGHPUT = 7

algo_colors = {
    "3pass_popc_none": "lightsteelblue",
    "cub_pss": "indigo",
    "3pass_pss_gmem": "blueviolet",
    "3pass_pss2_gmem": "plum",
    "3pass_fproc_none": "limegreen",
    "3pass_pproc_none": "darkgreen",
    "3pass_fproc_true": "dodgerblue",
    "3pass_pproc_true": "deepskyblue",
    "cub_flagged_bytemask": "orange",
    "cub_flagged_biterator": "gold",
    "single_thread": "orangered",
}

algo_markers = {
    "3pass_popc_none": "|",
    "cub_pss": "*",
    "3pass_pss_gmem": "^",
    "3pass_pss2_gmem": "v",
    "3pass_fproc_none": "1",
    "3pass_pproc_none": "2",
    "3pass_fproc_true": "3",
    "3pass_pproc_true": "4",
    "cub_flagged_bytemask": "d",
    "cub_flagged_biterator": "d",
    "single_thread": ".",
}

def read_data(path):
    data = []
    with open(path) as file:
        csv_reader = csv.DictReader(file, delimiter=";")
        for row in csv_reader:
            data_row = []
            # databytes;p;algo;chunklength;blocks;threads;time
            data_row.append(int(row["databytes"]))
            data_row.append(float(row["p"]))
            data_row.append(str(row["algo"]))
            data_row.append(int(row["chunklength"]))
            data_row.append(int(row["blocks"]))
            data_row.append(int(row["threads"]))
            time = float(row["time"])
            if time < 0:
                print("validation failed for " + data_row)
                exit()
            data_row.append(time)
            data.append(data_row)
    return data

def average_data(data):
    runs = {}
    times = {}
    new_data = []
    for row in data:
        row_key = tuple(row[:-1]) # exclude time in key
        run_count = runs.get(row_key, 0)
        runs[row_key] = run_count + 1
        time = times.get(row_key, 0)
        times[row_key] = time + row[DATA_COL_TIME]
    for key in times.keys():
        # save GiB/s throughput in the time column
        time = (key[DATA_COL_DATABYTES]/1024/1024/1024) / ((times[key] / runs[key]) / 1000)
        new_data.append(list(key)+[(times[key] / runs[key]), time])
    return new_data

def map_best_dimensions(data):
    bests = {}
    for row in data:
        # create key from databytes;p;algo;chunklength
        row_key = tuple(row[:(DATA_COL_CHUNKLENGTH+1)])
        # result_config is tuple of (throughput, blocks, threads)
        best_result_config = bests.get(row_key, (0, 0, 0))
        if row[DATA_COL_TROUGHPUT] > best_result_config[0]:
            bests[row_key] = (row[DATA_COL_TROUGHPUT], row[DATA_COL_BLOCKS], row[DATA_COL_THREADS])
    return bests

def dg_throughput_over_datasize(data, outdirpath):
    fixed_chunklength = 1024
    fixed_p = 0.5
    # filter for chunk,p and accumulate ordered list of datasizes
    datasizes = {}
    algos = {}
    algo_chunklengths = {}
    l_data = []
    for row in data:
        if row[DATA_COL_CHUNKLENGTH] in [fixed_chunklength, 0] and row[DATA_COL_P] == fixed_p:
            datasizes[row[DATA_COL_DATABYTES]] = None
            algos[row[DATA_COL_ALGO]] = None
            algo_chunklengths[row[DATA_COL_ALGO]] = row[DATA_COL_CHUNKLENGTH]
            l_data.append(row)
    datasizes = sorted(datasizes.keys())
    algos = sorted(algos.keys())
    # map to best dimensions
    l_best = map_best_dimensions(l_data)
    # for each algo create list of throughput values for different datasizes
    algo_troughputs = {}
    for alg in algos:
        algo_troughput = []
        for datasize in datasizes:
            algo_troughput.append(l_best[(datasize,fixed_p,alg,algo_chunklengths[alg])][0])
        algo_troughputs[alg] = algo_troughput
    # human readable datasizes
    datasizesHR = [int(ds/1024/1024) for ds in datasizes]
    # draw the thing
    fig, ax = plt.subplots()
    for alg in algos:
        ax.plot(
            datasizesHR,
            algo_troughputs[alg],
            color=algo_colors[alg],
            marker=algo_markers[alg],
            label=alg)
    ax.set_xlabel("Data in MiB")
    ax.set_xscale("log", base=2)
    ax.set_xticks(datasizesHR)
    ax.set_xticklabels(datasizesHR)
    ax.set_ylabel("Throughput (GiB/s)")
    ax.set_yscale("log", base=10)
    ax.set_title(f"Throughput at p={fixed_p} and chunklength={fixed_chunklength}")
    ax.legend()
    fig.savefig("throughput_over_datasize.png", dpi=200)

def main():
    args = sys.argv
    input_path = ""
    outdirpath = "./graphs/"
    if len(args) > 1:
        input_path = args[1]
    else:
        exit()
    data = read_data(input_path)
    # condense multiple runs of same type by average
    data = average_data(data)
    # diagrams
    dg_throughput_over_datasize(data, outdirpath)
    print("done")

if __name__ == "__main__":
    main()
