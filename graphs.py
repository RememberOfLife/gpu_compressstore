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
    "copy": "slategrey",
    "copy_add": "slategrey",
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
    "avx512": "firebrick",
    "async_streaming_3pass": "royalblue",
    "cpu": "lightcoral",
    # packages
    "3pass_fproc_true_sa": "dodgerblue",
    "3pass_fproc_true_cub": "deepskyblue",
    "3pass_pproc_true_cub": "limegreen",
    "3pass_pss_gmem_full": "crimson",
}

algo_markers = {
    "copy": None,
    "copy_add": None,
    "3pass_popc_none": "|",
    "cub_pss": "*",
    "3pass_pss_gmem": "^",
    "3pass_pss2_gmem": "v",
    "3pass_fproc_none": "1",
    "3pass_pproc_none": "2",
    "3pass_fproc_true": "3",
    "3pass_pproc_true": "4",
    "cub_flagged_bytemask": "D",
    "cub_flagged_biterator": "d",
    "single_thread": ".",
    "avx512": ".",
    "async_streaming_3pass": ".",
    "cpu": ".",
    # packages
    "3pass_fproc_true_sa": "X",
    "3pass_fproc_true_cub": "x",
    "3pass_pproc_true_cub": "x",
    "3pass_pss_gmem_full": "+",
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
                print(f"validation failed for {data_row}")
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
        throughput = (key[DATA_COL_DATABYTES]/1024/1024/1024) / ((times[key] / runs[key]) / 1000)
        new_data.append(list(key)+[(times[key] / runs[key]), throughput])
    return new_data

def map_best_dimensions(data):
    bests = {}
    for row in data:
        # create key from databytes;p;algo;chunklength
        row_key = tuple(row[:(DATA_COL_CHUNKLENGTH+1)])
        # result_config is tuple of (time, throughput, blocks, threads)
        best_result_config = bests.get(row_key, (0, 0, 0, 0))
        if row[DATA_COL_TROUGHPUT] > best_result_config[1]:
            bests[row_key] = (row[DATA_COL_TIME], row[DATA_COL_TROUGHPUT], row[DATA_COL_BLOCKS], row[DATA_COL_THREADS])
    return bests

# there is huge overlap in the code of the diagram computations, it can be removed and simplified using shared methods

def dg_runtime_over_datasize(data, outdirpath, filename, p):
    fixed_chunklength = 1024
    fixed_p = p
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
    # for each algo create list of time values for different datasizes
    algo_times = {}
    for alg in algos:
        algo_time = []
        for datasize in datasizes:
            algo_time.append(l_best[(datasize,fixed_p,alg,algo_chunklengths[alg])][0])
        algo_times[alg] = algo_time
    # human readable datasizes
    datasizesHR = [int(ds/1024/1024) for ds in datasizes]
    # draw the thing
    fig, ax = plt.subplots()
    for alg in algos:
        ax.plot(
            datasizesHR,
            algo_times[alg],
            color=algo_colors[alg],
            marker=algo_markers[alg],
            label=alg)
    ax.set_xlabel("Data in MiB")
    ax.set_xscale("log", base=2)
    ax.set_xticks(datasizesHR)
    ax.set_xticklabels(datasizesHR)
    ax.set_ylabel("Runtime in ms")
    ax.set_yscale("log", base=10)
    ax.set_title(f"Runtime over datasize at p={fixed_p}, chunklength={fixed_chunklength}")
    ax.legend()
    fig.savefig(outdirpath+filename+".png", dpi=200)
    plt.close(fig)

def dg_package_throughput_over_datasize(data, outdirpath, filename, p):
    fixed_chunklength = 1024
    fixed_p = p
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
    # for each algo create list of time values for different datasizes
    algo_times = {}
    for alg in algos:
        algo_time = []
        for datasize in datasizes:
            algo_time.append(l_best[(datasize,fixed_p,alg,algo_chunklengths[alg])][0])
        algo_times[alg] = algo_time
    datapoint_count = len(datasizes)
    # package algos together
    packages = {
        "cub_flagged_biterator": ["cub_flagged_biterator"],
        "cub_flagged_bytemask": ["cub_flagged_bytemask"],
        "3pass_fproc_true_sa": ["3pass_popc_none", "3pass_popc_none", "3pass_pss_gmem", "3pass_pss2_gmem", "3pass_fproc_true"],
        "3pass_fproc_true_cub": ["3pass_popc_none", "3pass_popc_none", "cub_pss", "3pass_fproc_true"],
        "3pass_pproc_true_cub": ["3pass_popc_none", "3pass_popc_none", "cub_pss", "3pass_pproc_true"],
        "async_streaming_3pass": ["async_streaming_3pass"],
        #"cub_pss": ["3pass_popc_none", "cub_pss"],
        #"3pass_pss_gmem_full": ["3pass_popc_none", "3pass_pss_gmem", "3pass_pss2_gmem"],
    }
    for package in packages.keys():
        algo_time = [0]*datapoint_count
        for sub_algo in packages[package]:
            for i in range(datapoint_count):
                # calculate throughput
                data_amount = datasizes[i]
                data_amount *= p * 2
                data_amount += datasizes[i]/8/8
                if "pss" in package:
                    data_amount = ((datasizes[i]/8/8) / fixed_chunklength) * 4 # /8 for using uint64_t and /8 for 8bits per bytemask byte
                data_amount = (data_amount/1024/1024/1024)
                algo_time[i] = data_amount / (algo_times[sub_algo][i] / 1000)
        algo_times[package] = algo_time
    algos = packages.keys()
    # human readable datasizes
    datasizesHR = [int(ds/1024/1024) for ds in datasizes]
    # draw the thing
    fig, ax = plt.subplots()
    for alg in algos:
        ax.plot(
            datasizesHR,
            algo_times[alg],
            color=algo_colors[alg],
            marker=algo_markers[alg],
            label=alg)
    # plot copy throughput
    # ax.plot(
    #     datasizesHR,
    #     [565.82]*datapoint_count, # hardcoded copy-add kernel performance
    #     color=algo_colors["copy_add"],
    #     marker=algo_markers["copy_add"],
    #     label="copy_add",
    #     dashes=[1, 1]
    # )
    ax.set_xlabel("Data in MiB")
    ax.set_xscale("log", base=2)
    ax.set_xticks(datasizesHR)
    ax.set_xticklabels(datasizesHR)
    ax.set_ylabel("Throughput in GiB/s")
    #ax.set_yscale("log", base=10)
    ax.set_title(f"Throughput over datasize at p={fixed_p}, chunklength={fixed_chunklength}")
    ax.legend()
    fig.savefig(outdirpath+filename+".png", dpi=200)
    plt.close(fig)

def dg_package_throughput_over_chunklength(data, outdirpath, filename, p):
    fixed_datasize = 2**30 * 8
    fixed_datasizeHR = fixed_datasize/1024/1024
    fixed_p = p
    # filter and accumulate
    chunklengths = {}
    algos = {}
    l_data = []
    for row in data:
        if row[DATA_COL_DATABYTES] == fixed_datasize and row[DATA_COL_P] == fixed_p:
            chunklengths[row[DATA_COL_CHUNKLENGTH]] = None
            algos[row[DATA_COL_ALGO]] = None
            l_data.append(row)
    chunklengths = sorted(chunklengths.keys())
    chunklengths.remove(0)
    algos = sorted(algos.keys())
    # map to best dimensions
    l_best = map_best_dimensions(l_data)
    # for each algo create list of time values for different chunklengths
    datapoint_count = len(chunklengths)
    algo_times = {}
    for alg in algos:
        algo_time = []
        for chunklength in [0]+chunklengths:
            time = l_best.get((fixed_datasize,fixed_p,alg,chunklength), (-1,))[0]
            if chunklength == 0 and time >= 0:
                algo_time = [time]*datapoint_count
                break
            elif chunklength == 0:
                continue
            algo_time.append(time)
        algo_times[alg] = algo_time
    # package algos together
    packages = {
        "cub_flagged_biterator": ["cub_flagged_biterator"],
        "cub_flagged_bytemask": ["cub_flagged_bytemask"],
        "3pass_fproc_true_sa": ["3pass_popc_none", "3pass_popc_none", "3pass_pss_gmem", "3pass_pss2_gmem", "3pass_fproc_true"],
        "3pass_fproc_true_cub": ["3pass_popc_none", "3pass_popc_none", "cub_pss", "3pass_fproc_true"],
        "3pass_pproc_true_cub": ["3pass_popc_none", "3pass_popc_none", "cub_pss", "3pass_pproc_true"],
        "async_streaming_3pass": ["async_streaming_3pass"],
    }
    for package in packages.keys():
        algo_time = [0]*datapoint_count
        for sub_algo in packages[package]:
            for i in range(datapoint_count):
                # calculate throughput
                data_amount = fixed_datasize
                data_amount *= p * 2
                data_amount += fixed_datasize/8/8
                data_amount = (data_amount/1024/1024/1024)
                algo_time[i] = data_amount / (algo_times[sub_algo][i] / 1000)
        algo_times[package] = algo_time
    algos = packages.keys()
    # draw the thing
    fig, ax = plt.subplots()
    for alg in algos:
        ax.plot(
            chunklengths,
            algo_times[alg],
            color=algo_colors[alg],
            marker=algo_markers[alg],
            label=alg)
    # plot copy throughput
    # ax.plot(
    #     chunklengths,
    #     [565.82]*datapoint_count, # hardcoded copy-add kernel performance
    #     color=algo_colors["copy_add"],
    #     marker=algo_markers["copy_add"],
    #     label="copy_add",
    #     dashes=[1, 1]
    # )
    ax.set_xlabel("Chunklength in Bits")
    ax.set_xscale("log", base=2)
    ax.set_xticks(chunklengths)
    ax.set_xticklabels(chunklengths)
    ax.set_ylabel("Throughput in GiB/s")
    #ax.set_yscale("log", base=10)
    ax.set_title(f"Throughput over chunklength at p={fixed_p}, datasize={int(fixed_datasizeHR)}MiB")
    ax.legend()
    fig.savefig(outdirpath+filename+".png", dpi=200)
    plt.close(fig)

def dg_package_throughput_over_p(data, outdirpath, filename, throughput):
    fixed_chunklength = 1024
    fixed_datasize = 2**30 * 8
    fixed_datasizeHR = fixed_datasize/1024/1024
    # filter and accumulate
    algo_chunklengths = {}
    datasizes = {}
    algos = {}
    p_vals = {}
    l_data = []
    for row in data:
        if row[DATA_COL_CHUNKLENGTH] in [fixed_chunklength, 0] and row[DATA_COL_DATABYTES] == fixed_datasize:
            algos[row[DATA_COL_ALGO]] = None
            p_vals[row[DATA_COL_P]] = None
            algo_chunklengths[row[DATA_COL_ALGO]] = row[DATA_COL_CHUNKLENGTH]
            l_data.append(row)
    p_vals = list(sorted(p_vals.keys()))
    algos = sorted(algos.keys())
    # map to best dimensions
    l_best = map_best_dimensions(l_data)
    # for each algo create list of time values for different datasizes
    algo_times = {}
    for alg in algos:
        algo_time = []
        for p_val in p_vals:
            algo_time.append(l_best[(fixed_datasize,p_val,alg,algo_chunklengths[alg])][0])
        algo_times[alg] = algo_time
    datapoint_count = len(p_vals)
    # package algos together
    packages = {
        "cub_flagged_biterator": ["cub_flagged_biterator"],
        "cub_flagged_bytemask": ["cub_flagged_bytemask"],
        "3pass_fproc_true_sa": ["3pass_popc_none", "3pass_popc_none", "3pass_pss_gmem", "3pass_pss2_gmem", "3pass_fproc_true"],
        "3pass_fproc_true_cub": ["3pass_popc_none", "3pass_popc_none", "cub_pss", "3pass_fproc_true"],
        "3pass_pproc_true_cub": ["3pass_popc_none", "3pass_popc_none", "cub_pss", "3pass_pproc_true"],
        "async_streaming_3pass": ["async_streaming_3pass"],
    }
    for package in packages.keys():
        algo_time = [0]*datapoint_count
        for sub_algo in packages[package]:
            for i in range(datapoint_count):
                # calculate throughput
                data_amount = fixed_datasize
                data_amount *= p_vals[i] * 2
                data_amount += fixed_datasize/8/8
                data_amount = (data_amount/1024/1024/1024)
                if throughput:
                    algo_time[i] = data_amount / (algo_times[sub_algo][i] / 1000) # this for throughput
                else:
                    algo_time[i] = algo_times[sub_algo][i] # this for runtime
        algo_times[package] = algo_time
    algos = packages.keys()
    # draw the thing
    fig, ax = plt.subplots()
    for alg in algos:
        ax.plot(
            range(len(p_vals)),
            algo_times[alg],
            color=algo_colors[alg],
            marker=algo_markers[alg],
            label=alg)
    # plot copy throughput
    # ax.plot(
    #     datasizesHR,
    #     [565.82]*datapoint_count, # hardcoded copy-add kernel performance
    #     color=algo_colors["copy_add"],
    #     marker=algo_markers["copy_add"],
    #     label="copy_add",
    #     dashes=[1, 1]
    # )
    ax.set_xlabel("Data in MiB")
    #ax.set_xscale("log", base=10)
    ax.set_xticks([i for i in range(len(p_vals))])
    #ax.set_xticks(p_vals)
    ax.set_xticklabels(p_vals)
    ax.set_ylabel("Throughput in GiB/s" if throughput else "Runtime in ms")
    #ax.set_yscale("log", base=10)
    title = "Throughput" if throughput else "Runtime" # cant get inlined for some reason
    ax.set_title(f"{title} over p at chunklength={fixed_chunklength}, datasize={int(fixed_datasizeHR)}MiB")
    ax.legend()
    fig.savefig(outdirpath+filename+".png", dpi=200)
    plt.close(fig)

def dg_runtime_heatmap_griddim_blockdim(data, outdirpath, filename, algo, p):
    # only available for algos with a chunklength
    fixed_chunklength = 1024
    fixed_datasize = 2**30 * 8
    fixed_datasizeHR = fixed_datasize/1024/1024
    # filter and accumulate
    griddims = {} # blocks
    blockdims = {} # threads
    l_data = {}
    for row in data:
        if row[DATA_COL_CHUNKLENGTH] == fixed_chunklength and row[DATA_COL_DATABYTES] == fixed_datasize and row[DATA_COL_P] == p and row[DATA_COL_ALGO] == algo and row[DATA_COL_BLOCKS] > 0:
            griddims[row[DATA_COL_BLOCKS]] = None
            blockdims[row[DATA_COL_THREADS]] = None
            l_data[(row[DATA_COL_BLOCKS], row[DATA_COL_THREADS])] = row[DATA_COL_TIME]
    griddims = list(reversed(list(sorted(griddims.keys()))))
    blockdims = list(sorted(blockdims.keys()))
    # draw the thing
    fig, ax = plt.subplots()

    matrix = []
    for i in range(len(griddims)):
        mrow = []
        for j in range(len(blockdims)):
            mrow.append(l_data[(griddims[i], blockdims[j])])
            ax.text(j, i, round(l_data[(griddims[i], blockdims[j])], 2), ha="center", va="center", color="crimson")
        matrix.append(mrow)

    mapcolor = "Blues_r"
    im = ax.imshow(
        matrix,
        cmap=mapcolor,
    )

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("runtime in ms", rotation=-90, va="bottom")

    ax.set_xlabel("blockdim")
    ax.set_xticks(range(len(blockdims)))
    ax.set_xticklabels(blockdims)
    #ax.set_xscale("log", base=2)
    ax.set_ylabel("griddim")
    ax.set_yticks(range(len(griddims)))
    ax.set_yticklabels(griddims)
    #ax.set_yscale("log", base=2)
    ax.set_title(f"{algo} runtime heatmap at p={p}, chunklength={fixed_chunklength}, datasize={int(fixed_datasizeHR)}MiB")
    fig.savefig(outdirpath+filename+".png", dpi=200)
    plt.close(fig)

def main():
    args = sys.argv
    input_paths = []
    outdirpath = "./graphs/"
    if len(args) == 4:
        # order is uniform,zipf,burst
        input_paths = args[1:]
    else:
        exit()
    dataUniform = read_data(input_paths[0])
    dataZipf = read_data(input_paths[1])
    dataBurst = read_data(input_paths[2])
    # condense multiple runs of same type by average
    dataUniform = average_data(dataUniform)
    dataZipf = average_data(dataZipf)
    zipf_p = 0.29 # hard coded p value of the zipf data
    dataBurst = average_data(dataBurst)
    burst_p = 0.5 # hard coded p value of the burst data
    # diagrams
    dg_runtime_over_datasize(dataUniform, outdirpath, "runtime_over_datasize_dense", 0.995)
    dg_runtime_over_datasize(dataUniform, outdirpath, "runtime_over_datasize_normal", 0.5)
    dg_runtime_over_datasize(dataUniform, outdirpath, "runtime_over_datasize_sparse", 0.005)
    dg_package_throughput_over_datasize(dataUniform, outdirpath, "package_throughput_over_datasize_normal", 0.5)
    dg_package_throughput_over_datasize(dataUniform, outdirpath, "package_throughput_over_datasize_sparse", 0.005)
    dg_package_throughput_over_chunklength(dataUniform, outdirpath, "package_throughput_over_chunklength_normal", 0.5)
    dg_package_throughput_over_chunklength(dataUniform, outdirpath, "package_throughput_over_chunklength_sparse", 0.005)
    dg_package_throughput_over_p(dataUniform, outdirpath, "package_throughput_over_p", True)
    dg_package_throughput_over_p(dataUniform, outdirpath, "package_runtime_over_p", False)
    #TODO make sparse and dense masks
    dg_runtime_heatmap_griddim_blockdim(dataUniform, outdirpath, "runtime_heatmap_dimensioning_3pass_popc_none", "3pass_popc_none", 0.5)
    dg_runtime_heatmap_griddim_blockdim(dataUniform, outdirpath, "runtime_heatmap_dimensioning_3pass_pss_gmem", "3pass_pss_gmem", 0.5)
    dg_runtime_heatmap_griddim_blockdim(dataUniform, outdirpath, "runtime_heatmap_dimensioning_3pass_pss2_gmem", "3pass_pss2_gmem", 0.5)
    dg_runtime_heatmap_griddim_blockdim(dataUniform, outdirpath, "runtime_heatmap_dimensioning_3pass_fproc_true", "3pass_fproc_true", 0.5)
    # more diagrams for different datasets
    dg_runtime_heatmap_griddim_blockdim(dataZipf, outdirpath, "zipf_runtime_heatmap_dimensioning_3pass_popc_none", "3pass_popc_none", zipf_p)
    dg_runtime_heatmap_griddim_blockdim(dataZipf, outdirpath, "zipf_runtime_heatmap_dimensioning_3pass_pss_gmem", "3pass_pss_gmem", zipf_p)
    dg_runtime_heatmap_griddim_blockdim(dataZipf, outdirpath, "zipf_runtime_heatmap_dimensioning_3pass_pss2_gmem", "3pass_pss2_gmem", zipf_p)
    dg_runtime_heatmap_griddim_blockdim(dataZipf, outdirpath, "zipf_runtime_heatmap_dimensioning_3pass_fproc_true", "3pass_fproc_true", zipf_p)
    dg_package_throughput_over_chunklength(dataZipf, outdirpath, "zipf_package_throughput_over_chunklength", zipf_p)
    dg_runtime_heatmap_griddim_blockdim(dataBurst, outdirpath, "burst_runtime_heatmap_dimensioning_3pass_popc_none", "3pass_popc_none", burst_p)
    dg_runtime_heatmap_griddim_blockdim(dataBurst, outdirpath, "burst_runtime_heatmap_dimensioning_3pass_pss_gmem", "3pass_pss_gmem", burst_p)
    dg_runtime_heatmap_griddim_blockdim(dataBurst, outdirpath, "burst_runtime_heatmap_dimensioning_3pass_pss2_gmem", "3pass_pss2_gmem", burst_p)
    dg_runtime_heatmap_griddim_blockdim(dataBurst, outdirpath, "burst_runtime_heatmap_dimensioning_3pass_fproc_true", "3pass_fproc_true", burst_p)
    dg_package_throughput_over_chunklength(dataBurst, outdirpath, "burst_package_throughput_over_chunklength", burst_p)
    print("done")

if __name__ == "__main__":
    main()
