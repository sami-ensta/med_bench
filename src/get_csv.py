"""
    Benchmark comparing the performances of the following estimators :
    - coefficient_product
    - med_dml
    - multiply_robust
    
    Except for coefficient_product, those estimators are parametrized with different setups :
    - regularization
    - calibration
    - forest
    - crossfitting
    
    Saves the results in a /results/simulations/.csv table
"""

import time
import itertools
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from rpy2.rinterface_lib.embedded import RRuntimeError
import pandas as pd
import pytest
from pathlib import Path
import seaborn as sns
from med_bench.src.get_simulated_data import simulate_data
from med_bench.src.get_estimation import get_estimation

from med_bench.src.benchmark_mediation import (
    ols_mediation,
    med_dml,
    medDML,
    multiply_robust_efficient,
)


exp_path = "results/simulations/20230725_simulations"

dim_m_list = [1, 1, 5] * 3
m_type_list = ["binary", "continuous", "continuous"] * 3
wt_list = [0.5, 0.5, 0.2] + [2, 0.8, 0.3] + [4, 2, 1]
wm_list = [0.5, 0.5, 0.2] + [10, 5, 5] + [2, 1, 1]
y_m_setting = [[False, False], [True, False], [False, True], [True, True]]
n_sizes = [500, 1000, 10000]
nb_rep = 50
dim_x = 5

cols = [
    "n",
    "dim_x",
    "dim_m",
    "type_m",
    "wt_list",
    "wm_list",
    "m_misspec",
    "y_misspec",
    "mediated_prop",
    "total",
    "direct_1",
    "direct_0",
    "indirect_1",
    "indirect_0",
]
res_list = list()

sim_idx = 1
for n in n_sizes:
    for nrep in range(nb_rep):
        for dim_setting in range(9):
            for y_m_set in y_m_setting:
                sim_idx += 1
                rg = default_rng(int(sim_idx * n / 100 * (dim_setting + 1)))
                m_set, y_set = y_m_set

                foldername = "{}/rep{}_n{}_setting{}_misspecM{}_misspecY{}".format(
                    exp_path, nrep, n, dim_setting, m_set, y_set
                )
                Path(foldername).mkdir(parents=True, exist_ok=True)
                (
                    x,
                    t,
                    m,
                    y,
                    total,
                    theta_1,
                    theta_0,
                    delta_1,
                    delta_0,
                    p_t,
                    th_p_t_mx,
                ) = simulate_data(
                    n,
                    rg,
                    mis_spec_m=m_set,
                    mis_spec_y=y_set,
                    dim_x=dim_x,
                    dim_m=dim_m_list[dim_setting],
                    seed=5,
                    type_m=m_type_list[dim_setting],
                    sigma_y=0.5,
                    sigma_m=0.5,
                    beta_t_factor=wt_list[dim_setting],
                    beta_m_factor=wm_list[dim_setting],
                )
                param_list = [
                    n,
                    5,
                    dim_m_list[dim_setting],
                    m_type_list[dim_setting],
                    wt_list[dim_setting],
                    wm_list[dim_setting],
                    m_set,
                    y_set,
                    delta_1 / total,
                    total,
                    theta_1,
                    theta_0,
                    delta_1,
                    delta_0,
                ]
                if (n == 500) and (nrep == 0):
                    # making some summary thing to have an overview of all settings
                    print(len(res_list))
                    res_list.append(param_list)
                    sns.distplot(th_p_t_mx[t.ravel() == 0])
                    sns.distplot(th_p_t_mx[t.ravel() == 1])
                    plt.savefig(
                        "{}/{}_overlap_t_m_x.pdf".format(exp_path, len(res_list))
                    )
                    plt.close()

                data_cols = (
                    ["x_{}".format(i) for i in range(dim_x)]
                    + ["t", "y"]
                    + ["m_{}".format(i) for i in range(dim_m_list[dim_setting])]
                )
                data_df = pd.DataFrame(np.hstack((x, t, y, m)), columns=data_cols)
                data_df.to_csv("{}/data.csv".format(foldername), index=False)
                param_df = pd.DataFrame([param_list], columns=cols)
                param_df.to_csv("{}/param.csv".format(foldername), index=False)

res_df = pd.DataFrame(res_list, columns=cols)
res_df.to_csv("{}/simulation_description.csv".format(exp_path), index=False)


estimator_list = [
    "coefficient_product",
    "DML_huber",
    "med_dml_noreg",
    "med_dml_reg",
    "med_dml_reg_not_normalized",
    "med_dml_reg_calibration",
    "med_dml_reg_forest",
    "med_dml_reg_forest_calibration",
    "med_dml_noreg_cf",
    "med_dml_reg_cf",
    "med_dml_reg_calibration_cf",
    "med_dml_reg_forest_cf",
    "med_dml_reg_forest_calibration_cf",
    "multiply_robust_noreg",
    "multiply_robust_reg",
    "multiply_robust_reg_calibration",
    "multiply_robust_forest",
    "multiply_robust_forest_calibration",
    "multiply_robust_noreg_cf",
    "multiply_robust_reg_cf",
    "multiply_robust_reg_calibration_cf",
    "multiply_robust_forest_cf",
    "multiply_robust_forest_calibration_cf",
]

estimator_list = [
    "DML_huber",
    "med_dml_noreg",
    "med_dml_reg",
    "med_dml_reg_not_normalized",
    "med_dml_reg_calibration",
    "med_dml_reg_forest",
    "med_dml_reg_forest_calibration",
    "med_dml_noreg_cf",
    "med_dml_reg_cf",
    "med_dml_reg_calibration_cf",
    "med_dml_reg_forest_cf",
    "med_dml_reg_forest_calibration_cf",
]


# folderpath = sys.argv[1]
folderpath = "results/simulations/20230725_simulations/rep0_n500_setting8_misspecMFalse_misspecYFalse"


PARAM_FOLDER = list(
    itertools.product(
        np.arange(50),
        [500, 1000],  # [500, 1000, 10000],
        np.arange(9),
        [False, True],
        [False, True],
    )
)

for param in PARAM_FOLDER:
    folderpath = f"results/simulations/20230725_simulations/rep{param[0]}_n{param[1]}_setting{param[2]}_misspecM{param[3]}_misspecY{param[4]}"

    data_df = pd.read_csv("{}/data.csv".format(folderpath))
    y = data_df.y.values
    t = data_df["t"].values
    x_cols = [c for c in data_df.columns if "x" in c]
    m_cols = [c for c in data_df.columns if "m" in c]
    x = data_df[x_cols].values
    m = data_df[m_cols].values

    param_df = pd.read_csv("{}/param.csv".format(folderpath))
    out_cols = param_df.columns
    base_list = list(param_df.values[0])

    if len(np.unique(m)) == 2:
        config = 1
    else:
        config = 5

    res_list = list()
    for estimator in estimator_list:
        val_list = list(base_list)
        val_list.append(estimator)
        start = time.time()
        try:
            effects = get_estimation(x, t.ravel(), m, y.ravel(), estimator, config)
        except RRuntimeError:
            effects = [np.nan * 6]
        except ValueError:
            effects = [np.nan * 6]
        duration = time.time() - start
        val_list += list(effects)
        val_list.append(duration)
        res_list.append(val_list)

    final_cols = list(out_cols) + [
        "estimator",
        "total_effect",
        "direct_treated_effect",
        "direct_control_effect",
        "indirect_treated_effect",
        "indirect_control_effect",
        "n_non_trimmed",
        "duration",
    ]

    res_df = pd.DataFrame(res_list, columns=final_cols)
    res_df.to_csv("{}/estimation_results.csv".format(folderpath), index=False, sep="\t")


tables = list()
non_working = list()

for index, param in enumerate(PARAM_FOLDER):
    folderpath = f"results/simulations/20230725_simulations/rep{param[0]}_n{param[1]}_setting{param[2]}_misspecM{param[3]}_misspecY{param[4]}"

    filename = "{}/estimation_results.csv".format(folderpath)
    try:
        timestamp = os.path.getmtime(filename)
        df = pd.read_csv(filename, sep="\t")
        df = df.assign(timestamp=timestamp)
        tables.append(df)
        # print(filename, 'ok')
    except FileNotFoundError:
        print(index, filename)
        non_working.append(index + 1)

big_table2 = pd.concat(tables, axis=0)
big_table2.to_csv("results/20230725_big_table_basic_simu.csv", sep="\t", index=False)
