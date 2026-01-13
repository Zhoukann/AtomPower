# Calculate per-cycle power dissipation
import re
import os
import sys
import numpy as np
import pandas as pd
from sympy import Symbol
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

design_name = sys.argv[1]
testbench = sys.argv[2]

# input
reg_clusters_file = f"./Design/Structure/{design_name}_reg_clusters.csv"
reg_waveform_file = f"./{testbench}/{design_name}_{testbench}_reg_waveform.csv"
power_waveform_file = f"./{testbench}/{design_name}_{testbench}_power_waveform.csv"



def aggregate_registers_by_cluster(reg_to_cluster, reg_inmodule_waveform):
    
    clusters = set(reg_to_cluster.values())

    nrows = reg_inmodule_waveform.shape[0]

    cluster_reg_inmodule_waveform = pd.DataFrame(0, index = range(nrows), columns = clusters)

    for reg in reg_inmodule_waveform.columns:

        cluster = reg_to_cluster.get(reg)

        #print(cluster)

        cluster_reg_inmodule_waveform[cluster] += reg_inmodule_waveform[reg]

    #cluster_reg_inmodule_waveform = cluster_reg_inmodule_waveform.loc[:, (cluster_reg_inmodule_waveform != 0).any(axis=0)]
    return cluster_reg_inmodule_waveform


def cluster_waveform(df, k = 500):

    X = df.iloc[:,:-1]

    #scaler = MinMaxScaler()
    #X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters = k, random_state = 2025)
    df["cluster"] = kmeans.fit_predict(X)

    cluster_centers = []

    for cluster_id in np.unique(df["cluster"]):

        cluster_df = df[df["cluster"] == cluster_id]

        cluster_waveforms = cluster_df.iloc[:,:-1].values

        cluster_center = kmeans.cluster_centers_[cluster_id]

        cluster_power = cluster_df.iloc[:,-2].mean()
        #print(cluster_power)

        cluster_centers.append(np.append(cluster_center,cluster_power))

    df1 = pd.DataFrame(cluster_centers,columns = df.columns[:-1])
    df1["Power"] = [row[-1] for row in cluster_centers]

    return df1




def waveform_process():
    
    cpus = set(range(31,46))
    os.sched_setaffinity(0,cpus)

    reg_waveform_df = pd.read_csv(reg_waveform_file)
    power_waveform_df = pd.read_csv(power_waveform_file)

    reg_clusters_df = pd.read_csv(reg_clusters_file)
    reg_to_cluster = dict(zip(reg_clusters_df['reg_name'], reg_clusters_df['Cluster']))


    modules = power_waveform_df.columns[1:] 

    for module in modules:

        module_tofile = module.replace("/","_")

        # output
        waveform_file = f"./{testbench}/waveform/{module_tofile}.csv"

        reg_inmodule = [reg for reg in reg_waveform_df.columns if module in reg]        
        
        print(module)

        if len(reg_inmodule) < 10:
            continue

        reg_inmodule_waveform = reg_waveform_df[reg_inmodule]
        cluster_reg_inmodule_waveform = aggregate_registers_by_cluster(reg_to_cluster, reg_inmodule_waveform)


        #cluster_reg_inmodule_waveform["Time"] = power_waveform_df["Time"]
        cluster_reg_inmodule_waveform["Power"] = power_waveform_df[module]

        df = cluster_reg_inmodule_waveform
        #df = cluster_waveform(cluster_reg_inmodule_waveform)

        df.to_csv(waveform_file,index = False)


waveform_process()  
                  
