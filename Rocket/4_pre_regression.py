import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

design_name = sys.argv[1]
testbench = sys.argv[2]

# input
core_waveform_file = f"./{testbench}/waveform/_TestDriver_testHarness_chiptop0_system_tile_prci_domain_element_reset_domain_boom_tile_core_.csv"
waveform_file_path = f"./{testbench}/waveform"
reg_waveform_file = f"./{testbench}/{design_name}_{testbench}_reg_waveform.csv"
reg_clusters_file = f"./Design/Structure/{design_name}_reg_clusters.csv"  
reg_features_file = f"./Design/Structure/{design_name}_reg_features.csv" 
stable_power_file = f"./Design/stable_power.csv"

# output
reg_clusters_weight_file = f"./{testbench}/{design_name}_{testbench}_reg_clusters_weight.csv"
reg_features_and_weight_pre_file = f"./{testbench}/{design_name}_{testbench}_reg_features_and_weight_pre.csv"

cluster_power_weight = {}

def mapes(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0
    return np.mean(np.abs((y_pred[mask] - y_true[mask])/y_true[mask]))


def savefiles(reg_waveform_df,reg_features_df,reg_to_cluster):

    global cluster_power_weight

    #print(cluster_power_weight)

    clusters = []
    power_weights = []

    reg_waveform_regs = reg_waveform_df.columns

    for reg_name in reg_features_df["reg_name"]:

        if reg_name in reg_waveform_regs:

            cluster = reg_to_cluster[reg_name]
        
            #print(type(cluster))
            #print(cluster_power_weight[cluster])

            clusters.append(cluster)
            power_weights.append(cluster_power_weight[str(cluster)])

        else:
            clusters.append(None)
            power_weights.append(None)

    reg_features_df["Cluster"] = clusters
    reg_features_df["PowerWeight"] = power_weights

    reg_features_and_weight_df = reg_features_df[reg_features_df["reg_name"].isin(reg_waveform_regs)]

    reg_features_and_weight_df.to_csv(reg_features_and_weight_pre_file,index = False)



def regression():

    cpus = set(range(31,46))
    os.sched_setaffinity(0,cpus)

    global cluster_power_weight


    stable_power_df = pd.read_csv(stable_power_file)
    stable_power_dict = dict(zip(stable_power_df["0"],stable_power_df["1"])) 

    reg_clusters_df = pd.read_csv(reg_clusters_file)
    reg_to_cluster = dict(zip(reg_clusters_df['reg_name'], reg_clusters_df['Cluster']))


    df = pd.DataFrame()

    for filename in os.listdir(waveform_file_path):

        file_path = os.path.join(waveform_file_path,filename)

        df_temp = pd.read_csv(file_path)

        df = pd.concat([df,df_temp], ignore_index = True)



    X = df.iloc[:,0:-1].values
    y = df.iloc[:, -1].values

    #X = pd.DataFrame()  
    #y = pd.DataFrame()

    #for col in df.columns:

    #    if col == "Power":
    #        y[col] = df[col]

    #    elif col != "Time": 
    #        X[col] = df[col]


    model = Lasso(alpha=0.1, fit_intercept=False, positive = True)
    model.fit(X, y)


    core_df = pd.read_csv(core_waveform_file)


    XX = core_df.iloc[:,0:-1].values
    yy = core_df.iloc[:, -1].values 
    #XX = pd.DataFrame()  
    #yy = pd.DataFrame()

    #for col in core_df.columns:

    #    if col == "Power":
    #        yy[col] = df[col]

    #    elif col != "Time": 
    #        XX[col] = df[col]

    
    stable_power = stable_power_dict.get("/TestDriver/testHarness/chiptop0/system/tile_prci_domain/element_reset_domain_boom_tile/core/",0)

    y_pred = model.predict(XX) + stable_power
    y_true = yy + stable_power

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    nrmse = rmse / np.mean(y_true) * 100
    mape = mapes(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"NRMSE: {nrmse}")
    print(f"MAPE: {mape}")
    print(f"R^2: {r2}")


    reg_waveform_df = pd.read_csv(reg_waveform_file)
    reg_features_df = pd.read_csv(reg_features_file)

    coefficients = model.coef_
    #print(len(coefficients))
    for idx, cluster in enumerate(df.keys()):
        if idx < len(coefficients):
            cluster_power_weight[cluster] = coefficients[idx]


    savefiles(reg_waveform_df,reg_features_df,reg_to_cluster)






regression()