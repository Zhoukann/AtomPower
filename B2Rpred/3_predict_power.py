import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time


design_name = sys.argv[1]
testbench = sys.argv[2]


def mapes(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0
    return np.mean(np.abs((y_pred[mask] - y_true[mask])/y_true[mask]))


def predict_power():

    start_time = time.time()

    cpus = set(range(31,46))
    os.sched_setaffinity(0,cpus)

    strip_path = "/TestDriver/testHarness/chiptop0/system/tile_prci_domain/element_reset_domain_rockettile"

    reg_waveform_file = f"../{design_name}/{testbench}/{design_name}_{testbench}_reg_waveform.csv"
    power_waveform_file = f"../{design_name}/{testbench}/{design_name}_{testbench}_power_waveform.csv"
    reg_features_and_weight_file = f"./{design_name}/{design_name}_reg_features_and_weight_predict.csv"
    stable_power_file = f"../{design_name}/Design/stable_power.csv"


    # output
    output_path = f"./{design_name}"
    output_file = f"{output_path}/{testbench}_predicted_power.csv"
    os.makedirs(output_path, exist_ok = True)

    module = f"{strip_path}/core/"

    stable_power_df = pd.read_csv(stable_power_file)
    stable_power_dict = dict(zip(stable_power_df["0"],stable_power_df["1"]))
    
    # stable_power
    stable_power = stable_power_dict.get(module,0)

    df_reg_waveform = pd.read_csv(reg_waveform_file)
    df_power_waveform = pd.read_csv(power_waveform_file)
    df_weights = pd.read_csv(reg_features_and_weight_file)

    # reg waveform
    reg_inmodule = [reg for reg in df_reg_waveform.columns if module in reg] 
    reg_inmodule_waveform = df_reg_waveform[reg_inmodule]


    reg_names = df_weights['reg_name'].values
    power_weights = df_weights['PowerWeight'].values
    weights_dict = dict(zip(reg_names, power_weights))


    X = reg_inmodule_waveform.values
    y_true = df_power_waveform[module].values

    num_registers = X.shape[1]
    adjusted_weights = np.zeros(num_registers)
    for i in range(num_registers):
        reg_name = reg_inmodule_waveform.columns[i]
        if reg_name in weights_dict:
            adjusted_weights[i] = weights_dict[reg_name]
        else:
            adjusted_weights[i] = 0

    # Predicted power
    y_pred = np.dot(X, adjusted_weights)

    end_time = time.time()

    rmse = np.sqrt(mean_squared_error(y_true + stable_power, y_pred + stable_power))
    mae = mean_absolute_error(y_true + stable_power, y_pred + stable_power)

    nrmse = rmse / np.mean(y_true + stable_power) * 100
    mape = mapes(y_true + stable_power, y_pred + stable_power) * 100

    #print(f"RMSE: {rmse}")
    #print(f"MAE: {mae}")
    print(f"NRMSE: {nrmse}")
    print(f"MAPE: {mape}")

    run_time = end_time - start_time

    print(f"Time: {run_time} Seconds")

    df_output = pd.DataFrame({
        'Time': df_reg_waveform['Time'],
        'ActualPower': y_true + stable_power,
        'PredictedPower': y_pred + stable_power,
    })
    df_output.to_csv(output_file, index=False)

predict_power()
