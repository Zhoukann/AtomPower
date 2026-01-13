import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time
design_name = sys.argv[1]
testbench = sys.argv[2]
#model_name = sys.argv[3]

# output
mape_file = f"./{design_name}_mape.csv"
ave_file = f"./{design_name}_ave.csv"
r_file = f"./{design_name}_r.csv"
pred_file = f"./{design_name}_pred.csv"
true_file = f"./{design_name}_true.csv"

def mapes(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0
    return np.mean(np.abs((y_pred[mask] - y_true[mask])/y_true[mask]))

def aves(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.abs((np.mean(y_pred) - np.mean(y_true))/np.mean(y_true))


def update_csv(file_path,module,value):

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col = 0)
    else:
        df = pd.DataFrame(columns = [testbench])

    if module not in df.index:
        df.loc[module] = [None] * len(df.columns)
    
    if testbench not in df.columns:
        df[testbench] = None

    df.loc[module, testbench] = value

    df.to_csv(file_path)


def predict_power(stable_power_df, df_reg_waveform, df_power_waveform, df_weights, module):

    start_time = time.time()

    stable_power_dict = dict(zip(stable_power_df["0"],stable_power_df["1"]))
    
    # stable_power
    stable_power = stable_power_dict.get(module,0)

    # reg waveform
    reg_inmodule = [reg for reg in df_reg_waveform.columns if module in reg] 
    reg_inmodule_waveform = df_reg_waveform[reg_inmodule]


    reg_names = df_weights['reg_name'].values
    power_weights = df_weights['PowerWeight'].values
    weights_dict = dict(zip(reg_names, power_weights))

    print(module)

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

    #rmse = np.sqrt(mean_squared_error(y_true + stable_power, y_pred + stable_power))
    mae = mean_absolute_error(y_true + stable_power, y_pred + stable_power)

    #nrmse = rmse / np.mean(y_true + stable_power) * 100
    mape = mapes(y_true + stable_power, y_pred + stable_power) * 100######
    ave = aves(y_true + stable_power, y_pred + stable_power) * 100######

    y_pred_mean = np.mean(y_pred + stable_power)
    y_true_mean = np.mean(y_true + stable_power)

    #print(f"RMSE: {rmse}")
    #print(f"MAE: {mae}")
    #print(f"NRMSE: {nrmse}")
    print(f"MAPE: {mape}")
    print(f"AVE: {ave}")

    run_time = end_time - start_time
    print(run_time)


    #df_output = pd.DataFrame({
    #    'Time': df_reg_waveform['Time'],
    #    'ActualPower': y_true,
    #    'PredictedPower': y_pred,
    #})
    #df_output.to_csv(output_file, index=False)

    update_csv(mape_file,module,mape)
    update_csv(ave_file,module,ave)
    update_csv(pred_file,module,y_pred_mean)
    update_csv(true_file,module,y_true_mean)


def main():

    cpus = set(range(31,46))
    os.sched_setaffinity(0,cpus)

    # input
    reg_waveform_file = f"../{design_name}/{testbench}/{design_name}_{testbench}_reg_waveform.csv"
    power_waveform_file = f"../{design_name}/{testbench}/{design_name}_{testbench}_power_waveform.csv"
    reg_features_and_weight_file = f"./{design_name}/{design_name}_reg_features_and_weight_predict.csv"
    stable_power_file = f"../{design_name}/Design/stable_power.csv"

    stable_power_df = pd.read_csv(stable_power_file)

    df_reg_waveform = pd.read_csv(reg_waveform_file)
    df_power_waveform = pd.read_csv(power_waveform_file)
    df_weights = pd.read_csv(reg_features_and_weight_file)


    for module in stable_power_df["0"]:

        predict_power(stable_power_df, df_reg_waveform, df_power_waveform, df_weights, module)


main()
