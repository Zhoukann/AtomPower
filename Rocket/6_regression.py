import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

design_name = sys.argv[1]
testbench = sys.argv[2]

# input
reg_waveform_file = f"./{testbench}/{design_name}_{testbench}_reg_waveform.csv"
power_waveform_file = f"./{testbench}/{design_name}_{testbench}_power_waveform.csv"
reg_features_and_weight_pre_combine_file = f"./{testbench}/{design_name}_{testbench}_reg_features_and_weight_pre.csv"
stable_power_file = f"./Design/stable_power.csv"

# output
reg_features_and_weight_file = f"./{testbench}/{design_name}_{testbench}_reg_features_and_weight.csv"

def mapes(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0
    return np.mean(np.abs((y_pred[mask] - y_true[mask])/y_true[mask]))


def custom_linear_regression_tf(learning_rate=0.001, epochs=500, alpha=0.01):

    cpus = set(range(31,46))
    os.sched_setaffinity(0,cpus)

    stable_power_df = pd.read_csv(stable_power_file)
    stable_power_dict = dict(zip(stable_power_df["0"],stable_power_df["1"]))
    
    stable_power = stable_power_dict.get("/TestDriver/testHarness/chiptop0/system/tile_prci_domain/element_reset_domain_boom_tile/core/",0)

    df_reg_waveform = pd.read_csv(reg_waveform_file)
    df_power_waveform = pd.read_csv(power_waveform_file)
    df_weights = pd.read_csv(reg_features_and_weight_pre_combine_file)

    X = df_reg_waveform.iloc[:, 1:].values  
    y = df_power_waveform.iloc[:, 1].values  

    reg_names = df_weights['reg_name'].values
    initial_weights = df_weights['PowerWeight'].values
    tree_feature_node_num = df_weights['tree_feature_node_num'].values
    tree_feature_intree_node_num = df_weights['tree_feature_intree_node_num'].values

    reg_name_to_weight = dict(zip(reg_names, initial_weights))

    feature_adjustment_factors1 = np.sqrt(np.max(tree_feature_node_num) / (tree_feature_node_num + 1))
    feature_adjustment_factors2 = np.sqrt(np.max(tree_feature_intree_node_num) / (tree_feature_intree_node_num + 0.01))
    feature_adjustment_factors = feature_adjustment_factors1 + feature_adjustment_factors2

    num_registers = X.shape[1]
    adjusted_weights = np.zeros(num_registers)

    for i in range(num_registers):
        reg_name = df_reg_waveform.columns[i + 1]  
        if reg_name in reg_name_to_weight:
            adjusted_weights[i] = reg_name_to_weight[reg_name] 
        else:
            adjusted_weights[i] = 0 

    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

    weights = tf.Variable(adjusted_weights, dtype=tf.float32)

    y_pred = tf.reduce_sum(X_tensor * weights, axis=1)

    rmse = np.sqrt(mean_squared_error(y_tensor + stable_power, y_pred + stable_power))
    mae = mean_absolute_error(y_tensor + stable_power, y_pred + stable_power)

    nrmse = rmse / np.mean(y_tensor + stable_power) * 100
    mape = mapes(y_tensor + stable_power, y_pred + stable_power) * 100
    r2 = r2_score(y_tensor + stable_power, y_pred + stable_power)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"NRMSE: {nrmse}")
    print(f"MAPE: {mape}")
    print(f"R^2: {r2}")
    
    def custom_loss(y_true, y_pred, weights, feature_adjustment_factors, alpha):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        #feature_adjustment_factor_sum = tf.reduce_sum(feature_adjustment_factors * weights)  
        feature_adjustment_factor_sum = tf.reduce_sum(feature_adjustment_factors * tf.square(weights))
        l2_reg = alpha * feature_adjustment_factor_sum  
        return mse + l2_reg

    optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:

            y_pred = tf.reduce_sum(X_tensor * weights, axis=1)

            loss = custom_loss(y_tensor, y_pred, weights, feature_adjustment_factors, alpha)
        
        gradients = tape.gradient(loss, [weights])
        optimizer.apply_gradients(zip(gradients, [weights]))

        weights.assign(tf.maximum(weights, 0))

        if (epoch + 1) % 50 == 0:

            rmse = np.sqrt(mean_squared_error(y_tensor + stable_power, y_pred + stable_power))
            mae = mean_absolute_error(y_tensor + stable_power, y_pred + stable_power)

            nrmse = rmse / np.mean(y_tensor + stable_power) * 100
            mape = mapes(y_tensor + stable_power, y_pred + stable_power) * 100
            r2 = r2_score(y_tensor + stable_power, y_pred + stable_power)

            print(f"Epoch {epoch + 1}/{epochs}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, NRMSE: {nrmse:.2f}, MAPE:{mape:.2f}, R^2: {r2:.3f}")

    updated_weights = weights.numpy()

    reg_name_to_updated_weight = dict(zip(df_reg_waveform.columns[1:], updated_weights)) 

    df_weights['PowerWeight'] = df_weights['reg_name'].map(reg_name_to_updated_weight)

    df_weights_sorted = df_weights.sort_values(by='reg_name').reset_index(drop=True)
    df_weights_sorted.to_csv(reg_features_and_weight_file, index=False)

    print(f"Training complete, results saved to {reg_features_and_weight_file}")
    return updated_weights

optimal_weights = custom_linear_regression_tf()
