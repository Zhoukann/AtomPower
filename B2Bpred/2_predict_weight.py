import os
import sys
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


design_name = sys.argv[1]
#model_name = sys.argv[2]

save_dir = f"./"


def predict(design_name):
    
    cpus = set(range(31,46))
    os.sched_setaffinity(0,cpus)

    features_file = f"../{design_name}/Design/Structure/{design_name}_reg_features.csv"

    # output
    output_path = f"./{design_name}"
    output_file = f"{output_path}/{design_name}_reg_features_and_weight_predict.csv"
    os.makedirs(output_path, exist_ok = True)
    
    data = pd.read_csv(features_file)

    features = data.iloc[:,1:].values

    scaler = joblib.load(os.path.join(save_dir,'power_model_scaler.pkl'))
    model = tf.keras.models.load_model(os.path.join(save_dir,'power_model.h5'))
    
    features_scaled = scaler.fit_transform(features)

    predictions = model.predict(features_scaled)

    data['PowerWeight'] = predictions

    data.to_csv(output_file, index = False)


def main():
    
    predict(design_name)


main()
