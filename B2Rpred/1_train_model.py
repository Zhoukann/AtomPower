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

#model_name = sys.argv[1]
train_epochs = sys.argv[1]

filelist_test = f"filelist2.f" 
filelist = f"filelist.f" 
save_dir = f"./"

def load_data(filelist):
    with open(filelist,'r') as f:
        files = f.readlines()

    all_data = []
    for feature_file in files:
        feature_file = feature_file.strip()
        data = pd.read_csv(feature_file)
        all_data.append(data)

    return pd.concat(all_data, ignore_index = True)


def train_model(filelist):

    cpus = set(range(31,46))
    os.sched_setaffinity(0,cpus)

    
    data = load_data(filelist)

    features = data.iloc[:,1:-2].values
    labels = data.iloc[:,-1].values

    scaler = StandardScaler()
    
    features_scaled = scaler.fit_transform(features)

    X_train = features_scaled
    y_train = labels


    data = load_data(filelist_test)

    features = data.iloc[:,1:-2].values
    labels = data.iloc[:,-1].values

    features_scaled = scaler.fit_transform(features)

    X_test = features_scaled
    y_test = labels


    model = Sequential([
        Dense(128, input_shape = (X_train.shape[1],), activation = 'relu'),
        Dense(64, activation = 'relu'),
        Dense(1, activation = 'linear')
    ])
    # 128, 64, 1 

    model.compile(
        optimizer = Adam(learning_rate = 0.005),
        loss = 'mse',
    )

    model.summary()

    history = model.fit(X_train, y_train, epochs = int(train_epochs), batch_size = 256, validation_data = (X_test, y_test))


    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"RMSE:{rmse}")
    print(f"MAE:{mae}")


    joblib.dump(scaler, os.path.join(save_dir,'power_model_scaler.pkl'))
    model.save(os.path.join(save_dir,'power_model.h5'))


train_model(filelist)
