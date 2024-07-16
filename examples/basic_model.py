import pandas as pd
from segmentae.anomaly_detection import (SegmentAE,
                                         Preprocessing,
                                         Clustering
                                         )
from segmentae.data_sources.examples import load_dataset
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

############################################################################################
### Data Loading

train, test, target = load_dataset(dataset_selection = 'network_intrusions', 
                                   split_ratio = 0.75) # Options | 'network_intrusions', 'default_credit_card', 
                                                       #         | 'htru2_dataset', 'shuttle_148'

test, future_data = train_test_split(test, train_size = 0.9, random_state = 5)

# Resetting Index is Required
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
future_data = future_data.reset_index(drop=True)

X_train = train.drop(columns=[target]).copy() #, train[target].astype(int)
X_test, y_test = test.drop(columns=[target]).copy(), test[target].astype(int)
X_future_data = future_data.drop(columns=[target]).copy() # future_data[target].astype(int)

############################################################################################
### Preprocessing

pr = Preprocessing(encoder = "IFrequencyEncoder",  # Options | "IFrequencyEncoder", "LabelEncoder", "OneHotEncoder", None
                   scaler = "MinMaxScaler",        # Options | "MinMaxScaler", "StandardScaler", "RobustScaler", None
                   imputer = None)                 # Options | "Simple","RandomForest","ExtraTrees","GBR","KNN","XGBoost","Lightgbm","Catboost", None

pr.fit(X = X_train)
X_train = pr.transform(X = X_train)
X_test = pr.transform(X = X_test)
X_future_data = pr.transform(X = X_future_data)

############################################################################################
### Example: Basic Autoencoder Model

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(X_train.shape[1])

# Define the encoder
encoder = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(24, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu')
])

# Define the decoder
decoder = Sequential([
    Dense(16, activation='relu', input_shape=(8,)),
    Dense(24, activation='relu'),
    Dense(32, activation='relu'),
    Dense(X_train.shape[1], activation='sigmoid')
])

# Combine encoder and decoder into an autoencoder model
autoencoder = Sequential([encoder, decoder])

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
autoencoder.fit(X_train, X_train, epochs=50, batch_size=None, validation_split=0.1)

###########################################
#### Clustering Implementation

cl_model = Clustering(cluster_model = ["KMeans"], # Options | "KMeans", "MiniBatchKMeans", "GMM", "Agglomerative", 
                      n_clusters = 3)
cl_model.clustering_fit(X = X_train)

############################################################################################
### Autoencoder + Clustering Integration

sg = SegmentAE(ae_model = autoencoder,
               cl_model = cl_model)

### Train Reconstruction

sg.reconstruction(input_data = X_train,
                  threshold_metric = 'mse')

### Reconstruction Error Result Performance

results = sg.evaluation(input_data = X_test,
                        target_col = y_test,
                        threshold_ratio = 2.0)

preds_test, recon_metrics_test = sg.preds_test, sg.reconstruction_test # Test Metadata by Cluster

############################################################################################
### Multiple Threshold Ratio Reconstruction Evaluation

threshold_ratios = [ 0.75, 1, 1.5, 2, 3, 4]

global_results = pd.concat([sg.evaluation(input_data = X_test,
                                          target_col = y_test,
                                          threshold_ratio = thr)["global metrics"]
                            for thr in threshold_ratios])

############################################################################################
### Anomaly Detection Predictions

best_ratio = global_results.sort_values(by="Accuracy", ascending=False).iloc[0]["Threshold Ratio"]

predictions = sg.detections(input_data = X_future_data,
                            threshold_ratio = best_ratio)
