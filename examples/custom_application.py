import pandas as pd
from segmentae.data_sources.examples import load_dataset
from segmentae.anomaly_detection import (SegmentAE,
                                         Preprocessing,
                                         Clustering,
                                         DenseAutoencoder,
                                         #BatchNormAutoencoder, 
                                         )
from sklearn.model_selection import train_test_split
import tensorflow as tf

############################################################################################
### Data Loading

train, test, target = load_dataset(dataset_selection = 'german_credit_card', 
                                   split_ratio = 0.75) # Options | 'german_credit_card', 'network_intrusions', 'default_credit_card'              

test, future_data = train_test_split(test, train_size = 0.9, random_state = 5)

# Resetting Index is Required
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
future_data = future_data.reset_index(drop=True)

X_train = train.drop(columns=[target]).copy() # X_train[target].astype(int)
X_test, y_test = test.drop(columns=[target]).copy(), test[target].astype(int)
X_future_data = future_data.drop(columns=[target]).copy() # future_data[target].astype(int)

############################################################################################
### Preprocessing

pr = Preprocessing(encoder = None,          # Options | "IFrequencyEncoder", "LabelEncoder", "OneHotEncoder", None
                   scaler = "MinMaxScaler", # Options | "MinMaxScaler", "StandardScaler", "RobustScaler", None
                   imputer = None)          # Options | "Simple","RandomForest","ExtraTrees","GBR","KNN","XGBoost","Lightgbm","Catboost", None

pr.fit(X = X_train)
X_train = pr.transform(X = X_train)
X_test = pr.transform(X = X_test)
X_future_data = pr.transform(X = X_future_data)

############################################################################################

### Clustering Implementation

cl_model = Clustering(cluster_model = ["KMeans"], # Options | KMeans, MiniBatchKMeans, GMM, Agglomerative
                      n_clusters = 3)
cl_model.clustering_fit(X = X_train)

### Autoencoder Implementation

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# "DenseAutoencoder"
denseAutoencoder = DenseAutoencoder(hidden_dims = [16, 8, 4],
                                    encoder_activation = 'relu',
                                    decoder_activation = 'relu',
                                    optimizer = 'adam',
                                    learning_rate = 0.001,
                                    epochs = 150,
                                    val_size = 0.15,
                                    stopping_patient = 10,
                                    dropout_rate = 0.1,
                                    batch_size = None)
denseAutoencoder.fit(input_data = X_train)

denseAutoencoder.summary() # Model Summary
denseAutoencoder.plot_training_loss() # Training and Validation Loss Plot

"""
# "BatchNormAutoencoder"
batchAutoencoder = BatchNormAutoencoder(hidden_dims = [32, 16, 8],
                                        encoder_activation = 'relu',
                                        decoder_activation = 'relu',
                                        optimizer = 'adam',
                                        learning_rate = 0.001,
                                        epochs = 150,
                                        val_size  = 0.15,
                                        stopping_patient = 10,
                                        dropout_rate = 0.1,
                                        batch_size = None)
batchAutoencoder.fit(input_data=X_train)
"""

############################################################################################
### Autoencoder + Clustering Integration

sg = SegmentAE(ae_model = denseAutoencoder, # batchAutoencoder
               cl_model = cl_model)

### Train Reconstruction

sg.reconstruction(input_data = X_train,
                  threshold_metric = 'mse')  #  Options | mse, mae, rmse, max_error

### Reconstruction Error Result Performance

results = sg.evaluation(input_data = X_test,
                        target_col = y_test,
                        threshold_ratio = 2.0)

preds_test, recon_metrics_test = sg.preds_test, sg.reconstruction_test # Test Metadata by Cluster

############################################################################################
### Multiple Threshold Ratio Reconstruction Evaluation

threshold_ratios=[ 0.75, 1, 1.5, 2, 3, 4]

global_results = pd.concat([sg.evaluation(input_data = X_test,
                                          target_col = y_test,
                                          threshold_ratio = thr)["global metrics"]
                            for thr in threshold_ratios])

############################################################################################
### Anomaly Detection Predictions

best_ratio = global_results['Threshold Ratio'].iloc[0]

predictions = sg.detections(input_data = X_future_data,
                            threshold_ratio = best_ratio)

# from SegmentAE.anomaly_detection import metrics_classification
# pred_final_metrics = metrics_classification(future_data[target].astype(int), predictions['Predicted Anomalies'])
