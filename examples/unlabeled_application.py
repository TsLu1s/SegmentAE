from segmentae.anomaly_detection import (SegmentAE,
                                         Preprocessing,
                                         Clustering,
                                         DenseAutoencoder,
                                         #BatchNormAutoencoder,
                                         )
from segmentae.data_sources.examples import load_dataset
import tensorflow as tf

############################################################################################
### Data Loading

train, future_data, target = load_dataset(dataset_selection = 'network_intrusions', 
                                          split_ratio = 0.75) # Options | 'german_credit_card', 'network_intrusions', 'default_credit_card'            

train, future_data = train.reset_index(drop=True), future_data.reset_index(drop=True)

X_train = train.drop(columns=[target]).copy()
X_future_data = future_data.drop(columns=[target]).copy()

#y_train = train[target].astype(int)
#y_future_data = future_data[target].astype(int)

############################################################################################
### Preprocessing

pr = Preprocessing(encoder = "IFrequencyEncoder",  # Options | "IFrequencyEncoder", "LabelEncoder", "OneHotEncoder", None
                   scaler = "MinMaxScaler",        # Options | "MinMaxScaler", "StandardScaler", "RobustScaler", None
                   imputer = None)                 # Options | "Simple","RandomForest","ExtraTrees","GBR","KNN","XGBoost","Lightgbm","Catboost", None

pr.fit(X = X_train)
X_train = pr.transform(X = X_train)
X_future_data = pr.transform(X = X_future_data)

############################################################################################

### Clustering Implementation

cl_model = Clustering(cluster_model = ["GMM"], # Options | "KMeans", "MiniBatchKMeans", "GMM", "Agglomerative", 
                      n_clusters = 3)
cl_model.clustering_fit(X = X_train)

### Autoencoder Implementation

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# "DenseAutoencoder"
denseAutoencoder = DenseAutoencoder(hidden_dims = [16, 12, 8, 4],
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

### Train Reconstruction Evaluation

sg.reconstruction(input_data = X_train,    
                  threshold_metric = 'mse')  #  Options | mse, mae, rmse, max_error

preds_train, recon_metrics_train =  sg.preds_train, sg.reconstruction_eval # Reconstruction Metadata by Cluster

############################################################################################
### Anomaly Detection Predictions

predictions = sg.detections(input_data = X_future_data,
                            threshold_ratio = 2)
