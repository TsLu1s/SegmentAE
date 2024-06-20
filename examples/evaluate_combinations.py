import pandas as pd
from segmentae.anomaly_detection import (SegmentAE,
                                         Preprocessing,
                                         Clustering,
                                         DenseAutoencoder,
                                         BatchNormAutoencoder)
from segmentae.data_sources.examples import load_dataset
from itertools import product
import tensorflow as tf

############################################################################################
### Data Loading

train, test, target = load_dataset(dataset_selection = 'german_credit_card', 
                                   split_ratio = 0.75) # Options | 'german_credit_card', 'network_intrusions', 'default_credit_card'                  

X_train = train.drop(columns=[target]).copy()
X_test = test.drop(columns=[target]).copy()

#y_train = train[target].astype(int)
y_test = test[target].astype(int)

############################################################################################
### Preprocessing

pr = Preprocessing(encoder = None,          # Options | "IFrequencyEncoder", "LabelEncoder", "OneHotEncoder", None
                   scaler = "MinMaxScaler", # Options | "MinMaxScaler", "StandardScaler", "RobustScaler", None
                   imputer = None)          # Options | "Simple","RandomForest","ExtraTrees","GBR","KNN","XGBoost","Lightgbm","Catboost", None

pr.fit(X = X_train)
X_train = pr.transform(X = X_train)
X_test = pr.transform(X = X_test)

############################################################################################
### Performance Evaluation - Detection of best Autoencoder + Clustering Combination

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

list_metrics, n = [], 1
for n_clusters, cluster, autoenc in product([1,2,3,4],
                                            ["KMeans", "MiniBatchKMeans", "GMM"],  # Options | KMeans, MiniBatchKMeans, GMM, Agglomerative
                                            ["DenseAutoencoder", "BatchNormAutoencoder"]):
    print("iteration", n)
    print("Cluster Model: ", cluster)
    print("Number of Clusters: ", n_clusters)
    print("Autoencoder: ", autoenc)
    print("")
    
    ### Autoencoder Implementation
    
    _epochs = 100
    
    if autoenc== "DenseAutoencoder":
        autoencoder = DenseAutoencoder(hidden_dims = [16, 12, 8, 4],
                                       encoder_activation = 'relu',
                                       decoder_activation = 'relu',
                                       optimizer = 'adam',
                                       learning_rate = 0.001,
                                       epochs = _epochs,
                                       val_size = 0.1,
                                       stopping_patient = 10,
                                       dropout_rate = 0.1,
                                       batch_size = None)
        autoencoder.fit(input_data = X_train)
        
    elif autoenc== "BatchNormAutoencoder":
        autoencoder = BatchNormAutoencoder(hidden_dims = [32, 16, 8],
                                           encoder_activation = 'relu',  
                                           decoder_activation = 'relu',  
                                           optimizer = 'adam',
                                           learning_rate = 0.001,
                                           epochs = _epochs,
                                           val_size  = 0.1,
                                           stopping_patient = 10,
                                           dropout_rate = 0.1,
                                           batch_size = None)
        autoencoder.fit(input_data = X_train)
        
    ### Clustering Implementation
    
    cl_model = Clustering(cluster_model = [cluster], 
                          n_clusters = n_clusters)
    cl_model.clustering_fit(X = X_train)
    
    ### Autoencoder + Clustering Integration
    
    sg = SegmentAE(ae_model = autoencoder,
                   cl_model = cl_model)
    
    ### Train Reconstruction
    
    sg.reconstruction(input_data = X_train,
                      threshold_metric = 'mse')
    
    ### Multiple Threshold Ratio Reconstruction Evaluation
    
    threshold_ratios = [ 0.75, 1, 1.5, 2, 3, 4]
    
    global_metrics_ae = pd.concat([sg.evaluation(input_data = X_test,
                                                 target_col = y_test, 
                                                 threshold_ratio = thr)["global metrics"]
                                   for thr in threshold_ratios])
    global_metrics_ae["Autoencoder"] = autoenc
    global_metrics_ae["Cluster"] = cluster
    global_metrics_ae["N_Clusters"] = n_clusters
    
    list_metrics.append(global_metrics_ae)
    n+=1
global_results = pd.concat(list_metrics,axis=0) ### All configurations performance




