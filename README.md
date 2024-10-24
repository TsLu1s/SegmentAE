[![LinkedIn][linkedin-shield]][linkedin-url]
[![Contributors][contributors-shield]][contributors-url]
[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]
[![Downloads][downloads-shield]][downloads-url]
[![Month Downloads][downloads-month-shield]][downloads-month-url]

[contributors-shield]: https://img.shields.io/github/contributors/TsLu1s/SegmentAE.svg?style=for-the-badge&logo=github&logoColor=white
[contributors-url]: https://github.com/TsLu1s/SegmentAE/graphs/contributors
[stars-shield]: https://img.shields.io/github/stars/TsLu1s/SegmentAE.svg?style=for-the-badge&logo=github&logoColor=white
[stars-url]: https://github.com/TsLu1s/SegmentAE/stargazers
[license-shield]: https://img.shields.io/github/license/TsLu1s/SegmentAE.svg?style=for-the-badge&logo=opensource&logoColor=white
[license-url]: https://github.com/TsLu1s/SegmentAE/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/luísfssantos/
[downloads-shield]: https://static.pepy.tech/personalized-badge/segmentae?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Total%20Downloads
[downloads-url]: https://pepy.tech/project/segmentae
[downloads-month-shield]: https://static.pepy.tech/personalized-badge/segmentae?period=month&units=international_system&left_color=grey&right_color=blue&left_text=Month%20Downloads
[downloads-month-url]: https://pepy.tech/project/segmentae


# SegmentAE: A Python Library for Anomaly Detection Optimization

## Framework Overview

`SegmentAE` is designed to enhance anomaly detection performance through the optimization of reconstruction error by integrating and intersecting clustering methods with tabular autoencoders. It provides a versatile, scalable and robust solution for anomaly detection applications in relevant domains such as financial fraud detection or network security, ensuring extensive customization and optimization capabilities.

## Key Features and Capabilities

### 1. General Applicability on Tabular Datasets

`SegmentAE` is engineered to handle a wide range of tabular datasets, making it suitable for various anomaly detection tasks across different use case contexts, it can be seamlessly integrated into diverse applications, ensuring broad utility and adaptability.

### 2. Optimization and Customization

The framework offers complete configurability for each component of the anomaly detection pipeline, this includes data preprocessing, clustering algorithms and provides the customization of baseline autoencoders or the integration of fully developed models. Each component therefore can be fine-tuned to achieve optimal performance tailored to specific use case.

### 3. Enhanced Detection Performance

By leveraging a combination of clustering algorithms and advanced anomaly detection techniques, `SegmentAE` aims to improve the accuracy and reliability of anomaly detection. The integration of tabular autoencoders with clustering mechanisms ensures that the framework effectively captures and identifies different patterns in the input data, optimizing this way the reconstruction error for each existent cluster of the anomaly detection, thereby enhancing predictive performance.

### Main Development Tools <a name = "pre1"></a>

Major frameworks used to built this project: 

* [TensorFlow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [Scikit-Learn](https://scikit-learn.org/stable/)
* [Atlantic](https://pypi.org/project/atlantic/)
* [MLimputer](https://pypi.org/project/mlimputer/)
    
## Where to get it <a name = "ta"></a>
    
Binary installer for the latest released version is available at the Python Package Index [(PyPI)](https://pypi.org/project/segmentae/).   

## Installation  

To install this package from Pypi repository run the following command:

```
pip install segmentae
```

## SegmentAE - Technical Components and Pipeline Structure

The `SegmentAE` framework consists of several integrated components, each playing a critical role in the optimization of anomaly detection through clustering and tabular autoencoders. The pipeline is structured to ensure seamless data flow and modular customization, allowing optimal changes for each use case specific needs.

### 1. Data Preprocessing

Proper preprocessing is crucial for ensuring the quality and consistency of the data fed into the subsequent stages of the pipeline. The data preprocessing module is responsible for preparing raw data for predictive applications, this includes:

- **Missing Value Imputation**: Multiple supervised algorithmic imputation options to handle and impute missing data points.
- **Normalization**: Scaling features to ensure they have comparable magnitudes, essential for the performance of many machine learning algorithms.
- **Categorical Encoding**: Transforming categorical variables into numerical representations suitable for machine learning algorithms, using methods such as label encoding, InverseFrequency encoding and one-hot encoding.

### 2. Clustering

Clustering forms the backbone of the `SegmentAE` framework, providing the capability to segment data into meaningful distinct groups. This segmentation helps in understanding the underlying structure of the input data and provides a basis for the anomaly detection reconstruction error improvements.

- **Clustering Algorithms**: Support and customization for a variety of algorithm options such as `K-Means`, `MiniBatchKMeans`, `GaussianMixture`, and `Agglomerative` clustering, allowing the framework to adapt to different data structures and distribution patterns.

### 3. Anomaly Detection - Baseline Autoencoders

The core of the `SegmentAE` framework is its anomaly detection optimization module, which employs advanced methods such as tabular autoencoders to identify anomalies. Autoencoders are neural networks designed to learn efficient representations of input data, enabling the detection of anomalies by measuring reconstruction errors. This framework includes 3 baseline autoencoder algorithms (`Dense`, `Batch Norm` & `Ensemble`) for user application that allow the customization of each, including the network architecture, training epochs, activation layers and others.

Furthermore, it's a main feature option for you to build your own autoencoder model (`Keras` based) and integrate it into the `SegmentAE` pipeline -> 
<a href="https://github.com/TsLu1s/SegmentAE/blob/main/examples/basic_model.py" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/Custom%20Model-blue?style=for-the-badge&logo=readme&logoColor=white" alt="Custom Model">
</a>

Also, application example for totally unlabeled data available here -> 
<a href="https://github.com/TsLu1s/SegmentAE/blob/main/examples/unlabeled_application.py" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/Unlabeled%20Example-blue?style=for-the-badge&logo=readme&logoColor=white" alt="Unlabeled Example">
</a>

## SegmentAE - Predictive Application

To demonstrate the usage of `SegmentAE`, a DenseAutoencoder is trained and integrated with KMeans clustering (with 3 clusters). The following script outlines the entire process from data loading, preprocessing, clustering, autoencoder training, integration with clustering for anomaly detection, evaluation performance, and predicting future anomalies.

```py
import pandas as pd
from segmentae.data_sources.examples import load_dataset
from segmentae.anomaly_detection import (SegmentAE,
                                         Preprocessing,
                                         Clustering,
                                         DenseAutoencoder,
                                         )
from sklearn.model_selection import train_test_split

## Data Loading

train, test, target = load_dataset(dataset_selection = 'network_intrusions', # Options | 'network_intrusions', 'default_credit_card', 
                                   split_ratio = 0.75)                       #         | 'htru2_dataset', 'shuttle_148'                         

test, future_data = train_test_split(test, train_size = 0.9, random_state = 5)

# Resetting Index is Required
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
future_data = future_data.reset_index(drop=True)

X_train, y_train = train.drop(columns=[target]).copy(), train[target].astype(int) 
X_test, y_test = test.drop(columns=[target]).copy(), test[target].astype(int)
X_future_data = future_data.drop(columns=[target]).copy()

## Preprocessing

pr = Preprocessing(encoder = "IFrequencyEncoder",  # Options | "IFrequencyEncoder", "LabelEncoder", "OneHotEncoder", None
                   scaler = "MinMaxScaler",        # Options | "MinMaxScaler", "StandardScaler", "RobustScaler", None
                   imputer = None)                 # Options | "Simple","RandomForest","ExtraTrees","GBR","KNN",
                                                   #         | "XGBoost","Lightgbm","Catboost", None

pr.fit(X = X_train)
X_train = pr.transform(X = X_train)
X_test = pr.transform(X = X_test)
X_future_data = pr.transform(X = X_future_data)

## Clustering Implementation

cl_model = Clustering(cluster_model = ["KMeans"], # Options | KMeans, MiniBatchKMeans, GMM, Agglomerative
                      n_clusters = 3)
cl_model.clustering_fit(X = X_train)

## Autoencoder Implementation

denseAutoencoder = DenseAutoencoder(hidden_dims = [16, 12, 8, 4],
                                    encoder_activation = 'relu',  
                                    decoder_activation = 'relu',
                                    optimizer = 'adam',
                                    learning_rate = 0.001,
                                    epochs = 150,
                                    val_size = 0.15,
                                    stopping_patient = 20,
                                    dropout_rate = 0.1,
                                    batch_size = None)
denseAutoencoder.fit(input_data = X_train)
denseAutoencoder.summary()

## Autoencoder + Clustering Integration

sg = SegmentAE(ae_model = denseAutoencoder, 
                cl_model = cl_model)

## Train Reconstruction

sg.reconstruction(input_data = X_train,
                  threshold_metric = 'mse')  # Options | mse, mae, rmse, max_error

## Reconstruction Performance (Assuming y_test existence)

results = sg.evaluation(input_data = X_test,
                        target_col = y_test, 
                        threshold_ratio = 2.0) # Selected Threshold Reconstruction Error Multiplier

preds_test, recon_metrics_test = sg.preds_test, sg.reconstruction_test # Test Metadata by Cluster

## Anomaly Detection Predictions

predictions = sg.detections(input_data = X_future_data,
                            threshold_ratio = 2.0)

```

## 👉 Grid Search Optimizer

`SegmentAE` utilizes a comprehensive optimization and evaluation methodology through its `SegmentAE_Optimization` pipeline to assess and enhance its anomaly detection capabilities. This approach incorporates grid search optimization strategy designed for extensive experimental ensembles, aiming to systematically identify the optimal combination of various configurations, including:

- Different autoencoders
- Multiple clustering algorithms
- A range of cluster numbers

Furthermore, the impact of different reconstruction error threshold ratios are also analysed, providing a nuanced understanding of the model's performance across multiple scenarios, identifying areas for potential improvement.
By employing this rigorous optimization strategy, `SegmentAE` can be fine-tuned to deliver superior anomaly detection results across diverse datasets and use cases, allowing data-driven decision-making in selecting the most effective models for specific applications -> <a href="https://github.com/TsLu1s/SegmentAE/blob/main/examples/optimizer_application.py" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/Optimizer%20Application-blue?style=for-the-badge&logo=readme&logoColor=white" alt="Optimizer Application">
</a>
## License

Distributed under the MIT License. See [LICENSE](https://github.com/TsLu1s/SegmentAE/blob/main/LICENSE) for more information.

## Contact 
 
Luis Santos - [LinkedIn](https://www.linkedin.com/in/lu%C3%ADsfssantos/)
