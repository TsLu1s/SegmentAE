import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
from typing import Optional

def load_dataset(dataset_selection : str = 'injection_quality',
                 split_ratio : float = 0.8,
                 random_state : Optional[int] = 5):       
    """
    Load and preprocess datasets for anomaly detection machine learning tasks.

    Parameters:
        dataset_selection (str): Name of the dataset to load. Defaults to 'injection_quality'.
        split_ratio (float): Ratio of the dataset to use for training. Defaults to 0.8.
        random_state (Optional[int]): Seed for random number generation. Defaults to 5.

    Returns:
        train (DataFrame): Training dataset.
        test (DataFrame): Testing dataset.
        target (str): Name of the target variable. (used for validation)
    """
    # Handle different dataset selections
    if dataset_selection == "german_credit_card":
        # Source : https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
        
        # fetch dataset 
        info = fetch_ucirepo(id=148) 
        
        # Concatenate features and targets
        data = pd.concat([info.data.features.reset_index(drop=True),
                          info.data.targets.reset_index(drop=True) ],axis=1)
        
        # Adjust target values to binary
        target, sr = 'class', 0.75
        data[target] = data[target].replace({1: 0, 2: 1, 3: 1, 4: 1, 5: 1})
        data[target] = data[target].astype(int)
        
        # Separate normal and fraud instances
        normal, fraud = data[data[target] == 0], data[data[target] == 1]
        
        # Split normal instances into training and testing sets
        train, test = train_test_split(normal, train_size=split_ratio, random_state=random_state)
        
        # Combine testing set with fraud instances and shuffle
        test = pd.concat([test,fraud])
        test = test.sample(frac=1, random_state=42)
    
    elif dataset_selection == "default_credit_card":
        # Source : https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
        
        # fetch dataset 
        info = fetch_ucirepo(id=350)
          
        # Concatenate features and targets
        data = pd.concat([info.data.features ,info.data.targets ],axis=1)
        target, sr = 'Y', 0.75
        
        # Cast target values to integer
        data[target] = data[target].astype(int)
        
        # Separate normal and fraud instances
        normal, fraud = data[data[target] == 0], data[data[target] == 1]
        
        # Split normal instances into training and testing sets
        train, test = train_test_split(normal, train_size=split_ratio, random_state=random_state)
        
        # Combine testing set with fraud instances and shuffle
        test = pd.concat([test,fraud])
        test = test.sample(frac=1, random_state=42)
        
    elif dataset_selection == "network_intrusions":
        # Source : https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
        
        # URL for the KDD Cup 1999 dataset
        url = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz'

        # Define feature names based on the dataset documentation
        column_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
            'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
        ]
        
        # Read the dataset into a pandas DataFrame
        data = pd.read_csv(url, compression='gzip', header=None, names=column_names)

        # Adjust target labels to binary
        data.label=data.label.apply(lambda x: 0 if x == 'normal.' else 1)
        
        # Define proportions for sampling
        proportions = {
            0: 0.5,
            1: 0.10
        }

        target, sr = 'label', 0.75
        
        # Drop columns with unique or single unique values, except for the target column
        data = data.drop(columns=[col for col in data.columns 
                               if (data[col].nunique() == len(data) or data[col].nunique() == 1)
                                                                             and col != target])
        
        # Sample instances based on defined proportions
        data = pd.concat([
            data[data[target] == label].sample(frac=proportion)
            for label, proportion in proportions.items()
        ]).reset_index(drop=True)
        
        # Separate normal and intrusion instances
        normal, intrusions = data[data[target] == 0], data[data[target] == 1]
        
        # Split normal instances into training and testing sets
        train, test = train_test_split(normal, train_size=split_ratio, random_state=random_state)
        
        # Combine testing set with intrusion instances and shuffle
        test = pd.concat([test,intrusions])
        test = test.sample(frac=1, random_state=42)
    
    # Reset index for consistency
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    
    # Print information about the dataset
    print({**{"Train Length": len(train),
              "Test Length": len(test),
              "Suggested Split_Ratio": sr},
           **{"Anomalies [1]" if key == 1 else "Normal [0]": value for key, value in test[target].value_counts().to_dict().items()}})
    
    return train, test, target
