# Purpose: Handles all data loading, validation, and preprocessing logic.

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def load_data(data_dir, features, num_qubits, reencoding_type, pca_components=None, random_seed=42):
    """Loads and prepares data from CSV files for the QML model."""
    print("1. Loading & Preparing Data")
    train_path = os.path.join(data_dir, "train_df.csv")
    test_path = os.path.join(data_dir, "test_df.csv")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"Error: Data files 'train_df.csv' and/or 'test_df.csv' not found in '{data_dir}'")
        sys.exit(1)

    train_df = pd.read_csv(train_path, index_col=0)
    test_df = pd.read_csv(test_path, index_col=0)

    X_train_df = train_df[features]
    y_train = train_df["ae_diff"].values
    X_test_df = test_df[features]
    y_test = test_df["ae_diff"].values
    pm7_test = test_df["AE_mopac"].values

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)
    
    if pca_components:
        print(f"Applying PCA, reducing features to {pca_components}")
        pca = PCA(n_components=pca_components, random_state=random_seed)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    if reencoding_type == 'sequential':
        if X_train.shape[1] != num_qubits:
            raise ValueError(f"CRITICAL: For 'sequential' re-encoding, number of features ({X_train.shape[1]}) must match number of qubits ({num_qubits}).")
    elif reencoding_type == 'parallel':
        if len(features) != 1:
            raise ValueError(f"CRITICAL: For 'parallel' re-encoding, exactly ONE feature must be provided in the --features argument. You provided {len(features)}.")
    
    np.clip(X_train, -1.0, 1.0, out=X_train)
    np.clip(X_test, -1.0, 1.0, out=X_test)
    
    print(f"Data ready. Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")
    return X_train, X_test, y_train, y_test, pm7_test