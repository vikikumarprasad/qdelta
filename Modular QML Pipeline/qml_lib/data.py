from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from typing import Tuple

def load_data(
    data_dir: str,
    features: list,
    num_qubits: int,
    reencoding_type: str,
    pca_components: int = None,
    random_seed: int = 42,
    target: str = "ae",       # NEW: 'ae' or 'dh'
    label: str = "delta"      # NEW: 'delta' (Δ-learning) or 'dft' (direct)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads and prepares data from CSV files for the QML model.
    Returns: X_train, X_test, y_train, y_test, pm7_test, dft_true
    """
    print("1. Loading & Preparing Data")
    train_path = os.path.join(data_dir, "train_df_new.csv")
    test_path  = os.path.join(data_dir, "test_df_new.csv")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"Error: Data files 'train_df_new.csv' and/or 'test_df_new.csv' not found in '{data_dir}'")
        sys.exit(1)

    train_df = pd.read_csv(train_path, index_col=0)
    test_df  = pd.read_csv(test_path, index_col=0)

    # --- feature presence check (train governs schema) ---
    missing = [f for f in features if f not in train_df.columns]
    if missing:
        raise KeyError(f"Features missing in train set: {missing}")
    missing_test = [f for f in features if f not in test_df.columns]
    if missing_test:
        raise KeyError(f"Features missing in test set: {missing_test}")

    X_train_df = train_df[features].copy()
    X_test_df  = test_df[features].copy()

    # --- choose target columns ---
    if target not in ("ae", "dh"):
        raise ValueError("target must be 'ae' or 'dh'")
    if target == "ae":
        ycol_delta, ycol_dft, pm7_default, dft_col = "ae_delta", "ae_dft", "AE_mopac", "ae_dft"
    else:
        ycol_delta, ycol_dft, pm7_default, dft_col = "dh_delta", "dh_dft", "DH_Mopac", "dh_dft"

    # training label selection (Δ-learning vs direct)
    if label not in ("delta", "dft"):
        raise ValueError("label must be 'delta' or 'dft'")
    y_train = train_df[ycol_delta].values if label == "delta" else train_df[ycol_dft].values
    y_test  = test_df [ycol_delta].values if label == "delta" else test_df [ycol_dft].values

    # detect PM7 / MOPAC column robustly on TEST (used for reconstruction & reporting)
    if pm7_default in test_df.columns:
        pm7_col = pm7_default
    else:
        pm7_candidates = [c for c in test_df.columns if ("mopac" in c.lower() or "pm7" in c.lower())]
        if not pm7_candidates:
            raise KeyError(f"PM7 column not found. Expected '{pm7_default}', "
                           "or any column containing 'pm7'/'mopac'.")
        pm7_col = pm7_candidates[0]

    pm7_test = test_df[pm7_col].values
    dft_true = test_df[dft_col].values

    # --- basic NaN checks (fail fast & clear) ---
    for name, arr in [("X_train", X_train_df.values), ("X_test", X_test_df.values),
                      ("y_train", y_train), ("y_test", y_test),
                      ("pm7_test", pm7_test), ("dft_true", dft_true)]:
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError(f"{name} contains NaN/Inf values. Please clean the dataset.")

    # --- scale features to [-1, 1] using TRAIN ONLY ---
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train_df)
    X_test  = scaler.transform(X_test_df)

    # --- optional PCA on scaled features (train-fit only) ---
    if pca_components is not None:
        if not isinstance(pca_components, int) or pca_components <= 0:
            raise ValueError("pca_components must be a positive integer")
        if pca_components > X_train.shape[1]:
            raise ValueError(f"pca_components={pca_components} exceeds feature count ({X_train.shape[1]})")
        print(f"Applying PCA, reducing features to {pca_components}")
        pca = PCA(n_components=pca_components, random_state=random_seed)
        X_train = pca.fit_transform(X_train)   # fit on train
        X_test  = pca.transform(X_test)        # transform test

    # --- re-encoding shape checks (unchanged) ---
    if reencoding_type == 'sequential':
        if X_train.shape[1] != num_qubits:
            raise ValueError(
                f"CRITICAL: For 'sequential' re-encoding, number of features ({X_train.shape[1]}) "
                f"must match number of qubits ({num_qubits})."
            )
    elif reencoding_type == 'parallel':
        if len(features) != 1:
            raise ValueError(
                f"CRITICAL: For 'parallel' re-encoding, exactly ONE feature must be provided "
                f"in the --features argument. You provided {len(features)}."
            )
    else:
        raise ValueError("reencoding_type must be 'sequential' or 'parallel'")

    # --- final clipping to guard numeric drift ---
    np.clip(X_train, -1.0, 1.0, out=X_train)
    np.clip(X_test,  -1.0, 1.0, out=X_test)

    print(f"[target={target}, label={label}] Using PM7 column: '{pm7_col}' | DFT column: '{dft_col}'")
    print(f"Data ready. Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")
    return X_train, X_test, y_train, y_test, pm7_test, dft_true
