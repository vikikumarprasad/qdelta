# QML Analysis Pipeline
# Author: Armaan Kautish
# Last Updated: July 4, 2025

# Import all the libraries we'll need for this script.
# These are for basic things like handling files, time, and command-line arguments.
import os
import sys
import time
import argparse
from datetime import datetime

# These are for handling data (numpy, pandas) and plotting (matplotlib).
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# These are for machine learning tasks, like splitting data and scaling features.
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# These are all the quantum machine learning tools from the sQUlearn library.
from squlearn import Executor, optimizers
from squlearn.encoding_circuit import (
    HubregtsenEncodingCircuit, ChebyshevPQC, YZ_CX_EncodingCircuit,
    HighDimEncodingCircuit, KyriienkoEncodingCircuit, ParamZFeatureMap, ChebyshevRx
)
from squlearn.kernel import FidelityKernel, ProjectedQuantumKernel, QKRR, QSVR, QGPR
from squlearn.observables import SummedPaulis
from squlearn.qnn import QNNRegressor, SquaredLoss
from squlearn.qrc import QRCRegressor

# These are for hyperparameter tuning, which helps find the best settings for our model.
import optuna
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import ray
from ray import tune

# This is a dictionary to easily access different encoding circuits by name.
ENCODING_MAP = {
    "hubregtsen": HubregtsenEncodingCircuit, "chebyshev": ChebyshevPQC,
    "yz_cx": YZ_CX_EncodingCircuit, "highdim": HighDimEncodingCircuit,
    "kyriienko": KyriienkoEncodingCircuit, "paramz": ParamZFeatureMap,
    "chebyshev_rx": ChebyshevRx
}

# This dictionary lets us choose an optimizer by name.
OPTIMIZER_MAP = {
    "adam": optimizers.Adam, "lbfgsb": optimizers.LBFGSB,
    "spsa": optimizers.SPSA, "slsqp": optimizers.SLSQP
}

# This dictionary defines the different models we can run and their tunable parameters.
MODEL_CONFIG = {
    "qsvr": (QSVR, {"C": ("loguniform", 1e-2, 1e3), "epsilon": ("loguniform", 1e-3, 1e1)}),
    "qkrr": (QKRR, {"alpha": ("loguniform", 1e-6, 1e1)}),
    "qgpr": (QGPR, {"alpha": ("loguniform", 1e-6, 1e1)}),
    "qnn": (QNNRegressor, {"lr": ("loguniform", 1e-4, 1e-1), "epochs": ("int", 20, 100)}),
    "qrcr": (QRCRegressor, {"lr": ("loguniform", 1e-4, 1e-1), "epochs": ("int", 20, 100)})
}


def setup_arguments():
    """This function defines all the settings you can pass to the script from the command line."""
    # Create a parser to handle the command-line arguments.
    parser = argparse.ArgumentParser(description="Run a QML pipeline.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Arguments for the script itself, like where to find data and save output.
    run_args = parser.add_argument_group("Script and Run Arguments")
    run_args.add_argument("--data_dir", type=str, default="../../data", help="Directory with train_df.csv and test_df.csv.")
    run_args.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    run_args.add_argument("--seed", type=int, default=42, help="A number to make sure our results are reproducible.")
    run_args.add_argument("--n_jobs", type=int, default=1, help="How many CPUs to use for parallel tasks.")

    # Arguments for the quantum model itself.
    model_args = parser.add_argument_group("Model and PQC Arguments")
    model_args.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIG.keys()), help="Which QML model to run.")
    model_args.add_argument("--encoding", type=str, required=True, choices=list(ENCODING_MAP.keys()), help="Which quantum circuit to use.")
    model_args.add_argument("--qubits", type=int, required=True, help="Number of qubits.")
    model_args.add_argument("--layers", type=int, required=True, help="Maximum number of layers in the circuit.")
    
    # Arguments specific to quantum kernels.
    kernel_args = parser.add_argument_group("Quantum Kernel Arguments")
    kernel_args.add_argument("--kernel", type=str, default="projected", choices=["projected", "fidelity"], help="Which type of kernel.")
    kernel_args.add_argument("--train_kernel", action="store_true", help="Set this flag to train the kernel's parameters.")
    kernel_args.add_argument("--kernel_optimizer", type=str, default="adam", choices=list(OPTIMIZER_MAP.keys()), help="Which optimizer to use for kernel training.")
    kernel_args.add_argument("--param_init", type=str, default="random", choices=["random", "zeros"], help="How to initialize circuit parameters.")

    # Arguments for the hyperparameter tuner.
    tuner_args = parser.add_argument_group("Hyperparameter Tuner Arguments")
    tuner_args.add_argument("--tuner", type=str, default="none", choices=["grid", "optuna", "skopt", "raytune", "none"], help="Which hyperparameter tuner to use.")
    tuner_args.add_argument("--n_trials", type=int, default=50, help="How many different settings the tuner should try.")
    tuner_args.add_argument("--cv_type", type=str, default="repeated", choices=["kfold", "repeated"], help="Which type of cross-validation to use.")
    tuner_args.add_argument("--cv_folds", type=int, default=5, help="Number of folds for cross-validation.")
    tuner_args.add_argument("--cv_repeats", type=int, default=3, help="Number of repeats for cross-validation.")

    # Arguments for the data.
    data_args = parser.add_argument_group("Data and Feature Arguments")
    data_args.add_argument("--features", nargs='+', required=True, help="A list of the feature columns to use from the data files.")
    data_args.add_argument("--pca_components", type=int, default=None, help="If you want to use PCA, specify the number of components.")

    # Return all the parsed arguments.
    return parser.parse_args()

def load_data(data_dir, features, num_qubits, pca_components=None, random_seed=42):
    """This function loads the data from CSV files and prepares it for the model."""
    print("1. Loading & Preparing Data")
    # Define the full paths to the training and testing data files.
    train_path = os.path.join(data_dir, "train_df.csv")
    test_path = os.path.join(data_dir, "test_df.csv")

    # Check if the data files actually exist before trying to load them.
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"Error: Data files 'train_df.csv' and/or 'test_df.csv' not found in '{data_dir}'")
        sys.exit(1) # Exit the script if data is missing.

    # Load the data using pandas.
    train_df = pd.read_csv(train_path, index_col=0)
    test_df = pd.read_csv(test_path, index_col=0)

    # Separate the features (X) from the target variable (y).
    X_train_df = train_df[features]
    y_train = train_df["ae_diff"].values
    X_test_df = test_df[features]
    y_test = test_df["ae_diff"].values
    pm7_test = test_df["AE_mopac"].values

    # Scale all feature values to be between -1 and 1. This is important for quantum circuits.
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)
    
    # If PCA is requested, use it to reduce the number of features.
    if pca_components:
        print(f"Applying PCA, reducing features to {pca_components}")
        pca = PCA(n_components=pca_components, random_state=random_seed)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # A very important check: the number of features must match the number of qubits.
    if X_train.shape[1] != num_qubits:
        raise ValueError(f"CRITICAL: Number of features ({X_train.shape[1]}) does not match number of qubits ({num_qubits}).")

    # Make sure no values are outside the [-1, 1] range after scaling.
    np.clip(X_train, -1.0, 1.0, out=X_train)
    np.clip(X_test, -1.0, 1.0, out=X_test)
    
    print(f"Data ready. Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")
    # Return all the prepared data arrays.
    return X_train, X_test, y_train, y_test, pm7_test

def get_pqc(encoding, num_qubits, num_layers):
    """This function builds the quantum circuit (PQC)."""
    # Look up the circuit class from our dictionary.
    pqc_class = ENCODING_MAP[encoding]
    # Create an instance of the circuit with the specified number of qubits and layers.
    return pqc_class(num_qubits=num_qubits, num_features=num_qubits, num_layers=num_layers)

def get_kernel(pqc, kernel_type, param_init, random_seed):
    """This function builds the quantum kernel from the circuit."""
    # Initialize the circuit's parameters, either with zeros or random values.
    if param_init == 'zeros':
        params = np.zeros(pqc.num_parameters)
    else: # 'random'
        rng = np.random.default_rng(random_seed)
        params = rng.uniform(0, 2 * np.pi, pqc.num_parameters)
    
    # The executor is what runs the quantum simulation.
    executor = Executor()
    # Choose between a Fidelity or Projected kernel.
    kernel_class = FidelityKernel if kernel_type == "fidelity" else ProjectedQuantumKernel
    # Create the kernel instance.
    return kernel_class(encoding_circuit=pqc, executor=executor, initial_parameters=params)

def get_optimizer(name, learning_rate):
    """This function gets an optimizer instance for training."""
    # Look up the optimizer class from our dictionary.
    if name in OPTIMIZER_MAP:
        return OPTIMIZER_MAP[name](options={"lr": learning_rate})
    
    # If the name isn't found, just use Adam as a default.
    print(f"Warning: Optimizer '{name}' not found, falling back to Adam.")
    return optimizers.Adam(options={"lr": learning_rate})

def tune_model(args, create_model_fn, search_space, X_train, y_train):
    """This function runs the hyperparameter tuning to find the best model settings."""
    print(f"2. Tuning with {args.tuner.upper()}")

    # Set up the cross-validation splitter. This splits the data for training and validation.
    cv_splitter = (
        RepeatedKFold(n_splits=args.cv_folds, n_repeats=args.cv_repeats, random_state=args.seed)
        if args.cv_type == "repeated"
        else KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    )

    # This is the "objective function" that the tuner tries to minimize.
    # It calculates the model's error (MAE) for a given set of parameters.
    def cv_objective(params):
        # Create a model with the given parameters.
        model = create_model_fn(params)
        scores = []
        # Loop through each cross-validation split.
        for train_idx, val_idx in cv_splitter.split(X_train):
            try:
                # It's very important to clone the model to get a fresh, unfitted one for each fold.
                fold_model = clone(model)
                # Train the model on the training part of the data.
                fold_model.fit(X_train[train_idx], y_train[train_idx])
                # Make predictions on the validation part.
                preds = fold_model.predict(X_train[val_idx])
                # Calculate the error and save it.
                scores.append(mean_absolute_error(y_train[val_idx], preds))
            except Exception as e:
                # If a fold fails for some reason, we give it a very high error score.
                print(f"CV fold failed with params {params}: {e}")
                return np.inf
        # The final score for this set of parameters is the average of all fold scores.
        return np.mean(scores)

    # Depending on the chosen tuner, run the appropriate tuning process.
    best_params = {}
    if args.tuner == "none":
        print("No tuning selected, using default parameters.")

    elif args.tuner == "grid":
        print("Starting GridSearchCV...")
        # GridSearchCV needs a list of values to try for each parameter.
        grid_search_space = {k: (list(range(v[1], v[2] + 1)) if v[0] == 'int' else np.logspace(np.log10(v[1]), np.log10(v[2]), 5)) for k, v in search_space.items()}
        
        grid_search = GridSearchCV(
            estimator=create_model_fn({}),
            param_grid=grid_search_space,
            scoring="neg_mean_absolute_error",
            cv=cv_splitter,
            n_jobs=args.n_jobs,
            verbose=3
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print(f"GridSearch best MAE: {-grid_search.best_score_:.4f}")

    elif args.tuner == "optuna":
        # The objective function for Optuna.
        def optuna_objective(trial):
            params = {}
            # Optuna suggests values for each parameter from the defined search space.
            for name, (space_type, low, high) in search_space.items():
                if space_type == "loguniform":
                    params[name] = trial.suggest_float(name, low, high, log=True)
                elif space_type == "int":
                    params[name] = trial.suggest_int(name, low, high)
            return cv_objective(params)
        
        study = optuna.create_study(direction="minimize")
        study.optimize(optuna_objective, n_trials=args.n_trials)
        best_params = study.best_params
        print(f"Optuna best MAE: {study.best_value:.4f}")

    elif args.tuner == "skopt":
        # Skopt needs the search space defined as a list of "dimensions".
        dimensions = []
        for name, (space_type, low, high) in search_space.items():
            if space_type == "loguniform":
                dimensions.append(Real(low, high, "log-uniform", name=name))
            elif space_type == "int":
                dimensions.append(Integer(low, high, name=name))

        # The objective function for skopt.
        @use_named_args(dimensions)
        def skopt_objective(**params):
            return cv_objective(params)
            
        result = gp_minimize(skopt_objective, dimensions, n_calls=args.n_trials, random_state=args.seed)
        best_params = {dim.name: val for dim, val in zip(dimensions, result.x)}
        print(f"Skopt best MAE: {result.fun:.4f}")
        
    elif args.tuner == "raytune":
        # Initialize the Ray library for parallel processing.
        if not ray.is_initialized():
            ray.init(num_cpus=args.n_jobs, ignore_reinit_error=True)
            
        # The objective function for Ray Tune.
        def raytune_objective(config):
            score = cv_objective(config)
            # Ray needs you to "report" the score back to it.
            tune.report(mae=score)

        # Convert our search space to Ray Tune's format.
        ray_search_space = {}
        for name, (space_type, low, high) in search_space.items():
            if space_type == "loguniform":
                ray_search_space[name] = tune.loguniform(low, high)
            elif space_type == "int":
                # Note: Ray's randint is exclusive on the upper bound, so we add 1.
                ray_search_space[name] = tune.randint(low, high + 1)

        analysis = tune.run(
            raytune_objective,
            config=ray_search_space,
            num_samples=args.n_trials,
            metric="mae",
            mode="min",
            verbose=1
        )
        best_params = analysis.best_config
        print(f"Ray Tune best MAE: {analysis.best_result['mae']:.4f}")
        # Shut down Ray when we're done with it.
        ray.shutdown()

    # After tuning, train one final model using the best parameters found.
    print("\n3. Training Final Model")
    print(f"Best settings found: {best_params}")
    final_model = create_model_fn(best_params)
    final_model.fit(X_train, y_train)
    
    # Return the fully trained model and the best parameters.
    return final_model, best_params

def save_results(args, y_test, y_pred, pm7_test, best_params):
    """This function calculates the final scores and saves all results."""
    print("\n4. Final Evaluation & Saving Results")
    # Make sure the output directory exists.
    os.makedirs(args.output_dir, exist_ok=True)

    # Calculate the final error metrics.
    base_mae = mean_absolute_error(y_test, y_pred)
    dft_true = pm7_test + y_test
    dft_pred = pm7_test + y_pred
    delta_mae = mean_absolute_error(dft_true, dft_pred)

    print(f"  - Final Corrected MAE (Δ-MAE): {delta_mae:.4f} kcal/mol")
    print(f"  - Base MAE (on correction):    {base_mae:.4f} kcal/mol")

    # Create a unique filename for this run based on the settings.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = f"{args.model.upper()}_{args.encoding}_q{args.qubits}_L{args.layers}_{args.tuner}"
    fname_base = os.path.join(args.output_dir, f"{config_str}_{timestamp}")
    
    # Save the predictions to a CSV file.
    pred_df = pd.DataFrame({
        "AE_PM7": pm7_test, "True_Delta": y_test, "Predicted_Delta": y_pred,
        "True_DFT": dft_true, "Predicted_DFT": dft_pred
    })
    pred_filename = f"{fname_base}_predictions.csv"
    pred_df.to_csv(pred_filename, index=False)
    print(f"Predictions saved to {pred_filename}")

    # Save a summary of this run to a central log file.
    log_path = os.path.join(args.output_dir, "summary_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    log_entry = {
        "Timestamp": timestamp, "Model": args.model.upper(), "Encoding": args.encoding,
        "Qubits": args.qubits, "Max_Layers": args.layers, "Tuner": args.tuner,
        "Train_Kernel": args.train_kernel, "Delta_MAE_kcal_mol": round(delta_mae, 4),
        "Best_Params": str(best_params) if best_params else "N/A"
    }
    # Add a header to the log file only if it's new.
    header = not os.path.exists(log_path)
    pd.DataFrame([log_entry]).to_csv(log_path, mode="a", header=header, index=False)
    print(f"Results logged to {log_path}")

    # Create and save a plot of the results.
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 8))
    lims = [min(dft_true.min(), dft_pred.min()) - 5, max(dft_true.max(), dft_pred.max()) + 5]
    plt.scatter(dft_true, dft_pred, alpha=0.7, edgecolor='k', s=25)
    plt.plot(lims, lims, 'r--', label=f"MAE = {delta_mae:.2f} kcal/mol")
    plt.xlabel("Reference ΔH (kcal/mol) [DFT]", fontsize=12)
    plt.ylabel(f"Predicted ΔH (kcal/mol) [PM7 + {args.model.upper()}]", fontsize=12)
    plt.title(f"{args.model.upper()} | {args.encoding.title()} | {args.qubits}Q | Tuner: {args.tuner.title()}", fontsize=14, pad=10)
    plt.legend(fontsize=12)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plot_filename = f"{fname_base}_results_plot.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close() # Close the plot to free up memory.
    print(f"Plot saved to {plot_filename}")


def main():
    """This is the main function that runs the entire pipeline from start to finish."""
    # Record the start time so we can see how long the script takes.
    start_time = time.time()
    # Get all the settings from the command line.
    args = setup_arguments()

    # Print a summary of the settings for this run.
    print(f"--- Starting QML Pipeline: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"Run Config: Model={args.model.upper()}, Tuner={args.tuner.upper()}, PQC={args.encoding}, Qubits={args.qubits}, Max Layers={args.layers}, Seed={args.seed}")
    if args.model in ["qsvr", "qkrr", "qgpr"]:
        print(f"Kernel Config: Train={args.train_kernel}, Optimizer={args.kernel_optimizer if args.train_kernel else 'N/A'}")
    print(f"Features: {args.features}\n")

    # Step 1: Load and prepare the data.
    X_train, X_test, y_train, y_test, pm7_test = load_data(
        args.data_dir, args.features, args.qubits, args.pca_components, args.seed
    )

    # Get the model class and the search space for the chosen model.
    model_class, search_space = MODEL_CONFIG[args.model]
    
    # Add the number of layers to the search space for all models.
    search_space["num_layers"] = ("int", 1, args.layers)
    
    # If we are training the kernel, also add the learning rate to the search space.
    if args.train_kernel and args.model in ["qsvr", "qkrr", "qgpr"]:
        search_space["lr"] = ("loguniform", 1e-4, 1e-1)

    # This is a "factory" function. Its job is to create a QML model
    # based on a dictionary of parameters. The tuner will use this function.
    def create_model_from_params(params):
        # Get the number of layers from the parameters, or use the default.
        num_layers = params.get("num_layers", args.layers)
        # Create the quantum circuit.
        pqc = get_pqc(args.encoding, args.qubits, num_layers)

        # If it's a kernel-based model (QSVR, QKRR, QGPR)...
        if args.model in ["qsvr", "qkrr", "qgpr"]:
            # Create the quantum kernel.
            kernel = get_kernel(pqc, args.kernel, args.param_init, args.seed)
            model_args = {"quantum_kernel": kernel}
            
            if args.model == "qsvr":
                model_args.update({"C": params.get("C", 1.0), "epsilon": params.get("epsilon", 0.1)})
            else: # QKRR or QGPR
                model_args.update({"alpha": params.get("alpha", 1e-2)})
            
            # Create the final model instance.
            model = model_class(**model_args)

            # If kernel training is enabled, set the training properties on the model.
            if args.train_kernel:
                model.train_kernel = True
                model.optimizer = get_optimizer(args.kernel_optimizer, params.get("lr", 0.01))
            return model
        
        # Otherwise, it's a QNN or QRCR model.
        else:
            optimizer = get_optimizer(args.optimizer, params.get("lr", 0.01))
            if args.model == "qnn":
                epochs = params.get("epochs", 50) 
                batch_size = params.get("batch_size", 16)
                variance = params.get("variance", 0.01)
                observable = SummedPaulis(num_qubits=args.qubits)
                return QNNRegressor(
                    pqc, observable, Executor(), SquaredLoss(), optimizer,
                    epochs=epochs, batch_size=batch_size, variance=variance
                )
            elif args.model == "qrcr":
                epochs = params.get("epochs", 50)
                batch_size = params.get("batch_size", 16)
                return QRCRegressor(
                    pqc, executor=Executor(), optimizer=optimizer, loss=SquaredLoss(),
                    epochs=epochs, batch_size=batch_size
                )
    
    # Step 2: Run the tuning process to find the best model.
    final_model, best_params = tune_model(args, create_model_from_params, search_space, X_train, y_train)
    
    # Step 3: Use the best model to make predictions on the test data.
    predictions = final_model.predict(X_test)
    # Step 4: Save all the results.
    save_results(args, y_test, predictions, pm7_test, best_params)
    
    # Record the end time and print the total duration.
    end_time = time.time()
    print(f"\n--- Pipeline finished in {end_time - start_time:.2f} seconds ---")

# This standard Python construct ensures that the main() function is called only when the script is executed directly.
if __name__ == "__main__":
    main()