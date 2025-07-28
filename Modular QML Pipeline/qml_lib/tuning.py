# Purpose: Manages the entire hyperparameter tuning process for all tuners.

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV

import optuna
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import ray
from ray import tune

from .models import create_model_from_params

def tune_model(args, search_space, X_train, y_train):
    """Runs hyperparameter tuning and returns the best parameters found."""
    print(f"2. Tuning with {args.tuner.upper()}")
    
    create_model_fn = lambda p: create_model_from_params(args, p)

    cv_splitter = (
        RepeatedKFold(n_splits=args.cv_folds, n_repeats=args.cv_repeats, random_state=args.seed)
        if args.cv_type == "repeated"
        else KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    )

    def cv_objective(params):
        model = create_model_fn(params)
        scores = []
        for train_idx, val_idx in cv_splitter.split(X_train):
            try:
                fold_model = clone(model)
                fold_model.fit(X_train[train_idx], y_train[train_idx])
                preds = fold_model.predict(X_train[val_idx])
                scores.append(mean_absolute_error(y_train[val_idx], preds))
            except Exception as e:
                print(f"CV fold failed with params {params}: {e}")
                return np.inf
        return np.mean(scores)

    best_params = {}
    if args.tuner == "none":
        print("No tuning selected, using default parameters.")
    elif args.tuner == "grid":
        print("Starting GridSearchCV...")
        grid_search_space = {k: (list(range(v[1], v[2] + 1)) if v[0] == 'int' else np.logspace(np.log10(v[1]), np.log10(v[2]), 5)) for k, v in search_space.items()}
        grid_search = GridSearchCV(
            estimator=create_model_fn({}), param_grid=grid_search_space, scoring="neg_mean_absolute_error",
            cv=cv_splitter, n_jobs=args.n_jobs, verbose=3
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print(f"GridSearch best MAE: {-grid_search.best_score_:.4f}")
    elif args.tuner == "optuna":
        def optuna_objective(trial):
            params = {}
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
        dimensions = []
        for name, (space_type, low, high) in search_space.items():
            if space_type == "loguniform":
                dimensions.append(Real(low, high, "log-uniform", name=name))
            elif space_type == "int":
                dimensions.append(Integer(low, high, name=name))
        @use_named_args(dimensions)
        def skopt_objective(**params):
            return cv_objective(params)
        result = gp_minimize(skopt_objective, dimensions, n_calls=args.n_trials, random_state=args.seed)
        best_params = {dim.name: val for dim, val in zip(dimensions, result.x)}
        print(f"Skopt best MAE: {result.fun:.4f}")
    elif args.tuner == "raytune":
        if not ray.is_initialized():
            ray.init(num_cpus=args.n_jobs, ignore_reinit_error=True)
        def raytune_objective(config):
            score = cv_objective(config)
            tune.report(mae=score)
        ray_search_space = {}
        for name, (space_type, low, high) in search_space.items():
            if space_type == "loguniform":
                ray_search_space[name] = tune.loguniform(low, high)
            elif space_type == "int":
                ray_search_space[name] = tune.randint(low, high + 1)
        analysis = tune.run(
            raytune_objective, config=ray_search_space, num_samples=args.n_trials,
            metric="mae", mode="min", verbose=1
        )
        best_params = analysis.best_config
        print(f"Ray Tune best MAE: {analysis.best_result['mae']:.4f}")
        ray.shutdown()

    return best_params