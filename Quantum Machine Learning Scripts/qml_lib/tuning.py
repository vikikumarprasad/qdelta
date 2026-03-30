# qml_lib/tuning.py
# Hyperparameter tuning for all supported tuners

from typing import Any, Dict, Iterable, List
import numpy as np

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV
from joblib import Parallel, delayed, parallel_backend

import optuna
from skopt import BayesSearchCV
from skopt.callbacks import DeltaYStopper, DeadlineStopper
from skopt.space import Real, Integer, Categorical

from .models import create_model_from_params


def _slice_rows(X, idx):
    # supports both numpy arrays and pandas DataFrames/Series
    try:
        return X.iloc[idx]
    except AttributeError:
        return X[idx]


def _normalize_int_grid(val: Any) -> List[int]:
    # normalises num_layers-style specs into explicit integer lists
    if isinstance(val, range):
        return list(map(int, val))
    if isinstance(val, np.ndarray):
        return [int(x) for x in val.tolist()]
    if isinstance(val, tuple) and len(val) == 3 and isinstance(val[0], str):
        if val[0].lower() in ("int", "range"):
            low, high = int(val[1]), int(val[2])
            return list(range(low, high + 1))
    if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
        return [int(x) for x in val]
    if isinstance(val, str) and "," in val:
        return [int(s.strip()) for s in val.split(",") if s.strip()]
    return [int(val)]


def _grid_points_from_spec(spec, default_num=7):
    # converts spec-like entries into explicit lists for GridSearchCV
    if isinstance(spec, range):
        return list(spec)
    if isinstance(spec, (list, tuple, np.ndarray)) and len(spec) > 0 and not isinstance(spec[0], str):
        return list(spec)
    if isinstance(spec, (list, tuple)) and len(spec) >= 3 and isinstance(spec[0], str):
        typ = spec[0].lower()
        low, high = spec[1], spec[2]
        N = int(spec[3]) if len(spec) >= 4 and isinstance(spec[3], (int, float)) else int(default_num)
        if typ == "int":
            lo, hi = int(low), int(high)
            if lo > hi: lo, hi = hi, lo
            if lo == hi: return [lo]
            return list(np.round(np.linspace(lo, hi, num=N)).astype(int))
        if typ == "loguniform":
            lo, hi = float(low), float(high)
            lo = max(lo, 1e-12)
            if lo == hi: return [lo]
            return list(np.logspace(np.log10(lo), np.log10(hi), num=N))
        if typ == "linspace":
            lo, hi = float(low), float(high)
            if lo == hi: return [lo]
            return list(np.linspace(lo, hi, num=N))
        if typ == "cat":
            return list(low)
    return [spec]


def _resolve_effective_pqk_backend(args) -> str:
    cached = getattr(args, "_effective_pqk_backend", None)
    if cached in ("qiskit", "pennylane"):
        return cached
    requested = getattr(args, "pqk_backend", "auto")
    if requested in ("qiskit", "pennylane"):
        return requested
    try:
        from qiskit_aer import Aer  # noqa: F401
        return "qiskit"
    except Exception:
        return "pennylane"


def _effective_tuner_n_jobs(args, backend_choice: str) -> int:
    n_jobs = max(1, int(getattr(args, "n_jobs", 1)))
    kernel_name          = getattr(args, "kernel", "").lower()
    effective_pqk_backend = _resolve_effective_pqk_backend(args)
    # PennyLane's projected kernel is not thread-safe; force serial execution
    if backend_choice == "threading" and kernel_name == "projected" and effective_pqk_backend == "pennylane":
        if n_jobs != 1:
            print("--- [Tuner]: Projected+PennyLane detected. Forcing n_jobs=1 for stability. ---")
        return 1
    return n_jobs


def _get_parallel_backend_choice(args):
    model_name  = getattr(args, "model",  "").lower()
    kernel_name = getattr(args, "kernel", "").lower()

    if model_name in ("qnn-iqp", "qnn-cpmap"):
        print("--- [Tuner]: QNN model detected. Forcing 'threading' to avoid pickling errors. ---")
        return "threading"

    if kernel_name in ("fidelity", "projected"):
        # 'threading' shares the module-level simulation cache across workers;
        # 'loky' gives each worker its own process and loses cache sharing
        print(f"--- [Tuner]: {kernel_name.title()} kernel detected. Using 'threading'. ---")
        return "threading"

    return "multiprocessing"


def tune_model(args, search_space: Dict[str, Any], X_train, y_train):
    print(f"2. Tuning with {args.tuner.upper()}")

    create_model_fn = lambda p: create_model_from_params(args, p)

    cv_splitter = (
        RepeatedKFold(n_splits=args.cv_folds, n_repeats=args.cv_repeats, random_state=args.seed)
        if getattr(args, "cv_type", "repeated") == "repeated"
        else KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    )

    def cv_objective(params: Dict[str, Any]) -> float:
        # coerce numpy scalar types to native Python before passing to sklearn
        clean_params = {}
        for k, v in params.items():
            if isinstance(v, np.integer):    clean_params[k] = int(v)
            elif isinstance(v, np.floating): clean_params[k] = float(v)
            else:                            clean_params[k] = v

        model = create_model_fn(clean_params)

        def _one_fold(train_idx, val_idx):
            try:
                X_tr = _slice_rows(X_train, train_idx)
                y_tr = _slice_rows(y_train, train_idx)
                X_va = _slice_rows(X_train, val_idx)
                y_va = _slice_rows(y_train, val_idx)
                try:
                    fold_model = clone(model)
                except Exception:
                    fold_model = create_model_fn(clean_params)
                fold_model.fit(X_tr, y_tr)
                return mean_absolute_error(y_va, fold_model.predict(X_va))
            except Exception as e:
                print(f"CV fold failed with params {clean_params}: {e}")
                return float("inf")

        backend_choice = _get_parallel_backend_choice(args)
        n_jobs         = _effective_tuner_n_jobs(args, backend_choice)

        with parallel_backend(backend_choice, n_jobs=n_jobs):
            scores = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_one_fold)(tr, va) for tr, va in cv_splitter.split(X_train)
            )

        if any(np.isinf(s) for s in scores):
            return float("inf")
        return float(np.mean(scores))

    if args.tuner == "none":
        print("No tuning selected, using default parameters.")
        return {}

    if args.tuner == "grid":
        print("Starting GridSearchCV...")
        grid_space = dict(search_space)
        model_name = getattr(args, "model", "").lower()

        if model_name == "qnn-iqp" and "num_layers" in grid_space:
            from .qnn import IQPCircuitWrapper
            L_grid       = _normalize_int_grid(grid_space.pop("num_layers"))
            num_features = getattr(args, "input_dim", args.qubits)
            grid_space["encoding_circuit"] = [
                IQPCircuitWrapper(num_qubits=args.qubits, num_features=num_features, num_layers=L)
                for L in L_grid
            ]
        if model_name == "qnn-cpmap" and "num_layers" in grid_space:
            from .qnn import CPKernelWrapper
            L_grid       = _normalize_int_grid(grid_space.pop("num_layers"))
            num_features = getattr(args, "input_dim", args.qubits)
            grid_space["encoding_circuit"] = [
                CPKernelWrapper(num_qubits=args.qubits, num_features=num_features, num_layers=L)
                for L in L_grid
            ]

        grid_points = getattr(args, "grid_points", 7)
        grid_space  = {k: _grid_points_from_spec(v, default_num=grid_points) for k, v in grid_space.items()}

        model_for_keys = create_model_from_params(args, {})
        valid_keys     = set(model_for_keys.get_params(deep=False).keys())
        grid_space     = {k: v for k, v in grid_space.items() if k in valid_keys}

        backend_choice = _get_parallel_backend_choice(args)
        n_jobs         = _effective_tuner_n_jobs(args, backend_choice)

        gs = GridSearchCV(
            estimator=create_model_from_params(args, {}),
            param_grid=grid_space,
            scoring="neg_mean_absolute_error",
            cv=cv_splitter,
            n_jobs=n_jobs,
            verbose=getattr(args, "verbose", 0),
        )
        print(f"--- [GridSearchCV]: Using '{backend_choice}' backend for {n_jobs} jobs. ---")
        with parallel_backend(backend_choice, n_jobs=n_jobs):
            gs.fit(X_train, y_train)

        best_params  = gs.best_params_
        print(f"GridSearch best MAE: {-gs.best_score_:.4f}")

        refine_rounds = getattr(args, "grid_refine", 0)
        if refine_rounds and refine_rounds > 0:
            bp = dict(best_params)

            def _refine_log_list(x, width_decades=0.5, n=5):
                lo, hi = x / (10 ** width_decades), x * (10 ** width_decades)
                lo = max(lo, 1e-12)
                return list(np.logspace(np.log10(lo), np.log10(hi), num=n))

            refine_grid: Dict[str, List[Any]] = {}
            for name, values in grid_space.items():
                if name in bp and isinstance(bp[name], (int, float)):
                    v = float(bp[name])
                    if v > 0 and name.lower() in {"c", "epsilon", "alpha", "sigma", "lr"}:
                        refine_grid[name] = _refine_log_list(v, width_decades=0.5, n=5)
                    elif name.lower() in {"epochs", "batch_size", "num_layers"}:
                        lo = max(1, int(round(0.7 * v)))
                        hi = max(lo, int(round(1.3 * v)))
                        refine_grid[name] = list(range(lo, hi + 1))
                    else:
                        refine_grid[name] = values
                else:
                    refine_grid[name] = values

            print("Starting refinement GridSearchCV around previous best...")
            gs2 = GridSearchCV(
                estimator=create_model_from_params(args, bp),
                param_grid=refine_grid,
                scoring="neg_mean_absolute_error",
                cv=cv_splitter,
                n_jobs=n_jobs,
                verbose=getattr(args, "verbose", 0),
            )
            with parallel_backend(backend_choice, n_jobs=n_jobs):
                gs2.fit(X_train, y_train)
            best_params = gs2.best_params_
            print(f"Refined Grid best MAE: {-gs2.best_score_:.4f}")

        return best_params

    if args.tuner == "optuna":
        def optuna_objective(trial):
            params: Dict[str, Any] = {}
            for name, spec in search_space.items():
                if not isinstance(spec, (list, tuple)) or len(spec) < 1:
                    continue
                space_type = spec[0]
                if space_type == "loguniform":
                    params[name] = trial.suggest_float(name, float(spec[1]), float(spec[2]), log=True)
                elif space_type == "uniform":
                    params[name] = trial.suggest_float(name, float(spec[1]), float(spec[2]), log=False)
                elif space_type == "int":
                    params[name] = trial.suggest_int(name, int(spec[1]), int(spec[2]))
                elif space_type == "cat":
                    params[name] = trial.suggest_categorical(name, list(spec[1]))
                else:
                    raise ValueError(f"Unsupported space type for {name}: {space_type}")
            return cv_objective(params)

        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study   = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(optuna_objective, n_trials=args.n_trials, n_jobs=1)
        print(f"Optuna best MAE: {study.best_value:.4f}")
        return study.best_params

    if args.tuner == "skopt":
        model_name = getattr(args, "model", "").lower()
        estimator  = create_model_from_params(args, {})
        valid_keys = set(estimator.get_params(deep=False).keys())

        space_src: Dict[str, Any] = dict(search_space)
        sk_space:  Dict[str, Any] = {}

        if model_name == "qnn-iqp" and "num_layers" in space_src:
            from .qnn import IQPCircuitWrapper
            L_grid       = _normalize_int_grid(space_src.pop("num_layers"))
            num_features = getattr(args, "input_dim", args.qubits)
            enc_candidates = [
                IQPCircuitWrapper(num_qubits=args.qubits, num_features=num_features, num_layers=L)
                for L in L_grid
            ]
            if "encoding_circuit" in valid_keys:
                sk_space["encoding_circuit"] = Categorical(enc_candidates)

        if model_name == "qnn-cpmap" and "num_layers" in space_src:
            from .qnn import CPKernelWrapper
            L_grid       = _normalize_int_grid(space_src.pop("num_layers"))
            num_features = getattr(args, "input_dim", args.qubits)
            enc_candidates = [
                CPKernelWrapper(num_qubits=args.qubits, num_features=num_features, num_layers=L)
                for L in L_grid
            ]
            if "encoding_circuit" in valid_keys:
                sk_space["encoding_circuit"] = Categorical(enc_candidates)

        space_src = {k: v for k, v in space_src.items() if k in valid_keys}
        for name, spec in space_src.items():
            if not isinstance(spec, (list, tuple)) or len(spec) < 1:
                continue
            t = spec[0]
            if t == "int":
                lo, hi = int(spec[1]), int(spec[2])
                if lo > hi: lo, hi = hi, lo
                sk_space[name] = Categorical([lo]) if lo == hi else Integer(lo, hi)
            elif t == "loguniform":
                sk_space[name] = Real(float(spec[1]), float(spec[2]), prior="log-uniform")
            elif t == "uniform":
                sk_space[name] = Real(float(spec[1]), float(spec[2]), prior="uniform")
            elif t == "cat":
                sk_space[name] = Categorical(list(spec[1]))
            else:
                raise ValueError(f"Unsupported space type for {name}: {t}")

        if not sk_space:
            print("No tunable hyperparameters after filtering; using defaults.")
            return {}

        backend_choice = _get_parallel_backend_choice(args)
        n_jobs         = _effective_tuner_n_jobs(args, backend_choice)
        n_points       = 1 if n_jobs == 1 else max(1, min(4, n_jobs // 3 or 1))
        n_init_raw     = max(8, 2 * len(sk_space) + 2)
        n_init         = 1
        while n_init < n_init_raw:
            n_init <<= 1

        opt = BayesSearchCV(
            estimator=estimator,
            search_spaces=sk_space,
            n_iter=args.n_trials,
            cv=cv_splitter,
            scoring="neg_mean_absolute_error",
            n_jobs=n_jobs,
            n_points=n_points,
            random_state=args.seed,
            optimizer_kwargs=dict(
                base_estimator="GP",
                acq_func="gp_hedge",
                acq_optimizer="auto",
                initial_point_generator="sobol",
                n_initial_points=n_init,
            ),
            refit=True,
            return_train_score=False,
            verbose=getattr(args, "verbose", 0),
        )

        effective_pqk_backend = _resolve_effective_pqk_backend(args)
        print(
            f"--- [BayesSearchCV]: Using '{backend_choice}' backend for {n_jobs} jobs "
            f"(effective_pqk_backend={effective_pqk_backend}). ---"
        )
        with parallel_backend(backend_choice, n_jobs=n_jobs):
            callbacks = [
                
                DeltaYStopper(delta=5e-3, n_best=15),
            ]
            if getattr(args, "time_budget_min", None):
                callbacks.append(DeadlineStopper(total_time=60.0 * float(args.time_budget_min)))
            opt.fit(X_train, y_train, callback=callbacks)

        best_params = dict(opt.best_params_)
        print(f"Skopt best MAE: {-opt.best_score_:.4f}")
        return best_params

    if args.tuner == "raytune":
        import ray
        from ray import tune

        if not ray.is_initialized():
            ray.init(num_cpus=getattr(args, "n_jobs", 1), ignore_reinit_error=True)

        def raytune_objective(config):
            tune.report({"mae": cv_objective(config)})

        ray_space = {}
        for name, spec in search_space.items():
            if not isinstance(spec, (list, tuple)) or len(spec) < 1:
                continue
            t = spec[0]
            if t == "loguniform":
                ray_space[name] = tune.loguniform(float(spec[1]), float(spec[2]))
            elif t == "uniform":
                ray_space[name] = tune.uniform(float(spec[1]), float(spec[2]))
            elif t == "int":
                ray_space[name] = tune.randint(int(spec[1]), int(spec[2]) + 1)
            elif t == "cat":
                ray_space[name] = tune.choice(list(spec[1]))
            else:
                raise ValueError(f"Unsupported space type for {name}: {t}")

        analysis = tune.run(
            raytune_objective,
            config=ray_space,
            num_samples=args.n_trials,
            metric="mae",
            mode="min",
            verbose=1,
        )
        best_params = analysis.best_config
        try:
            mae_best = analysis.best_result.get("mae", None)
            if mae_best is not None:
                print(f"Ray Tune best MAE: {mae_best:.4f}")
        except Exception:
            pass
        ray.shutdown()
        return best_params

    raise ValueError(f"Unknown tuner: {args.tuner}")
