# models.py
# Factory function for building QSVR, QKRR, and QGPR estimators.

from squlearn import optimizers
from .config import MODEL_CONFIG, OPTIMIZER_MAP
from .components import get_pqc, get_kernel


def create_model_from_params(args, params):
    """
    Builds and returns a kernel-based quantum estimator using the given parameters.

    Handles PQC construction, kernel initialization, optional kernel optimizer setup,
    and per-model hyperparameter assignment.
    """
    # sets circuit depth and feature count based on the re-encoding strategy
    if args.reencoding_type == "parallel":
        num_layers = 1
        num_features = 1
        if params.get("num_layers", 1) > 1:
            print("Warning: 'parallel' re-encoding forces layers=1; ignoring larger values.")
    else:
        num_layers = params.get("num_layers", args.layers)
        num_features = int(getattr(args, "input_dim", args.qubits))

    if args.encoding is None:
        raise ValueError(f"You must specify --encoding for model '{args.model}'")

    pqc = get_pqc(args.encoding, args.qubits, num_layers, num_features)

    model_class, _ = MODEL_CONFIG[args.model]

    # builds a kernel optimizer instance if kernel training is enabled
    kernel_optimizer_instance = None
    if getattr(args, "train_kernel", False):
        opt_lr = float(params.get("lr", 0.01))
        opt_name = getattr(args, "kernel_optimizer", "adam")
        opt_iter = getattr(args, "kernel_optimizer_iter", 100)
        opt_cls = OPTIMIZER_MAP.get(opt_name, optimizers.Adam)
        print(f"[Tuner]: Creating kernel optimizer {opt_name}(lr={opt_lr}, iter={opt_iter})")
        kernel_optimizer_instance = opt_cls(options={"lr": opt_lr, "maxiter": opt_iter})

    # temporarily overrides kernel_gamma with any tuned value before building the kernel
    _orig_kernel_gamma = getattr(args, "kernel_gamma", None)
    if "gamma" in params:
        args.kernel_gamma = float(params["gamma"])

    try:
        kernel = get_kernel(args, pqc, args.kernel, args.param_init, args.seed)
    finally:
        if _orig_kernel_gamma is not None:
            args.kernel_gamma = _orig_kernel_gamma

    if args.kernel == "projected":
        eff_backend = getattr(args, "_effective_pqk_backend", getattr(args, "pqk_backend", "auto"))
        print(f"[Kernel] Projected effective backend: {eff_backend}")

    model_kwargs = {"quantum_kernel": kernel}

    if args.model == "qsvr":
        model_kwargs["C"] = params.get("C", 1.0)
        model_kwargs["epsilon"] = params.get("epsilon", 0.1)
        if kernel_optimizer_instance is not None:
            print("Warning: train_kernel=true is not supported for QSVR. Kernel parameters will not be trained.")

    elif args.model == "qkrr":
        model_kwargs["alpha"] = params.get("alpha", 1e-6)
        if kernel_optimizer_instance is not None:
            print("Warning: train_kernel=true is not supported for QKRR. Kernel parameters will not be trained.")

    else:  # qgpr
        model_kwargs["sigma"] = params.get("sigma", 1.0)
        model_kwargs["normalize_y"] = params.get("normalize_y", True)
        model_kwargs["full_regularization"] = params.get("full_regularization", False)
        if kernel_optimizer_instance is not None:
            model_kwargs["optimizer"] = kernel_optimizer_instance
        elif getattr(args, "train_kernel", False):
            print("Warning: QGPR train_kernel=true but no optimizer was created.")

    return model_class(**model_kwargs)