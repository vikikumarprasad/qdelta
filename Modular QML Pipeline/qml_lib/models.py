# qml_lib/models.py
from squlearn import Executor
from squlearn.observables import SummedPaulis
from squlearn.qnn import SquaredLoss, QNNRegressor
from squlearn.qrc import QRCRegressor

from .config import MODEL_CONFIG
from .components import get_pqc, get_kernel, get_optimizer
from .qnn import create_qnn_model

def create_model_from_params(args, params):
    """Factory: builds an estimator consistent with sQUlearn's API."""

    # Your custom QNN variants (unchanged)
    if args.model in ["qnn-cpmap", "qnn-iqp"]:
        model_fn, _, extra_args = MODEL_CONFIG[args.model]
        return model_fn(args, params, **extra_args)

    # Configure encoding circuit depth/features for non-CPMap/IQP paths
    if args.reencoding_type == "parallel":
        num_layers = 1
        num_features = 1
        if params.get("num_layers", 1) > 1:
            print("Warning: 'parallel' re-encoding forces layers=1; ignoring larger values.")
    else:  # sequential
        num_layers = params.get("num_layers", args.layers)
        num_features = args.qubits

    if args.encoding is None and args.model not in ["qnn-cpmap", "qnn-iqp"]:
        raise ValueError(f"You must specify --encoding for model '{args.model}'")

    pqc = get_pqc(args.encoding, args.qubits, num_layers, num_features)

    # Resolve the estimator class from your MODEL_CONFIG
    model_class, _ = MODEL_CONFIG[args.model]

    # ---- Quantum kernel models ------------------------------------------------
    if args.model in ["qsvr", "qkrr", "qgpr"]:
        # Build quantum kernel via your factory (this is where trainable params belong)
        # If tuner supplied a gamma for PQK, override the default
        if "gamma" in params:
            setattr(args, "kernel_gamma", float(params["gamma"]))
        # build kernel        
        kernel = get_kernel(args, pqc, args.kernel, args.param_init, args.seed)

        if args.model == "qsvr":
            # sQUlearn.kernel.QSVR(quantum_kernel, C, epsilon, **kwargs)
            # Docs: https://squlearn.github.io/modules/generated/squlearn.kernel.QSVR.html
            model = model_class(
                quantum_kernel=kernel,
                C=params.get("C", 1.0),
                epsilon=params.get("epsilon", 0.1),
            )

        elif args.model == "qkrr":
            # sQUlearn.kernel.QKRR(quantum_kernel, alpha=1e-6, **kwargs)
            # Docs: https://squlearn.github.io/modules/generated/squlearn.kernel.QKRR.html
            model = model_class(
                quantum_kernel=kernel,
                alpha=params.get("alpha", 1e-6),
            )

        else:  # qgpr
            # sQUlearn.kernel.QGPR(quantum_kernel, sigma=1.0, normalize_y=True, full_regularization=False, **kwargs)
            # Docs: https://squlearn.github.io/modules/generated/squlearn.kernel.QGPR.html
            model = model_class(
                quantum_kernel=kernel,
                sigma=params.get("sigma", 1.0),
                normalize_y=params.get("normalize_y", True),
                full_regularization=params.get("full_regularization", False),
            )

        # NOTE: If you want kernel-parameter training, handle it inside get_kernel(...)
        # by creating trainable feature maps and exposing their parameters to the tuner.

        return model

    # ---- (Optional) Plain QNN/QRC paths kept for completeness ----------------
    # elif args.model == "qnn":
    #     optimizer = get_optimizer(args.optimizer, params.get("lr", 0.01), maxiter=args.optimizer_iter)
    #     epochs = params.get("epochs", 50)
    #     batch_size = params.get("batch_size", 16)
    #     operator = SummedPaulis(num_qubits=args.qubits)
    #     return QNNRegressor(
    #         encoding_circuit=pqc, operator=operator, executor=Executor(), loss=SquaredLoss(),
    #         optimizer=optimizer, epochs=epochs, batch_size=batch_size, variance=params.get("variance", 0.01)
    #     )

    # elif args.model == "qrcr":
    #     optimizer = get_optimizer(args.optimizer, params.get("lr", 0.01), maxiter=args.optimizer_iter)
    #     return QRCRegressor(
    #         pqc, executor=Executor(), optimizer=optimizer, loss=SquaredLoss(),
    #         epochs=params.get("epochs", 50), batch_size=params.get("batch_size", 16)
    #     )

    raise ValueError(f"Unknown model key: {args.model}")
