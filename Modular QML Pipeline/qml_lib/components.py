# qml_lib/components.py
# Purpose: Contains factory functions for building PQC, Kernel, and Optimizers.

import numpy as np
from squlearn import Executor, optimizers
from squlearn.kernel import FidelityKernel, ProjectedQuantumKernel

def get_pqc(encoding: str, num_qubits: int, num_layers: int, num_features: int):
    """Build and return the requested encoding circuit (PQC)."""
    from .config import ENCODING_MAP
    try:
        pqc_class = ENCODING_MAP[encoding]
    except KeyError as e:
        raise ValueError(f"Unknown encoding '{encoding}'. Check ENCODING_MAP.") from e
    return pqc_class(num_qubits=num_qubits, num_features=num_features, num_layers=num_layers)

def get_kernel(args, pqc, kernel_type: str, param_init: str, random_seed: int):
    """Construct a quantum kernel from a PQC."""
    # Parameter initialization
    try:
        n_params = pqc.num_parameters
    except AttributeError:
        # Some encodings expose a different attribute; fail clearly
        raise RuntimeError("Encoding circuit does not expose 'num_parameters' needed for init.")

    if param_init == "zeros":
        params = np.zeros(n_params)
    elif param_init == "random":
        rng = np.random.default_rng(random_seed)
        params = rng.uniform(0.0, 2.0 * np.pi, n_params)
    else:
        raise ValueError(f"Unsupported param_init '{param_init}'. Use 'random' or 'zeros'.")

    # Plain Executor() is the correct API for your installed sQUlearn
    executor = Executor()

    # Choose kernel
    if kernel_type == "fidelity":
        kernel_cls = FidelityKernel
        kernel_kwargs = dict(encoding_circuit=pqc, executor=executor, initial_parameters=params)
    elif kernel_type == "projected":
        kernel_cls = ProjectedQuantumKernel
        # 'regularization'='tikhonov' is the setting you wanted for projected kernels
        kernel_kwargs = dict(
            encoding_circuit=pqc,
            executor=executor,
            initial_parameters=params,
            regularization="tikhonov",
        )
    else:
        raise ValueError(f"Unknown kernel type '{kernel_type}'. Use 'fidelity' or 'projected'.")

    return kernel_cls(**kernel_kwargs)

def get_optimizer(name: str, learning_rate: float, maxiter: int = 100):
    """Return an sQUlearn optimizer instance."""
    from .config import OPTIMIZER_MAP
    opt_cls = OPTIMIZER_MAP.get(name)
    if opt_cls is None:
        print(f"Warning: Optimizer '{name}' not found; falling back to Adam.")
        opt_cls = optimizers.Adam
    return opt_cls(options={"lr": float(learning_rate), "maxiter": int(maxiter)})
