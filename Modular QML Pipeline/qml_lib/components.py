# qml_lib/components.py
# Purpose: Contains factory functions for building PQC, Kernel, and Optimizers.

import numpy as np
from squlearn import Executor, optimizers
from squlearn.kernel import FidelityKernel, ProjectedQuantumKernel

def get_pqc(encoding: str, num_qubits: int, num_layers: int, num_features: int):
    """Build and return the requested encoding circuit (PQC)."""
    # Import here to avoid circulars and to pick up your latest ENCODING_MAP
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
        # Prefer circuit-aware initializer if the PQC provides one (e.g., ChebyshevPQC)
        pqc_initializer = getattr(pqc, "generate_initial_parameters", None)

        if callable(pqc_initializer):
            print("--- Using PQC's built-in random parameter initializer. ---")
            try:
                # Try to call with a seed, which is best practice
                params = np.asarray(pqc_initializer(seed=int(random_seed))).flatten()
            except TypeError:
                # If the specific PQC's initializer doesn't accept a seed, call it without one
                print("Initializer does not accept a seed; using unseeded version.")
                params = np.asarray(pqc_initializer()).flatten()
        else:
            # Fallback to the generic uniform distribution if no special method exists
            print("--- Using generic random parameter initializer. ---")
            rng = np.random.default_rng(random_seed)
            params = rng.uniform(0.0, 2.0 * np.pi, n_params)
        
        # Final check to ensure parameter dimensions match the circuit's expectation
        if params.shape[0] != n_params:
            raise ValueError(
                f"Parameter initialization failed: PQC expects {n_params} parameters, "
                f"but the initializer generated {params.shape[0]}."
            )

    else:
        raise ValueError(f"Unknown param_init '{param_init}'. Use 'zeros' or 'random'.")

    # Plain Executor() is the correct API for your installed sQUlearn
    executor = Executor()

    # Choose kernel
    if kernel_type == "fidelity":
        kernel_cls = FidelityKernel
        kernel_kwargs = dict(encoding_circuit=pqc, executor=executor, initial_parameters=params)
    elif kernel_type == "projected":
        # Choose executor backend:
        #   - If args.pqk_backend == "qiskit", prefer AER statevector to avoid PL queuing issues
        #   - If "pennylane", use default Executor()
        #   - If "auto" (default), try Qiskit AER first, else fall back to default
        backend_choice = getattr(args, "pqk_backend", "auto")
        exec_obj = None
        if backend_choice in ("auto", "qiskit"):
            try:
                from qiskit_aer import Aer
                exec_obj = Executor(Aer.get_backend("aer_simulator_statevector"))
            except Exception:
                exec_obj = None
        if exec_obj is None:
            exec_obj = Executor()  # default backend (often PennyLane)

        # sQUlearn PQK example defaults: measurement="XYZ", outer_kernel="gaussian"
        gamma = float(getattr(args, "kernel_gamma", 0.5))
        kernel_cls = ProjectedQuantumKernel
        kernel_kwargs = dict(
            encoding_circuit=pqc,
            executor=exec_obj,
            measurement="XYZ",
            outer_kernel="gaussian",
            initial_parameters=params,
            gamma=gamma,
            regularization="tikhonov",
        )
    else:
        raise ValueError(f"Unknown kernel type '{kernel_type}'. Use 'fidelity' or 'projected'.")

    return kernel_cls(**kernel_kwargs)

def get_optimizer(name: str, learning_rate: float, maxiter: int = 100):
    """Return an sQUlearn optimizer instance."""
    # Import here to ensure we always use the latest registry you maintain
    from .config import OPTIMIZER_MAP
    opt_cls = OPTIMIZER_MAP.get(name)
    if opt_cls is None:
        print(f"Warning: Optimizer '{name}' not found; falling back to Adam.")
        opt_cls = optimizers.Adam
    return opt_cls(options={"lr": float(learning_rate), "maxiter": int(maxiter)})
