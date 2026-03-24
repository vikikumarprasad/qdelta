# components.py
# Factory functions for building PQCs and quantum kernels.

import numpy as np
from squlearn import Executor
from squlearn.kernel import FidelityKernel, ProjectedQuantumKernel


def get_pqc(encoding: str, num_qubits: int, num_layers: int, num_features: int):
    """Looks up the encoding class from the config map and returns an instantiated PQC."""
    from .config import ENCODING_MAP
    try:
        pqc_class = ENCODING_MAP[encoding]
    except KeyError as e:
        raise ValueError(f"Unknown encoding '{encoding}'. Check ENCODING_MAP.") from e
    return pqc_class(num_qubits=num_qubits, num_features=num_features, num_layers=num_layers)


def get_kernel(args, pqc, kernel_type: str, param_init: str, random_seed: int):
    """
    Initializes circuit parameters and constructs the requested quantum kernel.

    Tries Qiskit Aer statevector as the backend and falls back to PennyLane if unavailable.
    Stores the effective backend on args for downstream logging.
    """
    try:
        n_params = pqc.num_parameters
    except AttributeError:
        raise RuntimeError("Encoding circuit does not expose 'num_parameters' needed for init.")

    # initializes parameters as zeros or randomly
    if param_init == "zeros":
        params = np.zeros(n_params)

    elif param_init == "random":
        pqc_initializer = getattr(pqc, "generate_initial_parameters", None)

        if callable(pqc_initializer):
            print("Using PQC's built-in random parameter initializer.")
            try:
                params = np.asarray(pqc_initializer(seed=int(random_seed))).flatten()
            except TypeError:
                print("Initializer does not accept a seed; using unseeded version.")
                params = np.asarray(pqc_initializer()).flatten()
        else:
            print("Using generic random parameter initializer.")
            rng = np.random.default_rng(random_seed)
            params = rng.uniform(0.0, 2.0 * np.pi, n_params)

        if params.shape[0] != n_params:
            raise ValueError(
                f"Parameter initialization failed: PQC expects {n_params} parameters, "
                f"but the initializer generated {params.shape[0]}."
            )
    else:
        raise ValueError(f"Unknown param_init '{param_init}'. Use 'zeros' or 'random'.")

    if kernel_type == "fidelity":
        backend_choice = getattr(args, "pqk_backend", "auto")
        exec_obj = None
        if backend_choice in ("auto", "qiskit"):
            try:
                from qiskit_aer import Aer
                exec_obj = Executor(Aer.get_backend("aer_simulator_statevector"))
                print("[Kernel] Fidelity: using Qiskit Aer statevector backend.")
            except Exception as e:
                print(f"[Kernel] Fidelity: Qiskit Aer not available ({e}); falling back to PennyLane.")

        if exec_obj is None:
            exec_obj = Executor()
            print("[Kernel] Fidelity: using PennyLane backend (fallback).")

        kernel_cls = FidelityKernel
        kernel_kwargs = dict(
            encoding_circuit=pqc,
            executor=exec_obj,
            initial_parameters=params,
        )

    elif kernel_type == "projected":
        requested_backend = getattr(args, "pqk_backend", "auto")
        exec_obj = None
        effective_backend = None

        if requested_backend in ("auto", "qiskit"):
            try:
                from qiskit_aer import Aer
                exec_obj = Executor(Aer.get_backend("aer_simulator_statevector"))
                effective_backend = "qiskit"
                print("[Kernel] Projected: using Qiskit Aer statevector backend.")
            except Exception as e:
                if requested_backend == "qiskit":
                    raise RuntimeError(
                        f"[Kernel] Projected: --pqk-backend qiskit requested, but Aer is unavailable: {e}"
                    ) from e
                print(f"[Kernel] Projected: Qiskit Aer unavailable ({e}); falling back to PennyLane.")

        if exec_obj is None:
            exec_obj = Executor()
            effective_backend = "pennylane"
            print("[Kernel] Projected: using PennyLane backend (fallback).")

        # stores the effective backend on args so the pipeline can log it
        setattr(args, "_effective_pqk_backend", effective_backend)

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