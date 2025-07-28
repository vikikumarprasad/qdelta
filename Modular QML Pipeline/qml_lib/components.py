# Purpose: Contains factory functions for building PQC, Kernel, and Optimizers.

import numpy as np
from squlearn import Executor, optimizers
from squlearn.kernel import FidelityKernel, ProjectedQuantumKernel
from .config import ENCODING_MAP, OPTIMIZER_MAP # Relative import from this package

def get_pqc(encoding, num_qubits, num_layers, num_features):
    """Builds the Parametrized Quantum Circuit (PQC)."""
    pqc_class = ENCODING_MAP[encoding]
    return pqc_class(num_qubits=num_qubits, num_features=num_features, num_layers=num_layers)

def get_kernel(pqc, kernel_type, param_init, random_seed):
    """Builds the quantum kernel from a PQC."""
    if param_init == 'zeros':
        params = np.zeros(pqc.num_parameters)
    else: # 'random'
        rng = np.random.default_rng(random_seed)
        params = rng.uniform(0, 2 * np.pi, pqc.num_parameters)
    
    executor = Executor()
    kernel_class = FidelityKernel if kernel_type == "fidelity" else ProjectedQuantumKernel
    return kernel_class(encoding_circuit=pqc, executor=executor, initial_parameters=params)

def get_optimizer(name, learning_rate, maxiter=100):
    """Gets an optimizer instance for training."""
    if name in OPTIMIZER_MAP:
        return OPTIMIZER_MAP[name](options={"lr": learning_rate, "maxiter": maxiter})
    
    print(f"Warning: Optimizer '{name}' not found, falling back to Adam.")
    return optimizers.Adam(options={"lr": learning_rate, "maxiter": maxiter})