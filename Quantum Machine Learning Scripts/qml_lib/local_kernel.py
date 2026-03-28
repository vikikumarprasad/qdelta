# qml_lib/local_kernel.py
# CPKernelWrapper: adapts CPKernel for use as an EncodingCircuitBase in sQUlearn.

from squlearn.encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from .custom.kernel import CPKernel
import numpy as np


# Original CPKernel design defaults — produce a well-structured kernel across all reps.
# Random initialisation drives states toward Haar-random for reps >= 2,
# collapsing all Pauli expectations to zero and making the kernel matrix constant.
_CP_DEFAULT_PARAMS = np.array([
    -np.pi / 3,   # alpha
     np.pi / 6,   # beta
    -np.pi / 9,   # gamma
     np.pi / 7,   # param1
     np.pi / 9,   # param2
    -np.pi / 7,   # param3
])


class CPKernelWrapper(EncodingCircuitBase):

    def __init__(self, num_qubits: int, num_features: int, num_layers: int = 1):
        cpk          = CPKernel(num_features=int(num_features), reps=int(num_layers))
        actual_qubits = int(getattr(cpk, "qubits", num_qubits))

        try:
            self._trainable_params_list = cpk._kernel_params
            num_trainable_params        = len(self._trainable_params_list)
        except AttributeError:
            print("Warning: CPKernel has no '_kernel_params' attribute. Assuming 0 trainable parameters.")
            self._trainable_params_list = []
            num_trainable_params        = 0

        super().__init__(num_qubits=actual_qubits, num_features=int(num_features))

        self._num_parameters = num_trainable_params
        self.num_layers      = int(num_layers)
        self._cpk            = cpk
        self._circuit        = cpk.CPMap()

    @property
    def num_parameters(self) -> int:
        return self._num_parameters

    def generate_initial_parameters(self, seed=None):
        # Always return the design defaults; seed is accepted for API compatibility
        # with sQUlearn's interface but intentionally ignored here.
        return _CP_DEFAULT_PARAMS.copy()

    def get_circuit(self, features, parameters):
        n_feat   = self.num_features
        f_params = self._circuit.parameters[:n_feat]   # feature parameters
        t_params = self._circuit.parameters[n_feat:]   # trainable parameters

        if len(features) != len(f_params):
            raise ValueError(
                f"Feature length ({len(features)}) must match circuit feature "
                f"parameters ({len(f_params)})."
            )
        if len(parameters) != len(t_params):
            raise ValueError(
                f"Trainable parameter length ({len(parameters)}) must match "
                f"kernel parameters ({len(t_params)})."
            )

        param_dict = {}
        for p, val in zip(f_params, features):
            param_dict[p] = val.item() if hasattr(val, "item") else val
        for p, val in zip(t_params, parameters):
            param_dict[p] = val.item() if hasattr(val, "item") else val

        return self._circuit.assign_parameters(param_dict)

    def __repr__(self):
        return (f"CPKernelWrapper(qubits={self.num_qubits}, "
                f"features={self.num_features}, layers={self.num_layers}, "
                f"trainable_params={self.num_parameters})")


def register_custom_components():
    from .config import ENCODING_MAP
    if "cpkernel" not in ENCODING_MAP:
        print("--- Registering custom CPKernel encoding ---")
        ENCODING_MAP["cpkernel"] = CPKernelWrapper
