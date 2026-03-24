# local_kernel.py
# Wraps CPKernel as an sQUlearn EncodingCircuitBase so it can be used in kernel models.
# Also provides a registration hook to add it to the global encoding map.

from squlearn.encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from .custom.kernel import CPKernel
import numpy as np


class CPKernelWrapper(EncodingCircuitBase):
    """
    Adapts CPKernel for use as an sQUlearn encoding circuit.
    Separates feature parameters (data-bound) from trainable kernel parameters.
    """

    def __init__(self, num_qubits: int, num_features: int, num_layers: int = 1):
        # builds the CPKernel to determine the actual physical qubit count
        cpk = CPKernel(num_features=int(num_features), reps=int(num_layers))
        actual_qubits = int(getattr(cpk, "qubits", num_qubits))

        try:
            self._trainable_params_list = cpk._kernel_params
            num_trainable_params = len(self._trainable_params_list)
        except AttributeError:
            print("Warning: CPKernel has no '_kernel_params' attribute. Assuming 0 trainable parameters.")
            self._trainable_params_list = []
            num_trainable_params = 0

        super().__init__(num_qubits=actual_qubits, num_features=int(num_features))

        self._num_parameters = num_trainable_params
        self.num_layers = int(num_layers)
        self._cpk = cpk
        self._circuit = cpk.CPMap()

    @property
    def num_parameters(self) -> int:
        """Returns the number of trainable kernel parameters."""
        return self._num_parameters

    def get_circuit(self, features, parameters):
        """Binds feature values and trainable parameters to the CPMap circuit and returns it."""
        n_feature_params = self.num_features
        feature_params_in_circuit = self._circuit.parameters[:n_feature_params]

        if len(features) != len(feature_params_in_circuit):
            raise ValueError(
                f"Feature length ({len(features)}) must match circuit feature parameters ({len(feature_params_in_circuit)})."
            )

        param_dict = {}
        for i in range(len(feature_params_in_circuit)):
            val = features[i]
            if hasattr(val, "item"):
                try:
                    val = val.item()
                except Exception:
                    pass
            param_dict[feature_params_in_circuit[i]] = val

        trainable_params_in_circuit = self._circuit.parameters[n_feature_params:]

        if len(parameters) != len(trainable_params_in_circuit):
            raise ValueError(
                f"Trainable parameter length ({len(parameters)}) must match kernel parameters ({len(trainable_params_in_circuit)})."
            )

        for i in range(len(trainable_params_in_circuit)):
            val = parameters[i]
            if hasattr(val, "item"):
                try:
                    val = val.item()
                except Exception:
                    pass
            param_dict[trainable_params_in_circuit[i]] = val

        return self._circuit.assign_parameters(param_dict)

    def __repr__(self):
        return (
            f"CPKernelWrapper(qubits={self.num_qubits}, "
            f"features={self.num_features}, layers={self.num_layers}, "
            f"trainable_params={self.num_parameters})"
        )


def register_custom_components():
    """Adds CPKernelWrapper to the global encoding map if not already present."""
    from .config import ENCODING_MAP
    if "cpkernel" not in ENCODING_MAP:
        print("--- Registering custom CPKernel encoding ---")
        ENCODING_MAP["cpkernel"] = CPKernelWrapper