# qml_lib/qnn.py
# Purpose: Defines and builds QNN models with specific feature maps like CPMap and IQP.

from typing import Optional, Type
import numbers
import numpy as np

from qiskit import QuantumCircuit

from squlearn import Executor
from squlearn.qnn import QNNRegressor, SquaredLoss
from squlearn.observables import SummedPaulis
from squlearn.encoding_circuit.encoding_circuit_base import EncodingCircuitBase

from .components import get_optimizer
from .local_kernel import CPKernelWrapper  # CPMap as an EncodingCircuitBase


# QNN with exposed learning rate
class CustomQNNRegressor(QNNRegressor):
    """Expose optimizer learning-rate ('lr') as a tunable param for Grid/Optuna/etc."""

    def get_params(self, deep: bool = True):
        params = super().get_params(deep)
        if hasattr(self.optimizer, "options") and "lr" in self.optimizer.options:
            params["lr"] = self.optimizer.options["lr"]
        return params

    def set_params(self, **params):
        if "lr" in params and hasattr(self.optimizer, "options"):
            self.optimizer.options["lr"] = params.pop("lr")
        super().set_params(**params)
        return self


# IQP Encoding wrapper
class IQPCircuitWrapper(EncodingCircuitBase):
    """
    IQP-style encoding as an EncodingCircuitBase.

    Per layer:
      1) H on all qubits
      2) Single-qubit phase P(π/4 * f_i)
      3) Controlled-phase CP(π/2 * f_i * f_j) for all i<j
      4) H on all qubits

    - Supports both numeric and symbolic features (ParameterExpressions).
    - Enforces: num_qubits == num_features.
    """

    def __init__(self, num_qubits: int, num_features: int, num_layers: int = 1):
        if num_qubits != num_features:
            raise ValueError(
                f"For IQP, features ({num_features}) must equal qubits ({num_qubits})."
            )
        super().__init__(num_qubits=num_qubits, num_features=num_features)
        self.num_layers = int(num_layers)
        self._num_parameters = 0  # encoding has no trainable parameters

    @property
    def num_parameters(self) -> int:
        return self._num_parameters

    # ---- helpers for numeric/symbolic safety ----
    def _as_param(self, x):
        """Keep symbols symbolic; convert plain numerics to float."""
        if isinstance(x, (numbers.Number, np.number)):
            return float(x)
        return x

    def _sym_mul(self, a, b):
        """Multiply numbers or ParameterExpressions safely."""
        if isinstance(a, (numbers.Number, np.number)) and isinstance(
            b, (numbers.Number, np.number)
        ):
            return float(a) * float(b)
        return a * b  # ParameterExpression algebra supported in Qiskit

    # ---- circuit builder ----
    def get_circuit(self, features, parameters):
        if len(features) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {len(features)}"
            )

        n = self.num_qubits
        T_SCALE = np.pi / 4.0   # P(f_i * π/4)
        CS_SCALE = np.pi / 2.0  # CP(f_i f_j * π/2)

        qc = QuantumCircuit(n)
        for _ in range(self.num_layers):
            # 1) H on all qubits
            qc.h(range(n))

            # 2) Single-qubit phases
            for i, f in enumerate(features):
                angle = self._sym_mul(self._as_param(f), T_SCALE)
                qc.p(angle, i)

            # 3) Controlled phases on all pairs
            for i in range(n):
                for j in range(i + 1, n):
                    w_ij = self._sym_mul(self._as_param(features[i]), self._as_param(features[j]))
                    angle = self._sym_mul(w_ij, CS_SCALE)
                    qc.cp(angle, i, j)

            # 4) Final H column
            qc.h(range(n))

        return qc


# Factory to build a QNN model

def create_qnn_model(
    args,
    params: dict,
    encoding_wrapper: Optional[Type[EncodingCircuitBase]] = None,
):
    """
    Build a QNNRegressor with a specified (or inferred) encoding wrapper.

    - Uses args.input_dim (set in pipeline after load_data) for feature dimension.
    - For IQP: enforces num_qubits == num_features.
    - For CPMap (qnn-cpmap): uses your custom CPMap encoding via CPKernelWrapper.
    """

    # Tunables / hyperparameters (defaults can be overridden by search)
    lr = params.get("lr", 0.01)
    epochs = params.get("epochs", 50)
    batch_size = params.get("batch_size", 16)
    variance = params.get("variance", 0.01)
    num_layers = int(params.get("num_layers", getattr(args, "layers", 1)))

    # Dimensions from args
    num_features = int(getattr(args, "input_dim", args.qubits))
    requested_qubits = int(args.qubits)

    # Pick encoding wrapper if not supplied
    model_name = getattr(args, "model", "").lower()
    if encoding_wrapper is None:
        if model_name in ("qnn-iqp", "qnn_iqp"):
            encoding_wrapper = IQPCircuitWrapper
        elif model_name in ("qnn-cpmap", "qnn_cpmap"):
            encoding_wrapper = CPKernelWrapper
        else:
            raise ValueError(
                "Unknown QNN encoding. Pass 'encoding_wrapper' or use model "
                "'qnn-iqp' or 'qnn-cpmap'."
            )

    # Instantiate encoding circuit; tolerate wrappers that ignore num_layers
    # Instantiate the encoding circuit; always pass num_layers
    pqc = encoding_wrapper(
        num_qubits=requested_qubits,
        num_features=num_features,
        num_layers=num_layers,
    )

    # Align to the encoding circuit's actual physical qubits/features
    effective_qubits = int(getattr(pqc, "num_qubits", requested_qubits))
    effective_features = int(getattr(pqc, "num_features", num_features))

    # Optimizer / observable / executor
    optimizer = get_optimizer(args.optimizer, lr, maxiter=args.optimizer_iter)
    observable = SummedPaulis(num_qubits=effective_qubits)

    # Prefer Qiskit executor; fall back if not available
    try:
        execu = Executor("qiskit")
    except Exception:
        try:
            execu = Executor(backend="qiskit")
        except Exception:
            execu = Executor()

    # Build QNN (do NOT pass num_layers here; it's a wrapper property)
    model = CustomQNNRegressor(
        encoding_circuit=pqc,
        operator=observable,
        executor=execu,
        loss=SquaredLoss(),
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        variance=variance,
        # Make dims explicit and consistent with the encoding circuit:
        num_qubits=effective_qubits,
        num_features=effective_features,
    )
    return model
