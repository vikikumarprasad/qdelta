# qml_lib/config.py
# Purpose: Centralizes all static configurations for models, encodings, and optimizers.

from squlearn.encoding_circuit import (
    HubregtsenEncodingCircuit,
    ChebyshevPQC,
    YZ_CX_EncodingCircuit,
    HighDimEncodingCircuit,
    MultiControlEncodingCircuit,
    ParamZFeatureMap,
)
from squlearn import optimizers
from squlearn.kernel import QKRR, QSVR, QGPR
# from squlearn.qrc import QRCRegressor  # (unused now, keep comment if you plan to revive)

from .qnn import create_qnn_model, CPKernelWrapper, IQPCircuitWrapper


class CustomQGPR(QGPR):
    """
    Shim so GridSearchCV/our tuners can set params reliably.
    Also maps legacy 'alpha' to the correct 'sigma'.
    """
    def set_params(self, **params):
        # Back-compat: if someone passes alpha, treat it as sigma.
        if "alpha" in params and "sigma" not in params:
            params["sigma"] = params.pop("alpha")
        return super().set_params(**params)


# ---- Name -> Encoding class map (exact 6 requested + your customs) ----
ENCODING_MAP = {
    "yz_cx": YZ_CX_EncodingCircuit,
    "highdim": HighDimEncodingCircuit,
    "hubregtsen": HubregtsenEncodingCircuit,
    "chebyshev": ChebyshevPQC,
    "multicontrol": MultiControlEncodingCircuit,
    "paramz": ParamZFeatureMap,
    # custom (leave as-is)
    "cpkernel": CPKernelWrapper,
    "iqp": IQPCircuitWrapper,
}

# ---- Optimizer name -> class map ----
OPTIMIZER_MAP = {
    "adam": optimizers.Adam,
    "lbfgsb": optimizers.LBFGSB,
    "spsa": optimizers.SPSA,
    "slsqp": optimizers.SLSQP,
}

# ---- Model registry & their tuning spaces ----
# Each entry: model_key -> (callable_or_class, search_space, [optional extra_kwargs])
MODEL_CONFIG = {
    # Kernel models
    "qsvr": (
        QSVR,
        {
          "C": ("loguniform", 1e-2, 1e3),
          "epsilon": ("loguniform", 1e-4, 1e-1),
          # PQK outer Gaussian kernel bandwidth (ignored if kernel != 'projected')
          "gamma": ("loguniform", 1e-3, 1e1),
        },
    ),
    "qkrr": (
        QKRR,
        {
          "alpha": ("loguniform", 1e-6, 1e1),
          # PQK outer Gaussian kernel bandwidth
          "gamma": ("loguniform", 1e-3, 1e1),
        },
    ),
    "qgpr": (
        CustomQGPR,
        {
          "sigma": ("loguniform", 1e-2, 1e1),
          # PQK outer Gaussian kernel bandwidth
          "gamma": ("loguniform", 1e-3, 1e1),
        },
    ),

    # QNNs with fixed feature maps (wrappers provide the PQC)
    "qnn-cpmap": (
        create_qnn_model,
        {
            "lr": ("loguniform", 1e-4, 1e-1),
            "epochs": ("int", 80, 200),
            "batch_size": ("int", 16, 64),
            "num_layers": ("int", 1, 2),
            # optionally: "variance": ("loguniform", 1e-4, 1e-1),
        },
        {"encoding_wrapper": CPKernelWrapper},
    ),
    "qnn-iqp": (
        create_qnn_model,
        {
            "lr": ("loguniform", 5e-4, 2e-1),
            "epochs": ("int", 60, 200),
            "batch_size": ("int", 16, 64),
            "num_layers": ("int", 1, 2),
            # optionally: "variance": ("loguniform", 1e-4, 1e-1),
        },
        {"encoding_wrapper": IQPCircuitWrapper},
    ),
}
