# Purpose: Centralizes all static configurations for models, encodings, and optimizers.

from squlearn.encoding_circuit import (
    HubregtsenEncodingCircuit, ChebyshevPQC, YZ_CX_EncodingCircuit,
    HighDimEncodingCircuit, KyriienkoEncodingCircuit, ParamZFeatureMap, ChebyshevRx
)
from squlearn import optimizers
from squlearn.kernel import QKRR, QSVR, QGPR
from squlearn.qnn import QNNRegressor
from squlearn.qrc import QRCRegressor

# Dictionary to map encoding names to their sQUlearn classes
ENCODING_MAP = {
    "hubregtsen": HubregtsenEncodingCircuit, "chebyshev": ChebyshevPQC,
    "yz_cx": YZ_CX_EncodingCircuit, "highdim": HighDimEncodingCircuit,
    "kyriienko": KyriienkoEncodingCircuit, "paramz": ParamZFeatureMap,
    "chebyshev_rx": ChebyshevRx
}

# Dictionary to map optimizer names to their sQUlearn classes
OPTIMIZER_MAP = {
    "adam": optimizers.Adam, "lbfgsb": optimizers.LBFGSB,
    "spsa": optimizers.SPSA, "slsqp": optimizers.SLSQP
}

# Dictionary defining the QML models and their tunable hyperparameter spaces
MODEL_CONFIG = {
    "qsvr": (QSVR, {"C": ("loguniform", 1e-2, 1e3), "epsilon": ("loguniform", 1e-3, 1e1)}),
    "qkrr": (QKRR, {"alpha": ("loguniform", 1e-6, 1e1)}),
    "qgpr": (QGPR, {"regularization": ("loguniform", 1e-6, 1e1)}),
    "qnn": (QNNRegressor, {"lr": ("loguniform", 1e-4, 1e-1), "epochs": ("int", 20, 100)}),
    "qrcr": (QRCRegressor, {"lr": ("loguniform", 1e-4, 1e-1), "epochs": ("int", 20, 100)})
}
