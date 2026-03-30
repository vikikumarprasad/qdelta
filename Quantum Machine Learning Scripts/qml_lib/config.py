# config.py
# Central configuration for models, encodings, and optimizers.

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
from .local_kernel import CPKernelWrapper


class CustomQGPR(QGPR):

    def set_params(self, **params):
        if "alpha" in params and "sigma" not in params:
            params["sigma"] = params.pop("alpha")
        return super().set_params(**params)


# maps encoding name strings to their circuit classes
ENCODING_MAP = {
    "yz_cx": YZ_CX_EncodingCircuit,
    "highdim": HighDimEncodingCircuit,
    "hubregtsen": HubregtsenEncodingCircuit,
    "chebyshev": ChebyshevPQC,
    "multicontrol": MultiControlEncodingCircuit,
    "paramz": ParamZFeatureMap,
    "cpkernel": CPKernelWrapper,
}

# maps optimizer name strings to their sQUlearn classes
OPTIMIZER_MAP = {
    "adam": optimizers.Adam,
    "lbfgsb": optimizers.LBFGSB,
    "spsa": optimizers.SPSA,
    "slsqp": optimizers.SLSQP,
}

# maps model name strings to (estimator class, hyperparameter search space)
MODEL_CONFIG = {
    "qsvr": (
        QSVR,
        {
            "C": ("loguniform", 1e-2, 1e3),
            "epsilon": ("loguniform", 1e-4, 1e-1),
            "gamma": ("loguniform", 1e-3, 1e1),
        },
    ),
    "qkrr": (
        QKRR,
        {
            "alpha": ("loguniform", 1e-6, 1e1),
            "gamma": ("loguniform", 1e-3, 1e1),
        },
    ),
    "qgpr": (
        CustomQGPR,
        {
            "sigma": ("loguniform", 1e-2, 1e1),
            "gamma": ("loguniform", 1e-3, 1e1),
        },
    ),
}
