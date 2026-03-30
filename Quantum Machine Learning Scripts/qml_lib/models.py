# qml_lib/models.py

from squlearn import optimizers
from .config import MODEL_CONFIG, OPTIMIZER_MAP
from .components import get_pqc, get_kernel

import numpy as np
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process.kernels import Kernel as _SKLearnKernel
from sklearn.base import BaseEstimator, RegressorMixin

_PAULI_CACHE    = {}
_SV_CACHE       = {}
_FQK_GRAM_CACHE = {}


def _data_fingerprint(X):
    return hash(X.tobytes())


def _circuit_key(feature_map, random_seed):
    return (
        feature_map.__class__.__name__,
        int(feature_map.num_features),
        int(getattr(feature_map, "num_parameters", 0)),
        int(getattr(feature_map, "num_layers", 1)),
        int(random_seed),
    )


def clear_simulation_cache():
    _PAULI_CACHE.clear()
    _SV_CACHE.clear()
    _FQK_GRAM_CACHE.clear()

class _FastKernelBase(_SKLearnKernel):

    def __init__(self, feature_map, random_seed=42):
        super().__init__()
        self.feature_map = feature_map
        self.random_seed = int(random_seed)

        n_features = int(feature_map.num_features)
        n_params   = int(feature_map.num_parameters)
        x_syms     = ParameterVector("x", n_features)

        if n_params > 0:
            p_syms = ParameterVector("p", n_params)
            raw_qc = feature_map.get_circuit(x_syms, p_syms)
            initializer = getattr(feature_map, "generate_initial_parameters", None)
            if callable(initializer):
                try:
                    weights = np.asarray(initializer(seed=int(random_seed))).flatten()
                except TypeError:
                    weights = np.asarray(initializer()).flatten()
            else:
                rng     = np.random.default_rng(self.random_seed)
                weights = rng.uniform(-np.pi, np.pi, size=n_params)

            self._circuit = raw_qc.assign_parameters(
                {p_syms[i]: float(weights[i]) for i in range(n_params)}
            )
        else:
            self._circuit = feature_map.get_circuit(x_syms, [])

        remaining      = list(self._circuit.parameters)
        name_to_col    = {f"x[{i}]": i for i in range(n_features)}
        self._feat_col = [name_to_col[p.name] for p in remaining]
        self._fparams  = remaining
        self._n_qubits = self._circuit.num_qubits
        self._ckey     = _circuit_key(feature_map, self.random_seed)

    def _bind_sample(self, x_row):
        # assign feature values to the free parameters for one sample
        bind_dict = {
            self._fparams[k]: float(x_row[self._feat_col[k]])
            for k in range(len(self._fparams))
        }
        return self._circuit.assign_parameters(bind_dict)

    @property
    def theta(self):
        return np.array([], dtype=float)

    @theta.setter
    def theta(self, value):
        pass

    @property
    def bounds(self):
        return np.empty((0, 2), dtype=float)

    def is_stationary(self):
        return True

    def get_params(self, deep=True):
        return {"feature_map": self.feature_map, "random_seed": self.random_seed}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# Fidelity kernel: K(x, x') = |<psi(x)|psi(x')>|^2
# All statevectors are computed first, then K is obtained via a single matrix multiply.
class FastFidelityKernel(_FastKernelBase):

    def __init__(self, feature_map, random_seed=42):
        super().__init__(feature_map=feature_map, random_seed=random_seed)

    def _compute_statevectors(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)

        dim  = 2 ** self._n_qubits
        S    = np.empty((X.shape[0], dim), dtype=complex)
        slot = _SV_CACHE.setdefault(self._ckey, {})
        hits = 0

        for i, x_row in enumerate(X):
            key    = x_row.tobytes()
            cached = slot.get(key)
            if cached is not None:
                S[i] = cached
                hits += 1
            else:
                sv        = Statevector(self._bind_sample(x_row)).data
                slot[key] = sv
                S[i]      = sv

        if hits < X.shape[0]:
            print(f"[FastFQK] Simulated {X.shape[0] - hits} new samples "
                  f"({hits} cache hits, cache size={len(slot)})")
        return S

    def compute_gram(self, X):
        # return cached gram matrix if already computed for this fold
        gram_key = (self._ckey, _data_fingerprint(X))
        cached   = _FQK_GRAM_CACHE.get(gram_key)
        if cached is not None:
            return cached

        S = self._compute_statevectors(X)
        K = np.abs(S @ S.conj().T) ** 2
        _FQK_GRAM_CACHE[gram_key] = (S, K)
        print(f"[FastFQK] Gram {K.shape} computed + cached  "
              f"(dim=2^{self._n_qubits}={2**self._n_qubits})")
        return S, K

    def compute_cross(self, X_test, ref_svs):
        S_test = self._compute_statevectors(X_test)
        return np.abs(S_test @ ref_svs.conj().T) ** 2

    def _ensure_ref_cached(self, Y):
        # cache reference statevectors on first GPR call
        if not hasattr(self, "_ref_svs") or self._ref_svs is None:
            self._ref_svs = self._compute_statevectors(Y)
            print(f"[FastFQK] Cached ref statevectors {self._ref_svs.shape} (GPR path)")

    def __call__(self, X, Y=None, eval_gradient=False):
        ref = X if Y is None else Y
        self._ensure_ref_cached(ref)
        S_x = self._compute_statevectors(X)
        K   = np.abs(S_x @ self._ref_svs.conj().T) ** 2
        if eval_gradient:
            return K, np.empty((X.shape[0], X.shape[0], 0))
        return K

    def diag(self, X):
        return np.ones(X.shape[0])

    def clone_with_theta(self, theta):
        return FastFidelityKernel(feature_map=self.feature_map, random_seed=self.random_seed)

    def __repr__(self):
        return f"FastFidelityKernel(n_qubits={self._n_qubits}, dim={2**self._n_qubits})"


# Projected kernel: compute <X>, <Y>, <Z> on each qubit, then apply RBF.
# gamma_scale multiplies the auto-derived RBF bandwidth and is the tunable hyperparameter.
class FastProjectedKernel(_FastKernelBase):

    def __init__(self, feature_map, gamma_scale=1.0, random_seed=42):
        super().__init__(feature_map=feature_map, random_seed=random_seed)
        self.gamma_scale = float(gamma_scale)
        self._paulis     = self._build_paulis(self._n_qubits)
        self._n_paulis   = len(self._paulis)
        self._ref_vecs   = None
        self._gamma      = None

    @staticmethod
    def _build_paulis(n_qubits):
        # one X, Y, Z operator per qubit; qubit 0 is the rightmost in Qiskit's ordering
        paulis = []
        for qubit in range(n_qubits):
            for axis in "XYZ":
                label = ["I"] * n_qubits
                label[n_qubits - 1 - qubit] = axis
                paulis.append(SparsePauliOp("".join(label)))
        return paulis

    def _compute_pauli_vectors(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)

        vecs = np.empty((X.shape[0], self._n_paulis), dtype=float)
        slot = _PAULI_CACHE.setdefault(self._ckey, {})
        hits = 0

        for i, x_row in enumerate(X):
            key    = x_row.tobytes()
            cached = slot.get(key)
            if cached is not None:
                vecs[i] = cached
                hits    += 1
            else:
                sv        = Statevector(self._bind_sample(x_row))
                vec       = np.array([float(sv.expectation_value(p).real)
                                      for p in self._paulis])
                slot[key] = vec
                vecs[i]   = vec

        if hits < X.shape[0]:
            print(f"[FastPQK] Simulated {X.shape[0] - hits} new samples "
                  f"({hits} cache hits, cache size={len(slot)})")
        return vecs

    def _derive_gamma(self, ref_vecs):
        # bandwidth from variance of the training feature vectors, scaled by gamma_scale
        variance   = float(np.var(ref_vecs))
        d          = ref_vecs.shape[1]
        base_gamma = 1.0 / (d * variance) if variance > 1e-9 else 1.0
        return base_gamma * self.gamma_scale

    def compute_gram(self, X):
        ref_vecs    = self._compute_pauli_vectors(X)
        self._gamma = self._derive_gamma(ref_vecs)
        K = rbf_kernel(ref_vecs, ref_vecs, gamma=self._gamma)
        print(f"[FastPQK] Gram {K.shape} computed  |  "
              f"gamma_eff={self._gamma:.4g}  (gamma_scale={self.gamma_scale:.4g})")
        return ref_vecs, K

    def compute_cross(self, X_test, ref_vecs):
        test_vecs = self._compute_pauli_vectors(X_test)
        return rbf_kernel(test_vecs, ref_vecs, gamma=self._gamma)

    def _ensure_ref_cached(self, Y):
        # cache reference vectors and derived gamma on first GPR call
        if self._ref_vecs is not None:
            return
        self._ref_vecs = self._compute_pauli_vectors(Y)
        self._gamma    = self._derive_gamma(self._ref_vecs)
        print(f"[FastPQK] Cached ref set {self._ref_vecs.shape}  |  "
              f"gamma_eff={self._gamma:.4g}  gamma_scale={self.gamma_scale:.4g}  (GPR path)")

    def __call__(self, X, Y=None, eval_gradient=False):
        ref = X if Y is None else Y
        self._ensure_ref_cached(ref)
        x_vecs = self._compute_pauli_vectors(X)
        K      = rbf_kernel(x_vecs, self._ref_vecs, gamma=self._gamma)
        if eval_gradient:
            return K, np.empty((X.shape[0], X.shape[0], 0))
        return K

    def diag(self, X):
        return np.ones(X.shape[0])

    def clone_with_theta(self, theta):
        return FastProjectedKernel(
            feature_map=self.feature_map,
            gamma_scale=self.gamma_scale,
            random_seed=self.random_seed,
        )

    def get_params(self, deep=True):
        return {
            "feature_map": self.feature_map,
            "gamma_scale": self.gamma_scale,
            "random_seed": self.random_seed,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self):
        return (f"FastProjectedKernel(n_qubits={self._n_qubits}, "
                f"n_paulis={self._n_paulis}, gamma_scale={self.gamma_scale:.4g})")

class FastKernelRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        pqc,
        model_type,
        kernel_type,
        gamma=1.0,
        sigma=0.1,
        normalize_y=True,
        C=1.0,
        epsilon=0.1,
        alpha=1e-6,
        random_seed=42,
    ):
        self.pqc         = pqc
        self.model_type  = model_type
        self.kernel_type = kernel_type
        self.gamma       = float(gamma)
        self.sigma       = float(sigma)
        self.normalize_y = bool(normalize_y)
        self.C           = float(C)
        self.epsilon     = float(epsilon)
        self.alpha       = float(alpha)
        self.random_seed = int(random_seed)
        self._inner      = None
        self._kernel_obj = None
        self._ref_data   = None

    def _make_kernel(self):
        if self.kernel_type == "fidelity":
            return FastFidelityKernel(
                feature_map=self.pqc,
                random_seed=self.random_seed,
            )
        elif self.kernel_type == "projected":
            return FastProjectedKernel(
                feature_map=self.pqc,
                gamma_scale=self.gamma,
                random_seed=self.random_seed,
            )
        else:
            raise ValueError(f"Unknown kernel_type '{self.kernel_type}'")

    def fit(self, X, y):
        self._kernel_obj = self._make_kernel()

        if self.model_type in ("qsvr", "qkrr"):
            self._ref_data, K_train = self._kernel_obj.compute_gram(X)

            if self.model_type == "qsvr":
                from sklearn.svm import SVR
                self._inner = SVR(kernel="precomputed", C=self.C, epsilon=self.epsilon)
            else:
                from sklearn.kernel_ridge import KernelRidge
                self._inner = KernelRidge(kernel="precomputed", alpha=self.alpha)

            self._inner.fit(K_train, y)

        else:

            from sklearn.gaussian_process import GaussianProcessRegressor
            self._inner = GaussianProcessRegressor(
                kernel=self._kernel_obj,
                alpha=self.sigma ** 2,
                normalize_y=self.normalize_y,
                optimizer=None,
            )
            self._inner.fit(X, y)

        return self

    def predict(self, X):
        if self.model_type in ("qsvr", "qkrr"):
            K_test = self._kernel_obj.compute_cross(X, self._ref_data)
            return self._inner.predict(K_test)
        else:
            return self._inner.predict(X)

    def __repr__(self):
        return (f"FastKernelRegressor(model={self.model_type!r}, "
                f"kernel={self.kernel_type!r}, gamma={self.gamma:.4g}, "
                f"sigma={self.sigma:.4g}, C={self.C:.4g}, "
                f"epsilon={self.epsilon:.4g}, alpha={self.alpha:.2g})")


def create_model_from_params(args, params):

    if args.model in ["qnn-cpmap", "qnn-iqp"]:
        model_fn, _, extra_args = MODEL_CONFIG[args.model]
        return model_fn(args, params, **extra_args)

    if args.reencoding_type == "parallel":

        num_layers   = 1
        num_features = 1
        if params.get("num_layers", 1) > 1:
            print("Warning: 'parallel' re-encoding forces layers=1; ignoring larger values.")
    else:
        num_layers   = params.get("num_layers", args.layers)
        num_features = int(getattr(args, "input_dim", args.qubits))

    if args.encoding is None and args.model not in ["qnn-cpmap", "qnn-iqp"]:
        raise ValueError(f"You must specify --encoding for model '{args.model}'")

    pqc = get_pqc(args.encoding, args.qubits, num_layers, num_features)
    model_class, _ = MODEL_CONFIG[args.model]

    if args.model in ("qsvr", "qkrr", "qgpr"):

        if args.kernel in ("fidelity", "projected"):
            print(f"[Kernel] Using FastKernelRegressor  "
                  f"kernel={args.kernel}  model={args.model}  "
                  f"path={'precomputed' if args.model in ('qsvr','qkrr') else 'callable-GPR'}")
            return FastKernelRegressor(
                pqc=pqc,
                model_type=args.model,
                kernel_type=args.kernel,
                gamma=float(params.get("gamma", 1.0)),
                sigma=float(params.get("sigma", 0.1)),
                normalize_y=bool(params.get("normalize_y", True)),
                C=float(params.get("C", 1.0)),
                epsilon=float(params.get("epsilon", 0.1)),
                alpha=float(params.get("alpha", 1e-6)),
                random_seed=int(args.seed),
            )

        else:
            kernel_optimizer_instance = None

            if getattr(args, "train_kernel", False):
                opt_lr   = float(params.get("lr", 0.01))
                opt_name = getattr(args, "kernel_optimizer", "adam")
                opt_iter = getattr(args, "kernel_optimizer_iter", 100)
                opt_cls  = OPTIMIZER_MAP.get(opt_name, optimizers.Adam)
                print(f"[Tuner]: Creating kernel optimizer {opt_name}(lr={opt_lr}, iter={opt_iter})")
                kernel_optimizer_instance = opt_cls(options={"lr": opt_lr, "maxiter": opt_iter})

            # temporarily override kernel_gamma if the tuner supplied one
            _orig_kernel_gamma = getattr(args, "kernel_gamma", None)
            if "gamma" in params:
                args.kernel_gamma = float(params["gamma"])

            try:
                kernel = get_kernel(args, pqc, args.kernel, args.param_init, args.seed)
            finally:
                if _orig_kernel_gamma is not None:
                    args.kernel_gamma = _orig_kernel_gamma

            model_kwargs = {"quantum_kernel": kernel}

            if args.model == "qsvr":
                model_kwargs["C"]       = params.get("C", 1.0)
                model_kwargs["epsilon"] = params.get("epsilon", 0.1)
                if kernel_optimizer_instance is not None:
                    print("Warning: train_kernel=true is not supported for QSVR.")

            elif args.model == "qkrr":
                model_kwargs["alpha"] = params.get("alpha", 1e-6)
                if kernel_optimizer_instance is not None:
                    print("Warning: train_kernel=true is not supported for QKRR.")

            else:
                model_kwargs["sigma"]               = params.get("sigma", 1.0)
                model_kwargs["normalize_y"]         = params.get("normalize_y", True)
                model_kwargs["full_regularization"] = params.get("full_regularization", False)
                if kernel_optimizer_instance is not None:
                    model_kwargs["optimizer"] = kernel_optimizer_instance
                elif getattr(args, "train_kernel", False):
                    print("Warning: QGPR train_kernel=true but no optimizer was created.")

            return model_class(**model_kwargs)
