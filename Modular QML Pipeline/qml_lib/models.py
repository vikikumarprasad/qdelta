# Purpose: Assembles components into a complete, trainable sQUlearn model.

from squlearn import Executor
from squlearn.observables import SummedPaulis
from squlearn.qnn import SquaredLoss, QNNRegressor
from squlearn.qrc import QRCRegressor

from .config import MODEL_CONFIG
from .components import get_pqc, get_kernel, get_optimizer

def create_model_from_params(args, params):
    """Factory function to create a QML model from a dictionary of parameters."""
    if args.reencoding_type == 'parallel':
        num_layers = 1
        num_features = 1
        if params.get("num_layers", 1) > 1:
            print(f"Warning: For 'parallel' re-encoding, layers are forced to 1. Ignoring layers > 1.")
    else: # sequential
        num_layers = params.get("num_layers", args.layers)
        num_features = args.qubits

    # Create the PQC with the determined parameters
    pqc = get_pqc(args.encoding, args.qubits, num_layers, num_features)
    
    model_class, _ = MODEL_CONFIG[args.model]

    if args.model in ["qsvr", "qkrr", "qgpr"]:
        kernel = get_kernel(pqc, args.kernel, args.param_init, args.seed)
        model_args = {"quantum_kernel": kernel}
        
        if args.model == "qsvr":
            model_args.update({"C": params.get("C", 1.0), "epsilon": params.get("epsilon", 0.1)})
        elif args.model == "qkrr":
            model_args.update({"alpha": params.get("alpha", 1e-2)})
        elif args.model == "qgpr":
            model_args.update({"regularization": params.get("regularization", 1e-2)})
        
        model = model_class(**model_args)

        if args.train_kernel:
            model.train_kernel = True
            optimizer_iterations = args.optimizer_iter
            # Use the new --kernel_optimizer argument for full control
            model.optimizer = get_optimizer(args.kernel_optimizer, params.get("lr", 0.01), maxiter=optimizer_iterations)
        return model
    else:
        # The optimizer name is from args, but its learning rate can be tuned
        optimizer = get_optimizer(args.optimizer, params.get("lr", 0.01), maxiter=args.optimizer_iter)
        
        if args.model == "qnn":
            epochs = params.get("epochs", 50) 
            batch_size = params.get("batch_size", 16)
            variance = params.get("variance", 0.01)
            observable = SummedPaulis(num_qubits=args.qubits)
            return QNNRegressor(
                pqc, observable, Executor(), SquaredLoss(), optimizer,
                epochs=epochs, batch_size=batch_size, variance=variance
            )
        elif args.model == "qrcr":
            epochs = params.get("epochs", 50)
            batch_size = params.get("batch_size", 16)
            return QRCRegressor(
                pqc, executor=Executor(), optimizer=optimizer, loss=SquaredLoss(),
                epochs=epochs, batch_size=batch_size
            )
