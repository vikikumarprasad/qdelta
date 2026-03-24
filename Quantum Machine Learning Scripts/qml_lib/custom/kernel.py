# kernel.py
# Defines the CPKernel class which builds the full CPMap quantum circuit
# using C-Map and P-Map subcircuits with the meta-Fibonacci qubit layout.

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector
from .utility import mapping, CPaction
import numpy as np


class CPKernel:
    """Builds the CPMap encoding circuit with trainable kernel parameters."""

    def __init__(self, num_features, reps=1, insert_barriers=False, CP_params=None, CP_last_layer=False):
        """
        Args:
            num_features: number of input features.
            reps: number of times the circuit block is repeated.
            insert_barriers: whether to insert barriers between mapping layers.
            CP_last_layer: whether to apply CP layers on the final mapping layer.
            CP_params: unused, kept for API compatibility.
        """
        self.num_features = num_features
        self.reps = reps
        self.insert_barriers = insert_barriers
        self.CP_last_layer = CP_last_layer

        # symbolic Parameters are used so the kernel angles can be trained
        self.alpha = Parameter('alpha')
        self.beta = Parameter('beta')
        self.gamma = Parameter('gamma')
        self.param1 = Parameter('param1')
        self.param2 = Parameter('param2')
        self.param3 = Parameter('param3')

        # stored as a list so the wrapper can access them by index
        self._kernel_params = [self.alpha, self.beta, self.gamma, self.param1, self.param2, self.param3]

        self.mapping_list = mapping(self.num_features).mapping_list()
        self.qubits = self.mapping_list[0]
        self.params = ParameterVector('X', self.num_features)

    def cmap(self):
        """Returns the 2-qubit C-Map subcircuit inspired by the QCNN convolutional layer."""
        q1 = QuantumRegister(1, 'q1')
        q2 = QuantumRegister(1, 'q2')
        target = QuantumCircuit(q1, q2, name='C-Map')
        target.rz(-np.pi / 2, q2)
        target.cx(q2, q1)
        target.rz(self.alpha, q1)
        target.ry(self.beta, q2)
        target.cx(q1, q2)
        target.ry(self.gamma, q2)
        target.cx(q2, q1)
        target.rz(np.pi / 2, q1)
        return target

    def pmap(self):
        """Returns the 2-qubit P-Map subcircuit inspired by the QCNN pooling layer."""
        q1 = QuantumRegister(1, 'q1')
        q2 = QuantumRegister(1, 'q2')
        target = QuantumCircuit(q1, q2, name='P-Map')
        target.rz(-np.pi / 2, q2)
        target.cx(q2, q1)
        target.rz(self.param1, q1)
        target.ry(self.param2, q2)
        target.cx(q1, q2)
        target.ry(self.param3, q2)
        return target

    def CPMap(self):
        """Builds and returns the full CPKernel circuit by stacking C-Map and P-Map layers."""
        if self.num_features < 2:
            raise ValueError(
                "The CPMap contains 2-local interactions and cannot be "
                f"defined for less than 2 qubits. You provided {self.num_features}."
            )

        if self.num_features == 2:
            print('For 2-dimensional data, CPKernel gets converted to ZFeatureMap')

        mapping_list_ = self.mapping_list
        last_rep = mapping_list_[-1]
        qc = QuantumCircuit(self.qubits, name='CPKernel')

        for rep in range(self.reps):
            shift = 0
            for k in range(len(mapping_list_)):
                # Hadamard and phase encoding for each qubit in this mapping layer
                for i in range(mapping_list_[k]):
                    qc.h(i)
                    qc.p(self.params[i + shift], i)

                if mapping_list_[k] > last_rep:
                    # C-Map followed by P-Map for all non-final layers
                    c_list = CPaction(mapping_list_[k]).cmap_list()
                    for j in range(len(c_list)):
                        qc.append(self.cmap(), c_list[j])
                    p_list = CPaction(mapping_list_[k]).pmap_list()
                    for l in range(len(p_list)):
                        qc.append(self.pmap(), p_list[l])

                elif mapping_list_[k] == last_rep and last_rep > 1 and self.CP_last_layer is True:
                    # optionally applies CP layers on the last mapping layer
                    c_list = CPaction(mapping_list_[k]).cmap_list()
                    for j in range(len(c_list)):
                        qc.append(self.cmap(), c_list[j])
                    p_list = CPaction(mapping_list_[k]).pmap_list()
                    for l in range(len(p_list)):
                        qc.append(self.pmap(), p_list[l])

                else:
                    break

                shift = shift + self.mapping_list[k]

                if self.insert_barriers is True:
                    qc.barrier()

        return qc