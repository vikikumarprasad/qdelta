# qml_lib/kernel.py

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector
from .utility import mapping, CPaction
import numpy as np


class CPKernel:

    def __init__(self, num_features, reps=1, insert_barriers=False, CP_params=None, CP_last_layer=False):
        self.num_features    = num_features
        self.reps            = reps
        self.insert_barriers = insert_barriers
        self.CP_last_layer   = CP_last_layer

        if CP_params is None:
            CP_params = [-np.pi/3, np.pi/6, -np.pi/9, np.pi/7, np.pi/9, -np.pi/7]

        self.alpha  = Parameter("alpha")
        self.beta   = Parameter("beta")
        self.gamma  = Parameter("gamma")
        self.param1 = Parameter("param1")
        self.param2 = Parameter("param2")
        self.param3 = Parameter("param3")
        self._kernel_params = [self.alpha, self.beta, self.gamma,
                               self.param1, self.param2, self.param3]

        self.mapping_list = mapping(self.num_features).mapping_list()
        self.qubits       = self.mapping_list[0]
        self.params       = ParameterVector("X", self.num_features)

    def cmap(self):
        q1 = QuantumRegister(1, "q1")
        q2 = QuantumRegister(1, "q2")
        target = QuantumCircuit(q1, q2, name="C-Map")
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
        q1 = QuantumRegister(1, "q1")
        q2 = QuantumRegister(1, "q2")
        target = QuantumCircuit(q1, q2, name="P-Map")
        target.rz(-np.pi / 2, q2)
        target.cx(q2, q1)
        target.rz(self.param1, q1)
        target.ry(self.param2, q2)
        target.cx(q1, q2)
        target.ry(self.param3, q2)
        return target

    def CPMap(self):
        if self.num_features < 2:
            raise ValueError(
                "The CPMap contains 2-local interactions and cannot be "
                f"defined for less than 2 qubits. You provided {self.num_features}."
            )
        if self.num_features == 2:
            print("For 2-dimensional data, CPKernel gets converted to ZFeatureMap")

        mapping_list_ = self.mapping_list
        last_rep      = mapping_list_[-1]
        qc            = QuantumCircuit(self.qubits, name="CPKernel")

        for rep in range(self.reps):
            shift = 0
            for k in range(len(mapping_list_)):

                for i in range(mapping_list_[k]):
                    qc.h(i)
                    qc.p(self.params[i + shift], i)

                if mapping_list_[k] > last_rep:

                    for j in CPaction(mapping_list_[k]).cmap_list():
                        qc.append(self.cmap(), j)
                    for l in CPaction(mapping_list_[k]).pmap_list():
                        qc.append(self.pmap(), l)

                elif mapping_list_[k] == last_rep and last_rep > 1 and self.CP_last_layer:

                    for j in CPaction(mapping_list_[k]).cmap_list():
                        qc.append(self.cmap(), j)
                    for l in CPaction(mapping_list_[k]).pmap_list():
                        qc.append(self.pmap(), l)

                else:
                    break

                shift += self.mapping_list[k]
                if self.insert_barriers:
                    qc.barrier()

        return qc
