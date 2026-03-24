# utility.py
# Helper classes for CPKernel circuit construction and qubit mapping.


class MetaFibonacci:
    """Generates the meta-Fibonacci sequence and maps feature counts to qubit counts."""

    def __init__(self, n):
        self.n = n + 1
        self.sequence = [1, 1]
        self._generate_sequence()

    def _generate_sequence(self):
        # extends the sequence up to the nth term using the meta-Fibonacci recurrence
        for i in range(3, self.n + 1):
            next_term = self.sequence[i - self.sequence[i - 3] - 2] + self.sequence[i - self.sequence[i - 2] - 1]
            self.sequence.append(next_term)

    def get_sequence(self):
        """Returns the meta-Fibonacci sequence starting from index 1."""
        return self.sequence[1:]

    def num_qubits(self):
        """Returns the number of qubits corresponding to the given feature count."""
        number_of_qubits = self.sequence[1:]
        return number_of_qubits[self.n - 2]


class mapping:
    """Computes the layered qubit mapping list for a given feature count."""

    def __init__(self, num_features):
        self.num_features = num_features
        self.qubits = MetaFibonacci(self.num_features).num_qubits()

    def mapping_list(self):
        """Returns a list of qubit counts per mapping layer that sums to num_features."""
        m = self.qubits
        encoding_list = [m]
        while sum(encoding_list) < self.num_features:
            diff = self.num_features - sum(encoding_list)
            m //= 2
            encoding_list.append(min(m, diff))
        return encoding_list


class CPaction:
    """Generates qubit pair lists for the C-Map and P-Map layers."""

    def __init__(self, qubits):
        self.qubits = qubits

    def cmap_list(self):
        """Returns the ordered list of qubit pairs for the C-Map layer."""
        q = self.qubits
        input_list_ = []
        if q > 2:
            for i in range(q):
                input_list_.append([i, i + 1])
        else:
            input_list_.append([0, 1])
        input_list = input_list_[:-1]
        even_list = [input_list[i] for i in range(len(input_list)) if i % 2 == 0]
        odd_list = [input_list[i] for i in range(len(input_list)) if i % 2 != 0]
        c_list = [*even_list, *odd_list, *[[q - 1, 0]]]
        return c_list

    def pmap_list(self):
        """Returns the list of qubit pairs for the P-Map layer."""
        q = self.qubits
        p_list = []
        for i in range(q // 2):
            p_list.append([i, i + q - q // 2])
        return p_list