# Import required Qiskit modules
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer

# Import other useful libraries
import numpy as np
import logging
from typing import List, Tuple, Optional

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantumSudokuSolver:
    def __init__(self, grid_size: int = 2, initial_grid: Optional[List[List[int]]] = None):
        # Initialize the Sudoku solver with a grid size (default is 2x2)
        if grid_size < 2:
            raise ValueError("Grid size must be at least 2")

        self.grid_size = grid_size
        self.cells = grid_size * grid_size
        self.qubits_per_cell = int(np.ceil(np.log2(self.cells)))
        self.total_qubits = self.cells * self.qubits_per_cell

        # Validate and store the initial grid if provided
        if initial_grid is not None:
            self._validate_initial_grid(initial_grid)
            self.initial_grid = initial_grid
        else:
            self.initial_grid = None

        # Create quantum and classical registers
        self.data = QuantumRegister(self.total_qubits, name='cell')
        self.ancilla = QuantumRegister(1, name='anc')
        self.cl = ClassicalRegister(self.total_qubits, name='c')
        self.temp_ancilla = QuantumRegister(1, 'temp')

        # Create the main quantum circuit
        self.qc = QuantumCircuit(self.data, self.ancilla, self.temp_ancilla, self.cl)

        logger.info(f"Initialized {grid_size}x{grid_size} Sudoku solver with {self.total_qubits} qubits")

    def _validate_initial_grid(self, grid: List[List[int]]) -> None:
        # Check if the input grid matches the expected size and values
        if len(grid) != self.grid_size or any(len(row) != self.grid_size for row in grid):
            raise ValueError(f"Grid must be {self.grid_size}x{self.grid_size}")
        for row in grid:
            for val in row:
                if val is not None and (val < 1 or val > self.cells):
                    raise ValueError(f"Values must be between 1 and {self.cells}")

    def initialize_superposition(self) -> None:
        # Put all qubits into superposition and apply Z-gate to ancilla
        self.qc.h(self.data)
        self.qc.h(self.ancilla[0])
        self.qc.z(self.ancilla[0])
        if self.initial_grid is not None:
            self._fix_initial_values()

    def _fix_initial_values(self) -> None:
        # Encode the fixed (known) values of the Sudoku into the circuit
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.initial_grid[i][j] is not None:
                    cell_idx = i * self.grid_size + j
                    value = self.initial_grid[i][j] - 1
                    start = cell_idx * self.qubits_per_cell
                    binary = format(value, f'0{self.qubits_per_cell}b')
                    for k, bit in enumerate(binary):
                        if bit == '1':
                            self.qc.x(self.data[start + k])

    def build_oracle(self) -> QuantumCircuit:
        # Create a quantum oracle circuit that flags invalid Sudoku states
        oracle_data = QuantumRegister(self.total_qubits, name='cell')
        oracle_ancilla = QuantumRegister(1, name='anc')
        oracle_temp_ancilla = QuantumRegister(1, 'temp')
        oracle = QuantumCircuit(oracle_data, oracle_ancilla, oracle_temp_ancilla)

        # Check for duplicate values in rows
        for row in range(self.grid_size):
            for i in range(self.grid_size):
                for j in range(i + 1, self.grid_size):
                    cell1 = row * self.grid_size + i
                    cell2 = row * self.grid_size + j
                    self._check_inequality(oracle, cell1, cell2, oracle_temp_ancilla[0])

        # Check for duplicate values in columns
        for col in range(self.grid_size):
            for i in range(self.grid_size):
                for j in range(i + 1, self.grid_size):
                    cell1 = i * self.grid_size + col
                    cell2 = j * self.grid_size + col
                    self._check_inequality(oracle, cell1, cell2, oracle_temp_ancilla[0])

        return oracle

    def _check_inequality(self, circuit: QuantumCircuit, cell1: int, cell2: int, temp_ancilla_qubit) -> None:
        # XOR-based comparison to detect equality violations
        start1 = cell1 * self.qubits_per_cell
        start2 = cell2 * self.qubits_per_cell
        data_register = circuit.qregs[0]
        ancilla_register = circuit.qregs[1]

        circuit.cx(data_register[start1], temp_ancilla_qubit)
        circuit.cx(data_register[start2], temp_ancilla_qubit)
        circuit.cx(temp_ancilla_qubit, ancilla_register[0])
        circuit.cx(data_register[start2], temp_ancilla_qubit)
        circuit.cx(data_register[start1], temp_ancilla_qubit)

    def diffuser(self) -> QuantumCircuit:
        # Grover diffuser to amplify correct states
        diffuser = QuantumCircuit(self.total_qubits)
        diffuser.h(range(self.total_qubits))
        diffuser.x(range(self.total_qubits))
        diffuser.h(self.total_qubits - 1)
        diffuser.mcx(list(range(self.total_qubits - 1)), self.total_qubits - 1)
        diffuser.h(self.total_qubits - 1)
        diffuser.x(range(self.total_qubits))
        diffuser.h(range(self.total_qubits))
        return diffuser

    def solve(self, num_iterations: Optional[int] = None, shots: int = 1024) -> List[Tuple[List[int], int]]:
        # Solve the Sudoku puzzle using Grover’s algorithm
        if num_iterations is None:
            num_iterations = int(np.sqrt(2**self.total_qubits))

        print("\n✅ Problem:")
        print("Solve a 2x2 Sudoku puzzle where each number 1–4 must appear once in every row and column.")

        print("\n💡 Quantum Solution Strategy:")
        print("1. Put all cell values into quantum superposition.")
        print("2. Apply a quantum oracle to eliminate invalid Sudoku boards.")
        print("3. Use Grover's algorithm to amplify the probability of valid solutions.")
        print("4. Measure and interpret the results.")

        logger.info(f"Starting quantum Sudoku solver with {num_iterations} iterations")

        self.initialize_superposition()
        oracle = self.build_oracle()
        diffuser = self.diffuser()

        # Apply Grover's iterations
        for i in range(num_iterations):
            self.qc.compose(oracle, qubits=self.data[:] + self.ancilla[:] + self.temp_ancilla[:], inplace=True)
            self.qc.compose(diffuser, qubits=self.data[:], inplace=True)
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1} Grover iterations")

        self.qc.measure(self.data, self.cl)

        # Run the quantum circuit on the simulator
        backend = Aer.get_backend('qasm_simulator')
        job = backend.run(self.qc, shots=shots)
        result = job.result()
        counts = result.get_counts()

        solutions = self._interpret_results(counts)
        logger.info(f"Found {len(solutions)} valid solutions")

        return solutions

    def _interpret_results(self, counts: dict) -> List[Tuple[List[int], int]]:
        # Convert measurement results into Sudoku grids
        solutions = []
        for bitstring, count in counts.items():
            grid = []
            for i in range(self.cells):
                start = i * self.qubits_per_cell
                value = int(bitstring[start:start + self.qubits_per_cell], 2) + 1
                grid.append(value)
            if self._is_valid_solution(grid):
                solutions.append((grid, count))
        return sorted(solutions, key=lambda x: x[1], reverse=True)

    def _is_valid_solution(self, grid: List[int]) -> bool:
        # Check if each row and column contains unique values
        for row in range(self.grid_size):
            row_values = grid[row * self.grid_size:(row + 1) * self.grid_size]
            if len(set(row_values)) != self.grid_size:
                return False

        for col in range(self.grid_size):
            col_values = [grid[i * self.grid_size + col] for i in range(self.grid_size)]
            if len(set(col_values)) != self.grid_size:
                return False

        return True

# Utility function to print the Sudoku grid
def print_solution(grid: List[int], grid_size: int) -> None:
    for i in range(grid_size):
        row = grid[i * grid_size:(i + 1) * grid_size]
        print(" ".join(map(str, row)))

# Entry point of the program
def main():
    try:
        solver = QuantumSudokuSolver(grid_size=2)
        solutions = solver.solve()

        print("\n📋 Found Solutions:")
        for grid, count in solutions:
            print(f"\nSolution (count: {count}):")
            print_solution(grid, solver.grid_size)

        print(f"\n✅ Total valid solutions found: {len(solutions)}\n")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
