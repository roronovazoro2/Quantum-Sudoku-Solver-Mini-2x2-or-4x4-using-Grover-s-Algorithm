from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import QasmSimulator
import numpy as np
import logging
from typing import List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantumSudokuSolver:
    def __init__(self, grid_size: int = 2, initial_grid: Optional[List[List[int]]] = None):
        """
        Initialize the quantum Sudoku solver.
        
        Args:
            grid_size (int): Size of the Sudoku grid (default: 2 for 2x2)
            initial_grid (List[List[int]], optional): Initial grid configuration
        """
        if grid_size < 2:
            raise ValueError("Grid size must be at least 2")
        
        self.grid_size = grid_size
        self.cells = grid_size * grid_size
        self.qubits_per_cell = int(np.ceil(np.log2(grid_size * grid_size)))
        self.total_qubits = self.cells * self.qubits_per_cell
        
        # Validate initial grid if provided
        if initial_grid is not None:
            self._validate_initial_grid(initial_grid)
            self.initial_grid = initial_grid
        else:
            self.initial_grid = None
        
        # Initialize quantum registers
        self.data = QuantumRegister(self.total_qubits, name='cell')
        self.ancilla = QuantumRegister(1, name='anc')
        self.cl = ClassicalRegister(self.total_qubits, name='c')
        
        # Create quantum circuit
        self.qc = QuantumCircuit(self.data, self.ancilla, self.cl)
        
        logger.info(f"Initialized {grid_size}x{grid_size} Sudoku solver with {self.total_qubits} qubits")
    
    def _validate_initial_grid(self, grid: List[List[int]]) -> None:
        """Validate the initial grid configuration."""
        if len(grid) != self.grid_size or any(len(row) != self.grid_size for row in grid):
            raise ValueError(f"Grid must be {self.grid_size}x{self.grid_size}")
        
        for row in grid:
            for val in row:
                if val is not None and (val < 1 or val > self.grid_size * self.grid_size):
                    raise ValueError(f"Values must be between 1 and {self.grid_size * self.grid_size}")
    
    def initialize_superposition(self) -> None:
        """Initialize all qubits in superposition state."""
        self.qc.h(self.data)
        self.qc.h(self.ancilla[0])
        self.qc.z(self.ancilla[0])  # Put ancilla in |-⟩ state
        
        # If initial grid is provided, fix the corresponding qubits
        if self.initial_grid is not None:
            self._fix_initial_values()
    
    def _fix_initial_values(self) -> None:
        """Fix qubits according to the initial grid configuration."""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.initial_grid[i][j] is not None:
                    cell_idx = i * self.grid_size + j
                    value = self.initial_grid[i][j] - 1  # Convert to 0-based
                    start = cell_idx * self.qubits_per_cell
                    
                    # Convert value to binary and fix qubits
                    binary = format(value, f'0{self.qubits_per_cell}b')
                    for k, bit in enumerate(binary):
                        if bit == '1':
                            self.qc.x(self.data[start + k])
    
    def build_oracle(self) -> QuantumCircuit:
        """Build the oracle that checks Sudoku constraints."""
        oracle = QuantumCircuit(self.data, self.ancilla)
        
        # Check row constraints
        for row in range(self.grid_size):
            for i in range(self.grid_size):
                for j in range(i + 1, self.grid_size):
                    cell1 = row * self.grid_size + i
                    cell2 = row * self.grid_size + j
                    self._check_inequality(oracle, cell1, cell2)
        
        # Check column constraints
        for col in range(self.grid_size):
            for i in range(self.grid_size):
                for j in range(i + 1, self.grid_size):
                    cell1 = i * self.grid_size + col
                    cell2 = j * self.grid_size + col
                    self._check_inequality(oracle, cell1, cell2)
        
        return oracle
    
    def _check_inequality(self, circuit: QuantumCircuit, cell1: int, cell2: int) -> None:
        """Check if two cells have different values using quantum circuit."""
        start1 = cell1 * self.qubits_per_cell
        start2 = cell2 * self.qubits_per_cell
        
        # Create temporary ancilla qubits for comparison
        temp_ancilla = QuantumRegister(1, 'temp')
        circuit.add_register(temp_ancilla)
        
        # Compare each bit position
        for i in range(self.qubits_per_cell):
            circuit.cx(self.data[start1 + i], temp_ancilla[0])
            circuit.cx(self.data[start2 + i], temp_ancilla[0])
            circuit.cx(temp_ancilla[0], self.ancilla[0])
            circuit.cx(self.data[start2 + i], temp_ancilla[0])
            circuit.cx(self.data[start1 + i], temp_ancilla[0])
    
    def diffuser(self) -> QuantumCircuit:
        """Create the Grover diffuser circuit."""
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
        """
        Solve the Sudoku puzzle using Grover's algorithm.
        
        Args:
            num_iterations (int, optional): Number of Grover iterations
            shots (int): Number of measurement shots
            
        Returns:
            List[Tuple[List[int], int]]: List of (solution, count) tuples
        """
        if num_iterations is None:
            num_iterations = int(np.sqrt(2**self.total_qubits))
        
        logger.info(f"Starting quantum Sudoku solver with {num_iterations} iterations")
        
        # Initialize superposition
        self.initialize_superposition()
        
        # Get oracle and diffuser
        oracle = self.build_oracle()
        diffuser = self.diffuser()
        
        # Apply Grover iterations
        for i in range(num_iterations):
            self.qc.append(oracle, self.data[:] + self.ancilla[:])
            self.qc.append(diffuser, self.data[:])
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1} Grover iterations")
        
        # Measure the results
        self.qc.measure(self.data, self.cl)
        
        # Run the circuit
        backend = QasmSimulator()
        job = execute(self.qc, backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        solutions = self._interpret_results(counts)
        logger.info(f"Found {len(solutions)} valid solutions")
        
        return solutions
    
    def _interpret_results(self, counts: dict) -> List[Tuple[List[int], int]]:
        """Interpret the measurement results."""
        solutions = []
        for bitstring, count in counts.items():
            # Convert bitstring to grid values
            grid = []
            for i in range(self.cells):
                start = i * self.qubits_per_cell
                value = int(bitstring[start:start + self.qubits_per_cell], 2) + 1
                grid.append(value)
            
            # Check if the solution is valid
            if self._is_valid_solution(grid):
                solutions.append((grid, count))
        
        return sorted(solutions, key=lambda x: x[1], reverse=True)  # Sort by count
    
    def _is_valid_solution(self, grid: List[int]) -> bool:
        """Check if a solution satisfies all Sudoku constraints."""
        # Check rows
        for row in range(self.grid_size):
            row_values = grid[row * self.grid_size:(row + 1) * self.grid_size]
            if len(set(row_values)) != self.grid_size:
                return False
        
        # Check columns
        for col in range(self.grid_size):
            col_values = [grid[i * self.grid_size + col] for i in range(self.grid_size)]
            if len(set(col_values)) != self.grid_size:
                return False
        
        return True

def print_solution(grid: List[int], grid_size: int) -> None:
    """Print a Sudoku solution in a readable format."""
    for i in range(grid_size):
        row = grid[i * grid_size:(i + 1) * grid_size]
        print(" ".join(map(str, row)))

def main():
    try:
        # Create a 2x2 Sudoku solver
        solver = QuantumSudokuSolver(grid_size=2)
        
        # Solve the puzzle
        solutions = solver.solve()
        
        # Print results
        print("\nFound solutions:")
        for grid, count in solutions:
            print(f"\nSolution (count: {count}):")
            print_solution(grid, solver.grid_size)
            print()
            
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 