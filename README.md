# Quantum Sudoku Solver

This project implements a quantum Sudoku solver using Grover's algorithm. It can solve 2x2 Sudoku puzzles using quantum computing principles.

## Requirements

- Python 3.7+
- Qiskit
- NumPy

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the solver:
```bash
python quantum_sudoku.py
```

The program will:
1. Create a 2x2 Sudoku solver
2. Apply Grover's algorithm to find valid solutions
3. Print all found solutions with their measurement counts

## How it Works

The solver uses the following quantum computing concepts:
- Quantum superposition to explore all possible solutions simultaneously
- Grover's algorithm to amplify the probability of correct solutions
- Quantum oracles to check Sudoku constraints
- Quantum measurement to obtain classical results

## Implementation Details

- Each cell in the 2x2 grid is encoded using 2 qubits (allowing values 1-4)
- The total circuit uses 8 qubits for the grid cells
- An additional ancilla qubit is used for the oracle
- The oracle checks row and column constraints
- Grover's algorithm is applied to amplify correct solutions

## Limitations

- Currently only supports 2x2 Sudoku puzzles
- The number of qubits grows quadratically with grid size
- Simulation is limited by classical computing resources 