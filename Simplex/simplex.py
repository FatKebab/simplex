import numpy as np
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
import pandas as pd
import seaborn as sns

# Part 1: Read CSV and Parse Data
def parse_csv(filename):
    # Define column names explicitly while reading the CSV file
    df = pd.read_csv(filename, names=['x', 'y', 'type', 'value'])

    # Identify objective coefficients and maximize/minimize type from the first row
    first_row = df.iloc[0]
    objective_coeffs = [float(first_row['x']), float(first_row['y'])]
    maximize = first_row['type'] == 'max'

    # Parse constraints and bounds from the remaining rows
    constraints = []
    bounds = []
    for _, row in df.iloc[1:].iterrows():
        constraints.append([float(row['x']), float(row['y'])])
        bounds.append(float(row['value']))
    
    return objective_coeffs, constraints, bounds, maximize

# Part 2: Google OR-Tools Method
def solve_lp_with_ortools(objective_coeffs, constraints, bounds, maximize=True):
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        print("GLOP solver not available.")
        return None

    # Define variables
    num_vars = len(objective_coeffs)
    variables = [solver.NumVar(0, solver.infinity(), f'x{i+1}') for i in range(num_vars)]

    # Define objective
    objective = solver.Objective()
    for i, coeff in enumerate(objective_coeffs):
        objective.SetCoefficient(variables[i], coeff)
    if maximize:
        objective.SetMaximization()
    else:
        objective.SetMinimization()

    # Define constraints
    for i, (constraint_coeffs, bound) in enumerate(zip(constraints, bounds)):
        constraint = solver.RowConstraint(-solver.infinity(), bound, f'constraint_{i+1}')
        for j, coeff in enumerate(constraint_coeffs):
            constraint.SetCoefficient(variables[j], coeff)

    # Solve
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        solution = [v.solution_value() for v in variables]
        print("Optimal solution found:", solution)
        
        # Plot a 3D graph for the solution
        plot_3d_graph(objective_coeffs, constraints, bounds, solution)
        return solution
    else:
        print("No optimal solution found.")
        return None

def plot_3d_graph(objective_coeffs, constraints, bounds, solution):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(0, max(bounds), 100)
    y = np.linspace(0, max(bounds), 100)
    X, Y = np.meshgrid(x, y)

    for i, (constraint, bound) in enumerate(zip(constraints, bounds)):
        if len(constraint) >= 2:
            Z = (bound - constraint[0] * X - constraint[1] * Y) if len(constraint) == 2 else np.zeros_like(X)
            ax.plot_surface(X, Y, Z, alpha=0.3, color='blue')

    ax.scatter(*solution, color="red", s=100, label="Optimal Solution")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Objective")
    plt.legend()
    plt.show()

# Part 3: Custom Simplex Method
def simplex(c, A, b):
    num_vars = len(c)
    tableau = np.zeros((len(b) + 1, len(c) + len(b) + 1))
    tableau[:-1, :num_vars] = A
    tableau[:-1, num_vars:num_vars + len(b)] = np.eye(len(b))
    tableau[:-1, -1] = b
    tableau[-1, :num_vars] = -c

    # Basic variable setup
    basic_vars = [f's{i+1}' for i in range(len(b))] + ['Objective']
    row_names = [f'Basic {var}' for var in basic_vars]
    col_names = ["Basic Variable"] + [f'x{i+1}' for i in range(num_vars)] + [f's{i+1}' for i in range(len(b))] + ['RHS']

    def display_tableau(tableau, row_names, col_names, basic_vars):
        # Prepare DataFrame for display
        tableau_df = pd.DataFrame(tableau, index=row_names, columns=col_names[1:])
        tableau_df.insert(0, "Basic Variable", basic_vars)

        print("Current Tableau:")
        print(tableau_df)
        
        # Display numeric part as heatmap
        num_tableau = tableau[:, :-1]  # Exclude RHS for the heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(num_tableau, annot=True, cmap="YlGnBu", cbar=False, fmt=".2f")
        plt.xlabel("Variables")
        plt.ylabel("Constraints")
        plt.show()

    # Simplex algorithm
    while True:
        display_tableau(tableau, row_names, col_names, basic_vars)
        
        # Determine pivot column (most negative indicator in objective row)
        pivot_col = np.argmin(tableau[-1, :-1])
        if tableau[-1, pivot_col] >= 0:
            break

        # Determine pivot row (smallest positive ratio), excluding division by zero
        ratios = np.divide(
            tableau[:-1, -1], tableau[:-1, pivot_col], 
            out=np.full_like(tableau[:-1, -1], np.inf), 
            where=tableau[:-1, pivot_col] > 0
        )
        pivot_row = np.argmin(ratios)

        # Update basic variable name
        basic_vars[pivot_row] = col_names[pivot_col + 1]

        # Perform pivot
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    display_tableau(tableau, row_names, col_names, basic_vars)
    solution = tableau[:-1, -1]
    print("Optimal Solution:", solution)
    return solution

# Load data from CSV and run both methods
filename = 'func.csv'
objective_coeffs, constraints, bounds, maximize = parse_csv(filename)

print("Solution using OR-Tools:")
solve_lp_with_ortools(objective_coeffs, constraints, bounds, maximize)

print("\nSolution using Simplex Method:")
c = np.array(objective_coeffs)
A = np.array(constraints)
b = np.array(bounds)
simplex(c, A, b)
