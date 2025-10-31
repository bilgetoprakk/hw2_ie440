import csv
import math
import matplotlib.pyplot as plt

# --- FILE PATHS ---
# Example file names; replace with your actual file names if different
coordinates_path = 'coordinates_6.csv'
costs_path       = 'costs_6.csv'
demands_path     = 'demand_6.csv'

# The index of the facility selected in the first part (0-based). Facility 6 is index 5.
FACILITY_INDEX_TO_SOLVE = 5 

# --- DATA READING FUNCTIONS ---

def read_coordinates(path):
    """Reads customer coordinates (x, y) from a CSV file."""
    coords = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all((c is None or str(c).strip() == '') for c in row):
                continue
            try:
                # Assumes no header, reads the first two cells as floats
                x = float(row[0]); y = float(row[1])
                coords.append((x, y))
            except (ValueError, IndexError):
                continue # Skip malformed rows
    return coords

def read_demands(path):
    """Reads customer demands (h_j) from a single-column CSV file."""
    demands = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or str(row[0]).strip() == '':
                continue
            try:
                demands.append(float(row[0]))
            except ValueError:
                continue # Skip malformed rows
    return demands

def read_costs(path):
    """Reads transportation costs (c_ij) from a CSV file."""
    rows = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                rows.append([float(x) for x in row])
            except ValueError:
                continue # Skip malformed rows
    return rows

def transpose(mat):
    """Transposes a matrix (list of lists)."""
    return [list(col) for col in zip(*mat)]

# --- READ DATA ---
coordinates_data = read_coordinates(coordinates_path)  # n rows (customer coords)
demands_data     = read_demands(demands_path)          # n rows (customer demands)
costs_rows       = read_costs(costs_path)              # r x c (raw cost matrix)

# --- DIMENSION CHECK & ORIENTATION ---
n = len(coordinates_data)
if n == 0 or len(demands_data) != n:
    raise ValueError(f"Demands or coordinates mismatch: demands={len(demands_data)}, coords={n}")

r = len(costs_rows)
c = len(costs_rows[0])

# Determine cost matrix orientation and reshape to m x n (facility x customer)
if r == n:
    facility_cost_list = transpose(costs_rows)  # m x n
    m_facilities = c
elif c == n:
    facility_cost_list = costs_rows           # m x n
    m_facilities = r
else:
    raise ValueError(f"Unexpected costs.csv dimensions: {r}x{c}. Neither matches customer count n={n}.")

if FACILITY_INDEX_TO_SOLVE >= m_facilities:
    raise IndexError(f"Facility {FACILITY_INDEX_TO_SOLVE + 1} not found. Max facility count is {m_facilities}.")

print(f"Facility Index Used (k): {FACILITY_INDEX_TO_SOLVE + 1}")

# --- CALCULATE WEIGHTS C_kj = h_j * c_ij ---
# Get the cost row for the selected facility (Facility 6, index 5)
facility_k_costs = facility_cost_list[FACILITY_INDEX_TO_SOLVE] 

# Calculate the C_kj weights for the selected facility
Ckj_selected = [facility_k_costs[j] * demands_data[j] for j in range(n)]

# --- SOLUTION FUNCTIONS ---

def euclidean_distance(x1, y1, x2, y2):
    """Calculates the standard Euclidean distance."""
    return math.sqrt((x1 - x2)*2 + (y1 - y2)*2)

def objective_function_euclidean(Ckj, coord, coordinates):
    """Calculates the total weighted Euclidean cost (The Fermat-Weber Problem)."""
    xk, yk = coord
    return sum(
        Ckj[j] * euclidean_distance(xk, yk, xj, yj)
        for j, (xj, yj) in enumerate(coordinates)
    )

def calculate_initial_location(coordinates, Ckj):
    """Calculates the initial location (Center of Gravity) using the Weighted Average formula."""
    total_weight = sum(Ckj)
    if total_weight == 0:
        return (0.0, 0.0)
    
    # Weighted Average Formula (required starting point)
    x = sum(Ckj[j] * coordinates[j][0] for j in range(len(coordinates))) / total_weight
    y = sum(Ckj[j] * coordinates[j][1] for j in range(len(coordinates))) / total_weight
    return (x, y)

def weiszfeld_iteration(current_loc, Ckj, coordinates, tolerance=1e-4, iteration_limit=1000):
    """Applies the iterative Weiszfeld algorithm."""
    x_k, y_k = current_loc
    path = [(x_k, y_k)]

    for iteration in range(iteration_limit):
        numerator_x = 0
        numerator_y = 0
        denominator = 0

        for j, (x_j, y_j) in enumerate(coordinates):
            demand_cost = Ckj[j]
            distance = euclidean_distance(x_k, y_k, x_j, y_j)

            # Handle the case where the current location is exactly on a customer location (division by zero)
            if distance < 1e-10:
                print(f"Warning: Location converged onto customer {j}. Iteration stopped.")
                return (x_j, y_j), path
            
            weight = demand_cost / distance
            numerator_x += weight * x_j
            numerator_y += weight * y_j
            denominator += weight
        
        # Weiszfeld formula
        if denominator == 0:
            break
            
        new_x_k = numerator_x / denominator
        new_y_k = numerator_y / denominator

        # Check for convergence
        if euclidean_distance(x_k, y_k, new_x_k, new_y_k) < tolerance:
            break

        x_k, y_k = new_x_k, new_y_k
        path.append((new_x_k, new_y_k))
        
        cost = objective_function_euclidean(Ckj, (new_x_k, new_y_k), coordinates)
        print(f"Iteration {iteration + 1}: Location = ({new_x_k:.4f}, {new_y_k:.4f}), Cost = {cost:.4f}")
        
    return (new_x_k, new_y_k), path

# --- SOLUTION EXECUTION ---
# 1. Calculate the initial location (Center of Gravity)
initial_location = calculate_initial_location(coordinates_data, Ckj_selected)

print(f"Initial Location (Center of Gravity): ({initial_location[0]:.4f}, {initial_location[1]:.4f})")

# 2. Run Weiszfeld's Algorithm
optimal_loc_weiszfeld, path = weiszfeld_iteration(initial_location, Ckj_selected, coordinates_data)

# 3. Calculate the final cost
optimal_cost = objective_function_euclidean(Ckj_selected, optimal_loc_weiszfeld, coordinates_data)

print("\n--- FINAL RESULTS ---")
print(f"Optimal Location (Facility 6, Weiszfeld): ({optimal_loc_weiszfeld[0]:.6f}, {optimal_loc_weiszfeld[1]:.6f})")
print(f"Total Cost at Optimal Location: {optimal_cost:.4f}")

# --- PLOTTING RESULTS ---
try:
    customer_x = [x for x, _ in coordinates_data]
    customer_y = [y for _, y in coordinates_data]

    plt.figure(figsize=(10, 6))
    plt.scatter(customer_x, customer_y, c='blue', label='Customer Locations', marker='o')
    plt.scatter(optimal_loc_weiszfeld[0], optimal_loc_weiszfeld[1], c='red', 
                label=f'Optimal Location (Facility {FACILITY_INDEX_TO_SOLVE + 1})', 
                marker='x', s=200)
    plt.scatter(initial_location[0], initial_location[1], c='green', 
                label='Initial Location (Weighted Avg.)', marker='o', s=100)
    
    # Plot the iteration path
    for i in range(1, len(path)):
        x_values = [path[i-1][0], path[i][0]]
        y_values = [path[i-1][1], path[i][1]]
        plt.plot(x_values, y_values, 'k--', alpha=0.6)

    # Customer labels
    for i, (x, y) in enumerate(coordinates_data):
        plt.text(x, y, str(i), fontsize=8, ha='right')

    plt.title(f'Optimal Facility Location for Facility {FACILITY_INDEX_TO_SOLVE + 1} (Weiszfeld Alg.)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axhline(0, linewidth=0.5, ls='--')
    plt.axvline(0, linewidth=0.5, ls='--')
    plt.grid(linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()
except Exception as e:
    print(f"\nPlotting skipped due to an error: {e}")