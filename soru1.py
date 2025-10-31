import csv
import math
import matplotlib.pyplot as plt

# --- FILE PATHS ---
coordinates_path = 'coordinates_6.csv'
costs_path       = 'costs_6.csv'
demands_path     = 'demand_6.csv'

# Index of the specific facility to be solved (0-based)
# Facility 6 corresponds to index 5
FACILITY_INDEX_TO_SOLVE = 5

# --- DATA READING FUNCTIONS ---

def read_coordinates(path):
    """Reads customer coordinates (x, y) from a CSV file."""
    coords = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip empty rows or rows with only empty cells
            if not row or all((c is None or str(c).strip() == '') for c in row):
                continue
            # Assume no header, read first two columns as floats
            try:
                x = float(row[0]); y = float(row[1])
                coords.append((x, y))
            except (ValueError, IndexError):
                # Simple error handling for non-numeric/incomplete rows
                print(f"Skipping malformed coordinate row: {row}")
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
                print(f"Skipping malformed demand row: {row}")
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
            except ValueError as e:
                # Basic handling for non-numeric cells
                print(f"Skipping malformed cost row due to non-numeric data: {row}")
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
    raise ValueError(f"Mismatch or empty data: Demands={len(demands_data)}, Coords={n}")

r = len(costs_rows)
if r == 0:
    raise ValueError("Costs file appears empty.")
c = len(costs_rows[0])

# Determine cost matrix orientation (m x n: facility x customer)
if r == n:
    # Orientation: n x m (customer x facility) -> Transpose to m x n
    facility_cost_list = transpose(costs_rows)  # m x n
    m_facilities = c
    print(f"Cost orientation: n x m found -> Transposed. m={m_facilities}, n={n}")
elif c == n:
    # Orientation: m x n (facility x customer) -> Use directly
    facility_cost_list = costs_rows           # m x n
    m_facilities = r
    print(f"Cost orientation: m x n found -> Used directly. m={m_facilities}, n={n}")
else:
    raise ValueError(
        f"Unexpected costs.csv dimensions: {r}x{c}. Neither row nor column count matches customer count n={n}."
    )

# Safety check: Ensure all rows have length n
for idx, row in enumerate(facility_cost_list):
    if len(row) != n:
        raise ValueError(f"Costs matrix row {idx} length {len(row)} != n={n}")
        
if FACILITY_INDEX_TO_SOLVE >= m_facilities:
    raise IndexError(f"Facility index {FACILITY_INDEX_TO_SOLVE + 1} is out of range. Max facility count is {m_facilities}.")

print(f"Detected dimensions -> Number of Facilities (m) = {m_facilities}, Number of Customers (n) = {n}")

# --- CALCULATE WEIGHTS C[i][j] = h_j * c_ij ---
# Calculate C matrix (m x n) for all facilities first
C = [[facility_cost_list[i][j] * demands_data[j] for j in range(n)]
     for i in range(m_facilities)]

# Get the specific C_kj list for the chosen facility (Facility 6, index 5)
Ckj_selected = C[FACILITY_INDEX_TO_SOLVE]

# --- OBJECTIVE FUNCTION & OPTIMAL LOCATION ---

def squared_euclidean_distance(x1, y1, x2, y2):
    """Calculates the squared Euclidean distance."""
    return (x1 - x2)**2 + (y1 - y2)**2

def objective_function(Ckj, coord, coordinates):
    """Calculates the total weighted squared Euclidean cost."""
    xk, yk = coord
    return sum(
        Ckj[j] * squared_euclidean_distance(xk, yk, xj, yj)
        for j, (xj, yj) in enumerate(coordinates)
    )

def optimal_location_from_gradient(coordinates, Ckj):
    """
    Calculates the optimal location (center of gravity)
    using the analytic solution for the squared Euclidean distance.
    """
    total_weight = sum(Ckj)
    if total_weight == 0:
        # Fallback: simple average location if weights sum to zero
        if not coordinates:
            return (0.0, 0.0)
        x = sum(x for x, _ in coordinates) / len(coordinates)
        y = sum(y for _, y in coordinates) / len(coordinates)
        return (x, y)
    
    # Analytic solution: Weighted Average
    optimal_x = sum(Ckj[j] * coordinates[j][0] for j in range(len(coordinates))) / total_weight
    optimal_y = sum(Ckj[j] * coordinates[j][1] for j in range(len(coordinates))) / total_weight
    return (optimal_x, optimal_y)

# --- SOLVE FOR FACILITY 6 ---
optimal_loc_6 = optimal_location_from_gradient(coordinates_data, Ckj_selected)
objective_value_6 = objective_function(Ckj_selected, optimal_loc_6, coordinates_data)

print("\n--- RESULTS FOR FACILITY 6 ---")
print(f"Optimal Location: ({optimal_loc_6[0]:.6f}, {optimal_loc_6[1]:.6f})")
print(f"Minimum Objective Function Value: {objective_value_6:.6f}")

# --- PLOTTING ---
try:
    customer_x = [x for x, _ in coordinates_data]
    customer_y = [y for _, y in coordinates_data]

    plt.figure(figsize=(10, 6))
    plt.scatter(customer_x, customer_y, c='blue', label='Customer Locations', marker='o')
    plt.scatter(optimal_loc_6[0], optimal_loc_6[1], c='red', 
                label=f'Optimal Location for Facility {FACILITY_INDEX_TO_SOLVE + 1}', 
                marker='x', s=200)

    for j, (x, y) in enumerate(coordinates_data):
        plt.text(x, y, str(j), fontsize=8, ha='right')

    plt.title(f'Customer Locations and Optimal Location for Facility {FACILITY_INDEX_TO_SOLVE + 1} (Squared Euclidean Distance)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid(linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()
except Exception as e:
    print(f"\nPlotting skipped due to an error: {e}")