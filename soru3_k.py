import csv
import random
import math
import statistics as stats 
import matplotlib.pyplot as plt

# ------------------------------
# CONFIG
# ------------------------------
coordinates_path = 'coordinates_6.csv'
costs_path = 'costs_6.csv'
demands_path = 'demand_6.csv'

N_TRIALS = 1000 # Number of trials
TOL = 1e-9      # Convergence tolerance (for stability, not used directly in this specific ALA structure)
RNG_SEED = 42
random.seed(RNG_SEED) 

# --- DECISION VARIABLES (Homework Notation) ---
# xi : facility locations (list of tuples) -> We will store the locations found by the heuristic here.
# yij: assignment matrix (m x n, 0/1) -> 1 if customer j is assigned to facility i, 0 otherwise
X_vars = None  # [(x1_1, x2_1), ..., (x1_m, x2_m)]
Y_vars = None  # [[y11, y12, ... y1n], ..., [ym1, ... ymn]]

# --- DATA READING FUNCTIONS ---

def read_coordinates(path):
    """Reads customer coordinates (x, y)."""
    coords = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all((c is None or str(c).strip() == '') for c in row):
                continue
            try:
                coords.append((float(row[0]), float(row[1])))
            except (ValueError, IndexError):
                continue
    return coords

def read_demands(path):
    """Reads customer demands."""
    demands = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or str(row[0]).strip() == '':
                continue
            try:
                demands.append(float(row[0]))
            except ValueError:
                continue
    return demands

def read_costs(path):
    """Reads cost matrix."""
    rows = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                rows.append([float(x) for x in row])
            except ValueError:
                continue
    return rows

def transpose(mat): return [list(col) for col in zip(*mat)]

# --- READ DATA AND DETERMINE NUMBER OF FACILITIES ---

coordinates_data = read_coordinates(coordinates_path)
demands_data = read_demands(demands_path)
costs_rows = read_costs(costs_path)

n_customers = len(coordinates_data)
r_costs = len(costs_rows)
c_costs = len(costs_rows[0]) if r_costs > 0 else 0

# Set the Cost Matrix (Cmat) to m x n (Facility x Customer) format
if r_costs == n_customers and c_costs > 0:
    Cmat = transpose(costs_rows) # m x n
    N_FACILITIES = c_costs
elif c_costs == n_customers and r_costs > 0:
    Cmat = costs_rows # m x n
    N_FACILITIES = r_costs
else:
    raise ValueError("ERROR: Facility number could not be determined from cost file.")

print(f"Number of facilities determined from cost data: {N_FACILITIES}, Customers: {n_customers}\n")

# --- CORE FUNCTIONS (P=2) ---

def solve_single_facility(customer_indices, coords, demands, k_facility_index):
    """
   P=2 Location Step: Finds the Weighted Average (Centroid) location. Weight: Demand * Cost (h_j * C_kj)
    for the facility at index k_facility_index given assigned customers.
    """
    total_x, total_y, total_weight = 0.0, 0.0, 0.0
    k = k_facility_index
    for i in customer_indices:
        # Weight = Demand * Cost
        weight = demands[i] * Cmat[k][i]
        
        total_x += weight * coords[i][0]
        total_y += weight * coords[i][1]
        total_weight += weight
        
    if total_weight == 0:
        return None
    return (total_x / total_weight, total_y / total_weight)

def squared_euclidean_distance(coord1, coord2):
    """Calculates the quadratic Euclidean distance."""
    return (coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2

# --- ASSIGNMENTS -> y_ij (0/1) Auxiliary Function ---
def assignments_to_y(assignments, m, n):
    """
    Converts assignments list to y_ij binary matrix.
    assignments: list of lists, where assignments[i] contains indices of customers assigned to facility i
    """
    y = [[0]*n for _ in range(m)]
    for i_fac in range(m):
        for j in assignments[i_fac]:
            y[i_fac][j] = 1  # customer j is assigned to facility i_fac
    return y

def run_ALA():
    """Runs one iteration of the Alternate Location-Allocation Algorithm."""
    
    # Initial Random Assignment
    assignments = [[] for _ in range(N_FACILITIES)]
    for i in range(n_customers):
        k = random.randint(0, N_FACILITIES - 1)
        assignments[k].append(i)

    # Centroid
    locations = [solve_single_facility(assignments[k], coordinates_data, demands_data, k) 
                 for k in range(N_FACILITIES)]

    # repair empty facilities
    for k in range(N_FACILITIES):
        if locations[k] is None:
            random_customer = random.randint(0, n_customers - 1)
            locations[k] = coordinates_data[random_customer]

    best_objective = float('inf')
    
    while True:
        # Allocation Step
        new_assignments = [[] for _ in range(N_FACILITIES)]
        locations_backup = locations # The location has not changed, only the assignment has changed.
        
        for i, customer_coord in enumerate(coordinates_data):
            min_cost = float('inf')
            best_k = None
            
            for k in range(N_FACILITIES):

                d_sq = squared_euclidean_distance(customer_coord, locations_backup[k])
                cost = d_sq

                
                if cost < min_cost:
                    min_cost = cost
                    best_k = k
            
            if best_k is not None:
                new_assignments[best_k].append(i)
        
        # Location Step
        new_locations = [
            solve_single_facility(new_assignments[k], coordinates_data, demands_data, k)
            for k in range(N_FACILITIES)
        ]

        # Pin empty facilities to previous location
        updated_locations = []
        for k in range(N_FACILITIES):
            if new_locations[k] is None:
                updated_locations.append(locations[k])
            else:
                updated_locations.append(new_locations[k])

        # Objective Function
        new_objective = 0
        for k in range(N_FACILITIES):
            loc = updated_locations[k]
            for i in new_assignments[k]:
                # Amaç: Σ (h_i * C_ki * d^2(a_i, x_k))
                term = (demands_data[i] * Cmat[k][i] * squared_euclidean_distance(coordinates_data[i], loc))
                new_objective += term

    
        if new_objective < best_objective:
            best_objective = new_objective
            locations = updated_locations
            assignments = new_assignments
        else:
            # End if objective doesnt get better
            break

    return best_objective, locations, assignments

# ------------------------------
# MULTI-START (1000 trials)
# ------------------------------

all_vals = []
best_val, best_locations, best_assignments = float('inf'), None, None

for i in range(N_TRIALS):
    val, locs, assign = run_ALA()
    all_vals.append(val)
    if val < best_val:
        best_val, best_locations, best_assignments = val, locs, assign

# --- DECISION VARIABLES FROM BEST SOLUTION ---
# Decision variables:
X_vars = best_locations                                               # xi (sürekli)
Y_vars = assignments_to_y(best_assignments, N_FACILITIES, n_customers)  # yij (0/1 ikili)

# --- RESULTS ---
# ------------------------------

print("\n--- Cost-Weighted P=2 ALA Heuristic Results (Multi-Start) ---")
print(f"Number of Trials: {N_TRIALS}")
print(f"Best Objective Value: {best_val:.4f}")
print(f"Average Objective Value: {stats.mean(all_vals):.4f}")
print(f"Standard Deviation: {stats.stdev(all_vals):.4f}\n")

print("--- Best Solution Facility Details ---")
for i in range(N_FACILITIES):
    loc = best_locations[i]
    num_customers = len(best_assignments[i])
    status = "Active" if num_customers > 0 else "Empty"
    print(f"Facility_{i+1:<2} | Location: ({loc[0]:.2f}, {loc[1]:.2f}) | "
          f"Customers: {num_customers:<3} | Status: {status}")

# --- REPORT DECISION VARIABLES CLEARLY ---
print("\n--- Decision Variables (Heuristic Solution) ---")
for i, (x1, x2) in enumerate(X_vars, start=1):
    print(f"x_{i} = ({x1:.4f}, {x2:.4f})")

"""""
print("\ny_ij (assignment matrix, 1 if customer j assigned to facility i else 0)")
for i in range(N_FACILITIES):
    row = " ".join(str(int(v)) for v in Y_vars[i])
    print(f"i={i+1}: {row}")
"""""
# --- VALIDITY CHECKS ---
# Each customer assigned to exactly one facility
valid = all(sum(Y_vars[i][j] for i in range(N_FACILITIES)) == 1 for j in range(n_customers))
print(f"\nAssignment validity (∀j Σ_i y_ij = 1): {valid}")

# --- PLOTTING ---
# ------------------------------
try:
    plt.figure(figsize=(12, 8))
    cust_x, cust_y = zip(*coordinates_data)
    plt.scatter(cust_x, cust_y, color='blue', label='Customers', s=10, alpha=0.6)

    active_facilities = [best_locations[i] for i in range(N_FACILITIES) if len(best_assignments[i]) > 0]
    empty_facilities = [best_locations[i] for i in range(N_FACILITIES) if len(best_assignments[i]) == 0]

    # Active ones in red
    if active_facilities:
        ax, ay = zip(*active_facilities)
        plt.scatter(ax, ay, color='red', label='Active Facilities', s=200, marker='X', alpha=0.8)

    # Empty ones in gray
    if empty_facilities:
        ex, ey = zip(*empty_facilities)
        plt.scatter(ex, ey, color='gray', label='Empty Facilities', s=200, marker='X', alpha=0.5)

    # Draw assignment lines
    for k in range(N_FACILITIES):
        fac_loc = best_locations[k]
        for i in best_assignments[k]:
            cust_loc = coordinates_data[i]
            plt.plot([fac_loc[0], cust_loc[0]], [fac_loc[1], cust_loc[1]], color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    plt.title(f"Optimal Facility Location and Allocation (P=2 ALA) - M = {N_FACILITIES}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

except Exception as e:
    print(f"\nPlotting Error: {e}")

print("\n--- Customer Assignments by Facility ---")
for i in range(N_FACILITIES):
    assigned_customers = [j+1 for j, val in enumerate(Y_vars[i]) if val == 1]
    if assigned_customers:
        print(f"Facility_{i+1:<2}: Customers → {assigned_customers}")
    else:
        print(f"Facility_{i+1:<2}: No customers assigned")
