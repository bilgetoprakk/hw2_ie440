import csv
import random
import math
import matplotlib.pyplot as plt

import statistics as stats  
N_TRIALS = 100             # number of random initializations

# ------------------------------
# CONFIG
# ------------------------------
coordinates_path = 'coordinates_6.csv'
costs_path       = 'costs_6.csv'
demands_path     = 'demand_6.csv'

TOL = 1e-5  # convergence tolerance
EPS = 1e-12 # small epsilon to avoid division by zero

# fix randomness for reproducibility
random.seed(42) 

# --- DECISION VARIABLES (formal definition) ---
# xi  : facility locations (list/tuple) ->  will be filled by Weiszfeld iteration
# yij : (m x n) 0/1 assignment matrix -> derived from final assignments
X_vars = None  # [(x1_1, x2_1), ..., (x1_m, x2_m)]
Y_vars = None  # [[y11, y12, ... y1n], ..., [ym1, ... ymn]]

# ------------------------------
# IO helpers 
# ------------------------------
def read_coordinates(path):
    coords = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        for row in csv.reader(f):
            if not row or all((c is None or str(c).strip()=='' for c in row)): continue
            try:
                coords.append((float(row[0]), float(row[1])))
            except: pass
    return coords

def read_demands(path):
    dem = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        for row in csv.reader(f):
            if not row or str(row[0]).strip()=='' : continue
            try:
                dem.append(float(row[0]))
            except: pass
    return dem

def read_costs(path):
    rows = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        for row in csv.reader(f):
            if not row: continue
            try:
                rows.append([float(x) for x in row])
            except: pass
    return rows

def transpose(mat): return [list(col) for col in zip(*mat)]

# ------------------------------
# DATA LOADING AND MATRIX SHAPING
# ------------------------------
A  = read_coordinates(coordinates_path)
h  = read_demands(demands_path)
R  = read_costs(costs_path)

n = len(A) # number of customers
if n==0: raise ValueError("empty coordinates.")
if len(h)!=n: raise ValueError(f"Demand/coordinate mismatch: {len(h)} vs {n}")

r, c = len(R), len(R[0])
if r==n and c>0:
    Cmat = transpose(R)  # m x n
    m  = c             #number of facilities(m)
elif c==n and r>0:
    Cmat = R             # m x n
    m  = r             #number of facilities (m)
else:
    raise ValueError(f"Unexpected cost matrix size{r}x{c} (n={n})")

print(f"Data loaded.Number of facilities (m): {m}, number of customers(n): {n}")

# ------------------------------
#BASIC GEOMETRIC FUNCTIONS
# ------------------------------
def euclidian_distance(p, q):
    """Euclidian distance between two points."""
    return math.hypot(p[0]-q[0], p[1]-q[1])

# --- ASSIGNMENTS -> y_ij (0/1) helper ---
def assignments_to_y(assignments, m, n):
    """
    'assignments' (list of customer indices per facility) -> (m x n) binary y_ij matrix.
    y_ij = 1  ⇔ customer j assigned to facility i.
    """
    y = [[0]*n for _ in range(m)]
    for i_fac in range(m):
        for j in assignments[i_fac]:
            y[i_fac][j] = 1
    return y

# ------------------------------
# Initial Assignment and location
# ------------------------------
def random_allocation(number_of_customers, number_of_facilities):
    facility_assignments = [[] for _ in range(number_of_facilities)]
    for j in range(number_of_customers):
        facility_index = random.randint(0, number_of_facilities - 1)
        facility_assignments[facility_index].append(j)
    return facility_assignments

def initial_coordinate_for_one_facility(customer_indices, k_facility_index):
    """
    Initial centroid: w = h_j * C_kj
    """
    weight_x = 0
    weight_y = 0
    total_demand_cost = 0
    k = k_facility_index
    for j in customer_indices:
        w = h[j] * Cmat[k][j] 
        weight_x += A[j][0] * w
        weight_y += A[j][1] * w
        total_demand_cost += w
    if total_demand_cost == 0:
        if len(customer_indices) > 0:
            return A[customer_indices[0]]
        else:
            return A[random.randint(0, n-1)]
    return [weight_x / total_demand_cost, weight_y / total_demand_cost]

def initial_coordinates_for_all_facilities(random_allocation_list, number_of_facilities):
    return [initial_coordinate_for_one_facility(random_allocation_list[k], k)
            for k in range(number_of_facilities)]

# ------------------------------
# Allocation Step
# ------------------------------
def nearest_facility(list_of_new_locations):
    """
    Reassign customers to existing facilities.
    Criterion: min_k (C_kj * d(x_k, a_j))
    """
    facility_assignments = [[] for _ in range(m)]
    for j in range(n):
        aj = A[j]
        best_k = None
        best_val = float('inf')
        for k in range(m):
            xk = list_of_new_locations[k]
            val = Cmat[k][j] * euclidian_distance(xk, aj)
            if val < best_val:
                best_val = val
                best_k = k
        facility_assignments[best_k if best_k is not None else random.randint(0, m-1)].append(j)
    return facility_assignments

# ------------------------------
# LOCATION STEP (WEISZFELD)
# ------------------------------
def Weisfeld_Iterations(facility_locations_list, nearest_facility_assignments):
    """
    Update facility locations using the Weiszfeld iteration.
    Weight: (h_j * C_kj) / d(x_k, a_j)
    """
    final_locations_list = []
    for k in range(m):
        xk, yk = facility_locations_list[k]
        assigned_indices = nearest_facility_assignments[k]
        if len(assigned_indices) == 0:
            final_locations_list.append([xk, yk])
            continue
        numx = 0.0; numy = 0.0; den = 0.0
        hit = False
        for j in assigned_indices:
            ajx, ajy = A[j]
            w_h_c = h[j] * Cmat[k][j]
            d = euclidian_distance([xk, yk], A[j])
            if d < EPS:
                numx, numy = ajx, ajy
                hit = True
                break
            inv_w = w_h_c / d
            numx += inv_w * ajx
            numy += inv_w * ajy
            den  += inv_w
        if hit or den == 0.0:
            final_locations_list.append([numx, numy] if hit else [xk, yk])
            continue
        final_locations_list.append([numx/den, numy/den])
    return final_locations_list

# ------------------------------
# OBJECTIVE FUNCTION
# ------------------------------
def objective_function(final_facility_locations, list_of_customers_assigned_to_certain_facility):
    total = 0.0
    for k in range(m):
        xk = final_facility_locations[k]
        for j in list_of_customers_assigned_to_certain_facility[k]:
            aj = A[j]
            total += h[j] * Cmat[k][j] * euclidian_distance(xk, aj)
    return total

# ------------------------------
# MAIN SOLUTION LOOP
# ------------------------------

def ala_single_run():
    # --- Main solution loop ---
    initial_assignments = random_allocation(n, m)
    random_facility_coordinates = initial_coordinates_for_all_facilities(initial_assignments, m)

    nearest_facilities_customers_list1 = nearest_facility(random_facility_coordinates)
    Iteration1_Locations = Weisfeld_Iterations(random_facility_coordinates, nearest_facilities_customers_list1)

    Iteration_Locations = Iteration1_Locations
    iteration_number = 0

    while True:
        iteration_number += 1
        nearest_list = nearest_facility(Iteration_Locations)
        W_k = Weisfeld_Iterations(Iteration_Locations, nearest_list)
        total_difference = 0
        for k in range(m):
            diff_sq = (W_k[k][0] - Iteration_Locations[k][0])**2 + (W_k[k][1] - Iteration_Locations[k][1])**2
            total_difference += diff_sq
        Iteration_Locations = W_k
        if total_difference <= TOL:
            break

    # Decision variables and objective
    X_vars = [tuple(p) for p in W_k]
    Y_vars = assignments_to_y(nearest_list, m, n)
    obj    = objective_function(W_k, nearest_list)
    valid  = all(sum(Y_vars[i][j] for i in range(m)) == 1 for j in range(n))
    return obj, iteration_number, W_k, nearest_list, X_vars, Y_vars, valid



# ------------------------------
# MULTI-START:run N_TRIALS and collect statistics
# ------------------------------
all_vals = []
all_iters = []
best_val = float('inf')
best_pack = None  # (obj, iters, W_k, nearest_list, X_vars, Y_vars, valid)

for t in range(N_TRIALS):
    obj, iters, W_k, nearest_list, X_vars, Y_vars, valid = ala_single_run()
    all_vals.append(obj)
    all_iters.append(iters)
    if obj < best_val:
        best_val = obj
        best_pack = (obj, iters, W_k, nearest_list, X_vars, Y_vars, valid)

# İstatistikler
avg_val = stats.mean(all_vals)
std_val = stats.stdev(all_vals) if len(all_vals) >= 2 else 0.0
avg_iter = stats.mean(all_iters)

print("\n--- ALA (Weiszfeld) Multi-Start Summary ---")
print(f"Trials (N):                {N_TRIALS}")
print(f"Best Objective Value:      {best_val:.6f}")
print(f"Average Objective Value:   {avg_val:.6f}")
print(f"Std Dev Objective Value:   {std_val:.6f}")
print(f"Average #Iterations:       {avg_iter:.2f}")

# print details of best solution
obj, iters, W_k, nearest_list, X_vars, Y_vars, valid = best_pack

# --- y_ij MATRIX (binary 0/1 output) ---
print("\n--- y_ij (Assignment Matrix) ---")
for i in range(m):
    row = " ".join(str(int(v)) for v in Y_vars[i])
    print(f"i={i+1}: {row}")

# --- FACILITY -> CUSTOMERS(a list so that we see which customers are assigned to which facilities) ---
print("\n--- Facility-Customer Assignments ---")
for i in range(m):
    assigned_customers = [j+1 for j, v in enumerate(Y_vars[i]) if v == 1]  # 1-indexed gösterim
    if assigned_customers:
        print(f"Facility {i+1:<2}: {assigned_customers}")
    else:
        print(f"Facility {i+1:<2}: (empty)")

# --- Summary table ---
print("\n--- Facility Summary ---")
print(f"{'Facility':<10} {'#Customers':<12} {'Status'}")
print("-"*35)
for i in range(m):
    num_cust = sum(Y_vars[i])
    status = "Active" if num_cust > 0 else "Empty"
    print(f"{i+1:<10} {num_cust:<12} {status}")


print("\n--- Best Solution (Details) ---")
print(f"Iterations: {iters}")
print(f"Objective:  {obj:.6f}")
print(f"Assignment validity (∀j Σ_i y_ij = 1): {valid}")

print("\nBest Facility Locations:")
for k in range(m):
    loc = W_k[k]
    cnt = len(nearest_list[k])
    status = "Active" if cnt > 0 else "Empty"
    print(f"Facility_{k+1:<2}: Location ({loc[0]:.4f}, {loc[1]:.4f}) | Number of customers: {cnt:<4} | Status: {status}")

# ------------------------------
# PLOTTING (Best Solution)
# ------------------------------
try:
    plt.figure(figsize=(11, 8))
    customer_x_coords, customer_y_coords = zip(*A)
    plt.scatter(customer_x_coords, customer_y_coords, s=20, alpha=0.6, label='Customers', color='blue')

    active_facilities_locs = [W_k[k] for k in range(m) if len(nearest_list[k]) > 0]
    empty_facilities_locs  = [W_k[k] for k in range(m) if len(nearest_list[k]) == 0]

    if active_facilities_locs:
        ax, ay = zip(*active_facilities_locs)
        plt.scatter(ax, ay, marker='X', s=200, color='red', label='Active Facilities', zorder=5)
    if empty_facilities_locs:
        ex, ey = zip(*empty_facilities_locs)
        plt.scatter(ex, ey, marker='X', s=200, color='gray', alpha=0.5, label='Empty facilities', zorder=5)

    # Assignment lines
    for k in range(m):
        xk, yk = W_k[k]
        for j in nearest_list[k]:
            xj, yj = A[j]
            plt.plot([xk, xj], [yk, yj], linestyle='--', linewidth=0.5, color='orange', alpha=0.3)

    plt.title(f"Best Trial – Facility Locations & Assignments (m={m}, n={n})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordiante")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

except Exception as e:
    print(f"\nError faced during plotting: {e}")
