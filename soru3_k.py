import csv
import random
import math
import statistics as stats # İstatistikler için ekledik
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
random.seed(RNG_SEED) # Rastgeleliği sabitledik

# --- DATA READING FUNCTIONS ---

# (read_coordinates, read_demands, read_costs fonksiyonları aynı bırakılmıştır)

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

# Maliyet Matrisini (Cmat) m x n (Tesis x Müşteri) formatına ayarlama
if r_costs == n_customers and c_costs > 0:
    Cmat = transpose(costs_rows) # m x n
    N_FACILITIES = c_costs
elif c_costs == n_customers and r_costs > 0:
    Cmat = costs_rows # m x n
    N_FACILITIES = r_costs
else:
    raise ValueError("ERROR: Facility number could not be determined from cost file.")

print(f"Number of facilities determined from cost data: {N_FACILITIES}, Customers: {n_customers}\n")

# --- CORE FUNCTIONS (P=2 MANTIKLI) ---

def solve_single_facility(customer_indices, coords, demands, k_facility_index):
    """
    P=2 Location Step: Ağırlıklı Ortalama (Centroid) konumu bulur. 
    Ağırlık: Talep * Maliyet (h_j * C_kj)
    """
    total_x, total_y, total_weight = 0.0, 0.0, 0.0
    k = k_facility_index
    for i in customer_indices:
        # Ağırlık = Talep * Maliyet
        weight = demands[i] * Cmat[k][i]
        
        total_x += weight * coords[i][0]
        total_y += weight * coords[i][1]
        total_weight += weight
        
    if total_weight == 0:
        return None
    return (total_x / total_weight, total_y / total_weight)

def squared_euclidean_distance(coord1, coord2):
    """Karesel Öklid mesafesini hesaplar."""
    return (coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2

def run_ALA():
    """Runs one iteration of the Alternate Location-Allocation Algorithm."""
    
    # 1. Başlangıç Ataması
    assignments = [[] for _ in range(N_FACILITIES)]
    for i in range(n_customers):
        k = random.randint(0, N_FACILITIES - 1)
        assignments[k].append(i)

    # 2. Başlangıç Konumu (Centroid)
    locations = [solve_single_facility(assignments[k], coordinates_data, demands_data, k) 
                 for k in range(N_FACILITIES)]

    # Boş tesis onarımı (repair empty facilities)
    for k in range(N_FACILITIES):
        if locations[k] is None:
            random_customer = random.randint(0, n_customers - 1)
            locations[k] = coordinates_data[random_customer]

    best_objective = float('inf')
    
    while True:
        # A. ATAMA ADIMI (Allocation Step)
        new_assignments = [[] for _ in range(N_FACILITIES)]
        locations_backup = locations # Konum değişmedi, sadece atama değişiyor
        
        for i, customer_coord in enumerate(coordinates_data):
            min_cost = float('inf')
            best_k = None
            
            for k in range(N_FACILITIES):
                # Maliyet * Karesel Mesafe (Düzeltilen Mantık)
                d_sq = squared_euclidean_distance(customer_coord, locations_backup[k])
                cost = Cmat[k][i] * d_sq
                
                if cost < min_cost:
                    min_cost = cost
                    best_k = k
            
            if best_k is not None:
                new_assignments[best_k].append(i)
        
        # B. YERLEŞİM ADIMI (Location Step)
        new_locations = [
            solve_single_facility(new_assignments[k], coordinates_data, demands_data, k)
            for k in range(N_FACILITIES)
        ]

        # Boş tesisleri bir önceki konuma sabitle
        updated_locations = []
        for k in range(N_FACILITIES):
            if new_locations[k] is None:
                updated_locations.append(locations[k])
            else:
                updated_locations.append(new_locations[k])

        # C. AMAÇ FONKSİYONU (Objective Function)
        new_objective = 0
        for k in range(N_FACILITIES):
            loc = updated_locations[k]
            for i in new_assignments[k]:
                # Amaç: Σ (h_i * C_ki * d^2(a_i, x_k))
                term = (demands_data[i] * Cmat[k][i] * squared_euclidean_distance(coordinates_data[i], loc))
                new_objective += term

        # D. YAKINSAMA KONTROLÜ
        if new_objective < best_objective:
            best_objective = new_objective
            locations = updated_locations
            assignments = new_assignments
        else:
            # İyileşme yoksa döngüyü sonlandır
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