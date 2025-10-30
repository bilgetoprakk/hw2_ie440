import csv
import random
import math
import matplotlib.pyplot as plt

# ------------------------------
# CONFIG
# ------------------------------
coordinates_path = 'coordinates_6.csv'
costs_path       = 'costs_6.csv'
demands_path     = 'demand_6.csv'

TOL = 1e-5  # Yakınsama toleransı (İlk koddaki 0.00001'e denk gelir)
EPS = 1e-12 # Sıfıra bölme hatası için küçük bir değer

# RASTGELELİĞİ SABİTLEME
random.seed(42) 

# --- DECISION VARIABLES (formal tanım) ---
# xi  : tesis konumları (liste/tuple) -> Weiszfeld çıktısıyla dolduracağız
# yij : (m x n) 0/1 atama matrisi -> nihai atamalardan üretilecek
X_vars = None  # [(x1_1, x2_1), ..., (x1_m, x2_m)]
Y_vars = None  # [[y11, y12, ... y1n], ..., [ym1, ... ymn]]

# ------------------------------
# IO helpers (Yeni koddaki robust okuma fonksiyonları)
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
# VERİ YÜKLEME VE MATRİS ŞEKİLLENDİRME
# ------------------------------
A  = read_coordinates(coordinates_path)
h  = read_demands(demands_path)
R  = read_costs(costs_path)

n = len(A) # Müşteri sayısı
if n==0: raise ValueError("Boş koordinatlar.")
if len(h)!=n: raise ValueError(f"Talep/Koordinat uyuşmazlığı: {len(h)} vs {n}")

r, c = len(R), len(R[0])
if r==n and c>0:
    Cmat = transpose(R)  # m x n
    m    = c             # Tesis sayısı (m)
elif c==n and r>0:
    Cmat = R             # m x n
    m    = r             # Tesis sayısı (m)
else:
    raise ValueError(f"Beklenmedik maliyet matrisi boyutu {r}x{c} (n={n})")

print(f"Veriler Yüklendi. Tesis sayısı (m): {m}, Müşteri sayısı (n): {n}")

# ------------------------------
# TEMEL GEOMETRİ FONKSİYONLARI
# ------------------------------
def euclidian_distance(p, q):
    """İki nokta arasındaki Öklid mesafesi."""
    return math.hypot(p[0]-q[0], p[1]-q[1])

# --- ASSIGNMENTS -> y_ij (0/1) Yardımcısı ---
def assignments_to_y(assignments, m, n):
    """
    'assignments' (tesis başına müşteri indeks listesi) -> (m x n) ikili y_ij matrisi.
    y_ij = 1  ⇔ müşteri j tesis i'ye atanmış.
    """
    y = [[0]*n for _ in range(m)]
    for i_fac in range(m):
        for j in assignments[i_fac]:
            y[i_fac][j] = 1
    return y

# ------------------------------
# BAŞLANGIÇ ATAMA VE YERLEŞİM
# ------------------------------
def random_allocation(number_of_customers, number_of_facilities):
    facility_assignments = [[] for _ in range(number_of_facilities)]
    for j in range(number_of_customers):
        facility_index = random.randint(0, number_of_facilities - 1)
        facility_assignments[facility_index].append(j)
    return facility_assignments

def initial_coordinate_for_one_facility(customer_indices, k_facility_index):
    """
    Başlangıç centroid: w = h_j * C_kj
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
# ATAMA ADIMI (ALLOCATION)
# ------------------------------
def nearest_facility(list_of_new_locations):
    """
    Müşterileri mevcut tesislere yeniden atar.
    Kriter: min_k (C_kj * d(x_k, a_j))
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
# YERLEŞİM ADIMI (LOCATION - WEISZFELD)
# ------------------------------
def Weisfeld_Iterations(facility_locations_list, nearest_facility_assignments):
    """
    Weiszfeld yinelemesi ile tesislerin yeni yerleri.
    Ağırlık: (h_j * C_kj) / d(x_k, a_j)
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
# AMAÇ FONKSİYONU
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
# ANA ÇÖZÜM DÖNGÜSÜ
# ------------------------------
initial_assignments = random_allocation(n, m)
random_facility_coordinates = initial_coordinates_for_all_facilities(initial_assignments, m)

nearest_facilities_customers_list1 = nearest_facility(random_facility_coordinates)
Iteration1_Locations = Weisfeld_Iterations(random_facility_coordinates, nearest_facilities_customers_list1)

Iteration_Locations = Iteration1_Locations
iteration_number = 0

print("\n--- ALA (Weiszfeld) İterasyonları Başlıyor ---")
while True:
    iteration_number += 1
    nearest_list = nearest_facility(Iteration_Locations)
    W_k = Weisfeld_Iterations(Iteration_Locations, nearest_list)
    total_difference = 0
    for k in range(m):
        diff_sq = (W_k[k][0] - Iteration_Locations[k][0])**2 + \
                  (W_k[k][1] - Iteration_Locations[k][1])**2
        total_difference += diff_sq
    Iteration_Locations = W_k
    if total_difference <= TOL:
        print(f"\nAlgoritma Sona Erdi. (Toplam Karesel Fark: {total_difference:.12f})")
        break
    if iteration_number % 10 == 0 or iteration_number < 5:
        print(f"ITERASYON: {iteration_number}")

# --- DECISION VARIABLES FROM FINAL SOLUTION ---
# x_i : Weiszfeld sonrası nihai tesis konumları
# y_ij: nihai atamalardan 0/1 matris
X_vars = [tuple(p) for p in W_k]
Y_vars = assignments_to_y(nearest_list, m, n)

print("\n--- Nihai Sonuç ---")
print(f"Toplam İterasyon: {iteration_number}")
print(f"Nihai Amaç Değeri: {objective_function(W_k, nearest_list):.6f}")

print("\nNihai Tesis Konumları:")
for k in range(m):
    loc = W_k[k]
    cnt = len(nearest_list[k])
    status = "Aktif" if cnt > 0 else "Boş"
    print(f"Tesis_{k+1:<2}: Konum ({loc[0]:.4f}, {loc[1]:.4f}) | Müşteri Sayısı: {cnt:<4} | Durum: {status}")

# --- Karar Değişkenlerini Raporla ---
print("\n--- Decision Variables (Heuristic Solution) ---")
for i, (x1, x2) in enumerate(X_vars, start=1):
    print(f"x_{i} = ({x1:.4f}, {x2:.4f})")

print("\ny_ij (assignment matrix, 1 if customer j assigned to facility i else 0)")
for i in range(m):
    row = " ".join(str(int(v)) for v in Y_vars[i])
    print(f"i={i+1}: {row}")

# (Opsiyonel) Her müşteri tam bir tesise atanmış mı?  Σ_i y_ij = 1
valid = all(sum(Y_vars[i][j] for i in range(m)) == 1 for j in range(n))
print(f"\nAssignment validity (∀j Σ_i y_ij = 1): {valid}")

# ------------------------------
# ÇİZİM KISMI
# ------------------------------
try:
    plt.figure(figsize=(11, 8))
    customer_x_coords, customer_y_coords = zip(*A)
    plt.scatter(customer_x_coords, customer_y_coords, s=20, alpha=0.6, label='Müşteriler', color='blue')

    active_facilities_locs = []
    empty_facilities_locs = []
    for k in range(m):
        if len(nearest_list[k]) > 0:
            active_facilities_locs.append(W_k[k])
        else:
            empty_facilities_locs.append(W_k[k])

    if active_facilities_locs:
        ax, ay = zip(*active_facilities_locs)
        plt.scatter(ax, ay, marker='X', s=200, color='red', label='Aktif Tesisler', zorder=5)
    if empty_facilities_locs:
        ex, ey = zip(*empty_facilities_locs)
        plt.scatter(ex, ey, marker='X', s=200, color='gray', alpha=0.5, label='Boş Tesisler', zorder=5)

    for k in range(m):
        xk, yk = W_k[k]
        for j in nearest_list[k]:
            xj, yj = A[j]
            plt.plot([xk, xj], [yk, yj], linestyle='--', linewidth=0.5, color='orange', alpha=0.3)

    plt.title(f"Tesis Yerleşimi ve Müşteri Atamaları (m={m}, n={n})")
    plt.xlabel("X Koordinatı")
    plt.ylabel("Y Koordinatı")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

except Exception as e:
    print(f"\nÇizim sırasında bir hata oluştu: {e}")
