import csv
import math
import matplotlib.pyplot as plt

coordinates_path = 'coordinates_6.csv'
costs_path       = 'costs_6.csv'
demands_path     = 'demand_6.csv'

def read_coordinates(path):
    coords = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all((c is None or str(c).strip() == '') for c in row):
                continue
            # ilk iki hücreyi float'a çevir (başlık yok varsayımı)
            x = float(row[0]); y = float(row[1])
            coords.append((x, y))
    return coords

def read_demands(path):
    demands = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or str(row[0]).strip() == '':
                continue
            demands.append(float(row[0]))
    return demands

def read_costs(path):
    rows = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append([float(x) for x in row])
    return rows  # ham şekil: r x c (n×m veya m×n olabilir)

# --- VERİLERİ OKU ---
coordinates_data = read_coordinates(coordinates_path)   # n satır
demands_data     = read_demands(demands_path)           # n satır
costs_rows       = read_costs(costs_path)               # r x c

# --- BOYUT & YÖN ---
n = len(coordinates_data)
if n != len(demands_data):
    raise ValueError(f"Demands ile coordinates uyuşmuyor: demands={len(demands_data)}, coords={n}")

r = len(costs_rows)
if r == 0:
    raise ValueError("Costs dosyası boş görünüyor.")
c = len(costs_rows[0])

def transpose(mat):
    return [list(col) for col in zip(*mat)]

# costs yönünü n'e göre belirle:
# r == n → (müşteri×tesis) → transpose;  c == n → (tesis×müşteri) → doğrudan
if r == n:
    facility_cost_list = transpose(costs_rows)  # m×n
    m_facilities = c
    print(f"costs yönü: n×m bulundu → transpose edildi. m={m_facilities}, n={n}")
elif c == n:
    facility_cost_list = costs_rows             # m×n
    m_facilities = r
    print(f"costs yönü: m×n bulundu → doğrudan kullanıldı. m={m_facilities}, n={n}")
else:
    raise ValueError(
        f"costs.csv boyutu beklenmiyor: {r}×{c}. Ne satır ne sütun sayısı müşteri sayısı n={n} ile eşleşmiyor."
    )

# Güvenlik: tüm satırlar n uzunluğunda mı?
for idx, row in enumerate(facility_cost_list):
    if len(row) != n:
        raise ValueError(f"Costs matrisi satır {idx} uzunluğu {len(row)} ≠ n={n}")

print(f"Algılanan boyutlar → Tesis sayısı (m) = {m_facilities}, Müşteri sayısı (n) = {n}")

# --- C[i][j] = h_j * c_ij ---
C = [[facility_cost_list[i][j] * demands_data[j] for j in range(n)]
     for i in range(m_facilities)]

# --- Amaç & optimum ---
def squared_euclidean_distance(x1, y1, x2, y2):
    return (x1 - x2)**2 + (y1 - y2)**2

def objective_function(Ckj, coord, coordinates):
    xk, yk = coord
    return sum(
        Ckj[j] * squared_euclidean_distance(xk, yk, xj, yj)
        for j, (xj, yj) in enumerate(coordinates)
    )

def optimal_location_from_gradient(coordinates, Ckj):
    total = sum(Ckj)
    if total == 0:
        x = sum(x for x, _ in coordinates) / len(coordinates)
        y = sum(y for _, y in coordinates) / len(coordinates)
        return (x, y)
    x = sum(Ckj[j] * coordinates[j][0] for j in range(len(coordinates))) / total
    y = sum(Ckj[j] * coordinates[j][1] for j in range(len(coordinates))) / total
    return (x, y)

results = []
best = None
for i in range(m_facilities):
    opt = optimal_location_from_gradient(coordinates_data, C[i])
    val = objective_function(C[i], opt, coordinates_data)
    results.append((i, val, opt))
    if best is None or val < best[1]:
        best = (i, val, opt)

print("\nTesis özetleri (ilk 10 satır):")
for i, val, opt in results[:10]:
    print(f"  Tesis {i}: Amaç = {val:.6f}, Optimum = ({opt[0]:.6f}, {opt[1]:.6f})")

print(f"\n*** En iyi tesis: {best[0]}  | Amaç = {best[1]:.6f}  | Konum = ({best[2][0]:.6f}, {best[2][1]:.6f})")

# --- Grafik (opsiyonel) ---
try:
    customer_x = [x for x, _ in coordinates_data]
    customer_y = [y for _, y in coordinates_data]

    plt.figure(figsize=(10, 6))
    plt.scatter(customer_x, customer_y, label='Customers', marker='o')
    plt.scatter(best[2][0], best[2][1], label=f'Best Facility {best[0]}', marker='x', s=200)

    for j, (x, y) in enumerate(coordinates_data):
        plt.text(x, y, str(j), fontsize=8, ha='right')

    plt.title('Customers and Optimal Facility Location (Squared Euclidean)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axhline(0, linewidth=0.5, ls='--')
    plt.axvline(0, linewidth=0.5, ls='--')
    plt.grid(linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()
except Exception as e:
    print(f"Grafik çizimi atlandı: {e}")
