import os, math, csv
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# ====================== CONFIGURATION =======================
# ============================================================

# Fixed by HW3
EPS2 = 0.005            # Tolerance for exact line search (Golden Section)
A, B = -100.0, 100.0    # Interval [a, b] for line search

# Two parameter sets for Cyclic Coordinate Search (CCS)
# Each tuple: (epsilon1, initial_point)
CCS_SETS = [
    (1e-4, np.array([1.0, 0.0], dtype=float)),     # Set 1
    (1e-5, np.array([1.0, -1.0], dtype=float)),    # Set 2
]

# Two parameter sets for Hook–Jeeves (HJ)
# Each tuple: (epsilon1, initial_point, initial_delta)
HJ_SETS = [
    (1e-4, np.array([1.0, 0.0], dtype=float), 1.0),    # Set 1
    (1e-5, np.array([1.0, -1.0], dtype=float), 2.0),   # Set 2
]

# Two initial simplexes for Nelder–Mead Simplex
# Each simplex has 3 vertices (2D problem)
SIMPLEX_SETS = [
    np.array([[2.0, 3.0],[2.05,3.0],[2.0,3.05]], dtype=float),
    np.array([[0.0, 0.0],[0.5, 0.0],[0.0,0.5]], dtype=float),
]

# ============================================================
# ==================== OBJECTIVE FUNCTION ====================
# ============================================================

def f(x):
    """
    Objective function to minimize:
        f(x1, x2) = (5*x1 - x2)^4 + (x1 - 2)^2 + x1 - 2*x2 + 12
    Input:  x = [x1, x2]
    Output: scalar function value
    """
    x1, x2 = float(x[0]), float(x[1])
    return (5*x1 - x2)**4 + (x1 - 2)**2 + x1 - 2*x2 + 12

# ============================================================
# ============== 1D GOLDEN SECTION SEARCH ====================
# ============================================================

def golden_section_search(phi, a, b, tol):
    """
    1D Golden Section Search to minimize a unimodal function phi(alpha).
    Arguments:
        phi : function of one variable (alpha)
        a, b : interval bounds
        tol : tolerance for interval size
    Returns:
        Approximate minimizer alpha* in [a, b]
    """
    gr = (math.sqrt(5.0) - 1.0) / 2.0  # ~0.618
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = phi(c)
    fd = phi(d)

    while abs(b - a) > tol:
        if fc < fd:
            b, fd = d, fc
            d = c
            c = b - gr * (b - a)
            fc = phi(c)
        else:
            a, fc = c, fd
            c = d
            d = a + gr * (b - a)
            fd = phi(d)

    return 0.5 * (a + b)

def exact_line_search(x, d, eps2=EPS2, a=A, b=B):
    """
    Exact line search along direction d using Golden Section Search.
    Minimizes phi(alpha) = f(x + alpha*d) over [a, b].
    """
    d = np.asarray(d, float)
    def phi(alpha):
        return f(x + alpha * d)
    return golden_section_search(phi, a, b, eps2)

# ============================================================
# ============== CYCLIC COORDINATE SEARCH (CCS) ==============
# ============================================================

def cyclic_coordinate_search(x0, eps1, eps2=EPS2, a=A, b=B, max_iter=10000):

    x = np.array(x0, float)
    n = len(x)
    hist = []
    step = 0  
    for _k in range(max_iter):
        f_old = f(x.copy())
        for i in range(n):
            d = np.zeros(n); d[i] = 1.0
            alpha = exact_line_search(x, d, eps2, a, b)
            x_next = x + alpha * d
            hist.append([step, tuple(x), f(x), tuple(d), float(alpha), tuple(x_next)])
            x = x_next
            step += 1  

        if abs(f(x) - f_old) < eps1:
            break

    return x, f(x), hist
# ============================================================
# ========= HOOK–JEEVES (EXPLORATORY + PATTERN MOVE) =========
# ============================================================

def exploratory_move(x, delta):
    """
    Explore local neighborhood around x with step size delta.
    Tries x +/- delta*e_i for each coordinate; returns the best neighbor.
    """
    x_best = x.copy()
    f_best = f(x_best)
    for i in range(len(x)):
        for s in (+1, -1):
            xt = x.copy()
            xt[i] += s * delta
            ft = f(xt)
            if ft < f_best:
                x_best, f_best = xt, ft
    return x_best

def hook_jeeves(x0, eps1, delta0=1.0, eps2=EPS2, a=A, b=B, max_iter=10000):
    """
    Hook–Jeeves Pattern Search:
      - Exploratory move: find a locally improved point
      - Pattern move: move further along improvement direction
      - Reduce delta when no improvement is found
    Stop when delta < eps1 or iteration limit is reached.
    Returns:
        x_star, f_star, history_rows
        history_rows columns: [k, x(k), f(x(k)), xtemp, d(k), alpha(k), x(k+1)]
    """
    x = np.array(x0, float)
    delta = float(delta0)
    hist = []
    k = 0

    while delta > eps1 and k < max_iter:
        xtemp = exploratory_move(x, delta)
        d = xtemp - x

        # No improvement: shrink step and continue
        if np.allclose(d, 0.0):
            delta *= 0.5
            continue

        # Pattern move along direction d with exact line search
        alpha = exact_line_search(x, d, eps2, a, b)
        x_new = x + alpha * d
        hist.append([k, tuple(x), f(x), tuple(xtemp), tuple(d), float(alpha), tuple(x_new)])
        x = x_new
        k += 1

    return x, f(x), hist

# ============================================================
# ================== SIMPLEX (NELDER–MEAD) ===================
# ============================================================

def simplex_search(x_simplex, alpha=1.0, beta=0.5, gamma=2.0, tol=1e-5, max_iter=2000):
    """
    Nelder–Mead Simplex method in 2D.
    Uses reflection (R), expansion (E), contraction/shrink (C).
    Stop when |f(highest) - f(lowest)| < tol or iteration limit is reached.
    Returns:
        x_star, f_star, history_rows
        history_rows columns: [iter, xbar, xh, xl, xnew, f(xnew), step_type]
    """
    simplex = np.array(x_simplex, float).copy()  # (n+1, n)
    n = simplex.shape[1]
    hist = []

    def order():
        fvals = np.array([f(x) for x in simplex])
        idx = np.argsort(fvals)
        return simplex[idx], fvals[idx]

    simplex, fvals = order()
    it = 0

    while it < max_iter and abs(fvals[-1] - fvals[0]) > tol:
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1].copy()
        best  = simplex[0].copy()

        # Reflection
        x_ref = centroid + alpha * (centroid - worst)
        f_ref = f(x_ref)
        step, x_new, f_new = None, None, None

        if f_ref < fvals[0]:
            # Expansion
            x_exp = centroid + gamma * (x_ref - centroid)
            f_exp = f(x_exp)
            if f_exp < f_ref:
                simplex[-1] = x_exp; step = 'E'; x_new = x_exp; f_new = f_exp
            else:
                simplex[-1] = x_ref; step = 'R'; x_new = x_ref; f_new = f_ref
        elif fvals[0] <= f_ref < fvals[-2]:
            # Accept reflection
            simplex[-1] = x_ref; step = 'R'; x_new = x_ref; f_new = f_ref
        else:
            # Contraction
            if f_ref < fvals[-1]:
                x_con = centroid + beta * (x_ref - centroid)      # outside contraction
            else:
                x_con = centroid + beta * (worst - centroid)      # inside contraction
            f_con = f(x_con)
            if f_con < fvals[-1]:
                simplex[-1] = x_con; step = 'C'; x_new = x_con; f_new = f_con
            else:
                # Shrink toward best
                for i in range(1, n+1):
                    simplex[i] = best + 0.5 * (simplex[i] - best)
                step = 'C'
                x_new = simplex[-1].copy()
                f_new = f(x_new)

        simplex, fvals = order()
        hist.append([it, tuple(centroid), tuple(worst), tuple(best), tuple(x_new), float(f_new), step])
        it += 1

    return simplex[0].copy(), fvals[0].copy(), hist

# ============================================================
# ====================== CSV & PRETTY PRINT ==================
# ============================================================

def write_csv(path, rows, header):
    """
    Write rows to a CSV file with a header.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def _vfmt(v, nd=4):
    """
    Format a 2D vector (tuple/list/ndarray) as a string with nd decimals.
    """
    return f"({v[0]:.{nd}f}, {v[1]:.{nd}f})"

def pretty_print_ccs(title, rows, x_star, f_star, max_rows=30):
    """
    Print CCS table.
    Columns: k, x(k), f(x(k)), d(k), alpha(k), x(k+1)
    """
    print("\n" + title)
    header = ["k", "x^(k)", "f(x^(k))", "d^(k)", "alpha(k)", "x^(k+1)"]
    widths = [4, 23, 14, 18, 10, 23]
    rule = "-" * (sum(widths) + 5)
    line = f"{header[0]:>{widths[0]}} {header[1]:>{widths[1]}} {header[2]:>{widths[2]}} {header[3]:>{widths[3]}} {header[4]:>{widths[4]}} {header[5]:>{widths[5]}}"
    print(line); print(rule)
    for r in rows[:max_rows]:
        k, xk, fk, dk, ak, xkp1 = r
        print(f"{k:>{widths[0]}} {_vfmt(xk):>{widths[1]}} {fk:>{widths[2]}.6f} {_vfmt(dk):>{widths[3]}} {ak:>{widths[4]}.6f} {_vfmt(xkp1):>{widths[5]}}")
    if len(rows) > max_rows:
        print("... (full table saved to CSV)")
    print(rule)
    print(f"x* = {_vfmt(x_star,6)}")
    print(f"f(x*) = {f_star:.12f}")

def pretty_print_hj(title, rows, x_star, f_star, max_rows=30):
    """
    Print Hook–Jeeves table.
    Columns: k, x(k), f(x(k)), xtemp, d(k), alpha(k), x(k+1)
    """
    print("\n" + title)
    header = ["k", "x^(k)", "f(x^(k))", "x_temp", "d^(k)", "alpha(k)", "x^(k+1)"]
    widths = [4, 23, 14, 23, 18, 10, 23]
    rule = "-" * (sum(widths) + 6)
    line = f"{header[0]:>{widths[0]}} {header[1]:>{widths[1]}} {header[2]:>{widths[2]}} {header[3]:>{widths[3]}} {header[4]:>{widths[4]}} {header[5]:>{widths[5]}} {header[6]:>{widths[6]}}"
    print(line); print(rule)
    for r in rows[:max_rows]:
        k, xk, fk, xt, dk, ak, xkp1 = r
        print(f"{k:>{widths[0]}} {_vfmt(xk):>{widths[1]}} {fk:>{widths[2]}.6f} {_vfmt(xt):>{widths[3]}} {_vfmt(dk):>{widths[4]}} {ak:>{widths[5]}.6f} {_vfmt(xkp1):>{widths[6]}}")
    if len(rows) > max_rows:
        print("... (full table saved to CSV)")
    print(rule)
    print(f"x* = {_vfmt(x_star,6)}")
    print(f"f(x*) = {f_star:.12f}")

def pretty_print_simplex(title, rows, x_star, f_star, max_rows=30):
    """
    Print Simplex table.
    Columns: Iter, xbar, xh, xl, xnew, f(xnew), type
    """
    print("\n" + title)
    header = ["Iter", "xbar", "xh", "xl", "xnew", "f(xnew)", "type"]
    widths = [6, 23, 23, 23, 23, 14, 6]
    rule = "-" * (sum(widths) + 6)
    line = f"{header[0]:>{widths[0]}} {header[1]:>{widths[1]}} {header[2]:>{widths[2]}} {header[3]:>{widths[3]}} {header[4]:>{widths[4]}} {header[5]:>{widths[5]}} {header[6]:>{widths[6]}}"
    print(line); print(rule)
    for r in rows[:max_rows]:
        it, xbar, xh, xl, xnew, fnew, t = r
        print(f"{it:>{widths[0]}} {_vfmt(xbar):>{widths[1]}} {_vfmt(xh):>{widths[2]}} {_vfmt(xl):>{widths[3]}} {_vfmt(xnew):>{widths[4]}} {fnew:>{widths[5]}.6f} {t:>{widths[6]}}")
    if len(rows) > max_rows:
        print("... (full table saved to CSV)")
    print(rule)
    print(f"x* = {_vfmt(x_star,6)}")
    print(f"f(x*) = {f_star:.12f}")

# ============================================================
# ============================ MAIN ==========================
# ============================================================

if __name__ == "__main__":
    out_dir = "./outputs"
    os.makedirs(out_dir, exist_ok=True)

    # Analytical reference (for quick validation)
    y_star = -(2**(-1/3))
    x1_star = 6.5
    x2_star = 5 * x1_star - y_star
    f_star_true = f(np.array([x1_star, x2_star]))
    print("\nAnalytical optimum (reference):")
    print(f"x* = ({x1_star:.9f}, {x2_star:.9f})")
    print(f"f(x*) = {f_star_true:.12f}")

    # ---- CCS (two runs) ----
    for i, (eps1, x0) in enumerate(CCS_SETS, 1):
        x_star, f_star, hist = cyclic_coordinate_search(x0, eps1, eps2=EPS2, a=A, b=B)
        csv_path = os.path.join(out_dir, f"CCS_set{i}.csv")
        write_csv(csv_path, hist, ["k","x(k)","f(x(k))","d(k)","alpha(k)","x(k+1)"])
        pretty_print_ccs(
            f"CCS - Set {i}  (eps1={eps1}, x0={tuple(x0)}, eps2={EPS2}, [a,b]=[{A},{B}])",
            hist, x_star, f_star
        )
        # Optional path plot (trajectory of accepted points)
        if hist:
            traj = np.array([np.array(p[-1]) for p in hist])
            fig = plt.figure()
            plt.plot(traj[:,0], traj[:,1])
            plt.title(f"CCS trajectory (set {i})")
            plt.xlabel("x1"); plt.ylabel("x2")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"CCS_trajectory_set{i}.png"))
            plt.close(fig)

    # ---- Hook–Jeeves (two runs) ----
    for i, (eps1, x0, d0) in enumerate(HJ_SETS, 1):
        x_star, f_star, hist = hook_jeeves(x0, eps1, delta0=d0, eps2=EPS2, a=A, b=B)
        csv_path = os.path.join(out_dir, f"HJ_set{i}.csv")
        write_csv(csv_path, hist, ["k","x(k)","f(x(k))","xtemp","d(k)","alpha(k)","x(k+1)"])
        pretty_print_hj(
            f"Hook-Jeeves - Set {i}  (eps1={eps1}, x0={tuple(x0)}, delta0={d0}, eps2={EPS2}, [a,b]=[{A},{B}])",
            hist, x_star, f_star
        )
        if hist:
            traj = np.array([np.array(p[-1]) for p in hist])
            fig = plt.figure()
            plt.plot(traj[:,0], traj[:,1])
            plt.title(f"Hook-Jeeves trajectory (set {i})")
            plt.xlabel("x1"); plt.ylabel("x2")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"HJ_trajectory_set{i}.png"))
            plt.close(fig)

    # ---- Simplex (two initial simplexes) ----
    for i, simp0 in enumerate(SIMPLEX_SETS, 1):
        x_star, f_star, hist = simplex_search(
            simp0, alpha=1.0, beta=0.5, gamma=2.0, tol=1e-5, max_iter=2000
        )
        csv_path = os.path.join(out_dir, f"Simplex_set{i}.csv")
        write_csv(csv_path, hist, ["Iteration","xbar","xh","xl","xnew","f(xnew)","type"])
        pretty_print_simplex(
            f"Simplex - Set {i}  (alpha,beta,gamma)=(1,0.5,2), tol=1e-5, max_iter=2000",
            hist, x_star, f_star
        )
        if hist:
            traj = np.array([np.array(p[4]) for p in hist])
            fig = plt.figure()
            plt.plot(traj[:,0], traj[:,1])
            plt.title(f"Simplex accepted points (set {i})")
            plt.xlabel("x1"); plt.ylabel("x2")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"Simplex_trajectory_set{i}.png"))
            plt.close(fig)

    print(f"\nAll CSVs and plots saved under: {out_dir}")
