import numpy as np
import matplotlib.pyplot as plt
import math
import sys

# Ensure UTF-8 encoding for stdout/stderr on Windows consoles so
# Unicode characters (e.g. Greek letters, arrows) print correctly.
# Python 3.7+ exposes TextIOWrapper.reconfigure which we use when
# available; if it fails we silently continue.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        # best-effort: if reconfigure fails, continue without crashing
        pass

# =============================================================================
#  REGRESSION ANALYSIS - Insurance Data (1987-1996)
#  Compares Linear / Quadratic / Cubic fits with and without an outlier.
# =============================================================================


# -- 0.  HELPERS --------------------------------------------------------------

def fit_model(x_data, y_data, degree):
    """
    Fit a polynomial of `degree` via the Normal Equation:
        beta = (X^T X)^{-1} X^T y
    Returns (beta, R2, y_pred).
    """
    X = np.vander(x_data, degree + 1, increasing=True)
    beta = np.linalg.inv(X.T @ X) @ X.T @ y_data
    y_pred = X @ beta
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    return beta, r_squared, y_pred


def set_ytick_step(ax, step=1000):
    """Place y-axis ticks every `step` units and enable a light grid."""
    ymin, ymax = ax.get_ylim()
    new_min = math.floor(ymin / step) * step
    new_max = math.ceil(ymax / step) * step
    ax.set_yticks(np.arange(new_min, new_max + step, step))
    ax.grid(True, axis="y", alpha=0.3)


def print_section(title):
    """Print a clearly visible section header."""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def print_subsection(title):
    print(f"\n  -- {title} --")


# -- 1.  DATASET ---------------------------------------------------------------

# x = years since 1987  (0 -> 1987, 9 -> 1996)
# y = number of insured persons
x              = np.array([0, 1, 2, 3,    4,    5,    6,    7,    8,    9])
y_with_outlier = np.array([12400, 10900, 10000, 1050, 9500, 8900, 8000, 7800, 7600, 7200])

# Remove x=3  (y=1050 - clear data-entry error)
x_cleaned = np.delete(x, 3)
y_cleaned = np.delete(y_with_outlier, 3)


# -- 2.  MODEL FITTING & TERMINAL OUTPUT --------------------------------------

def analyse_dataset(label, x_data, y_data):
    """
    Fit Linear / Quadratic / Cubic models, print a formatted summary,
    and return a dict of {model_name: R2}.
    """
    print_section(f"DATASET: {label}")
    results = {}
    model_specs = [("Linear", 1), ("Quadratic", 2), ("Cubic", 3)]

    for model_name, degree in model_specs:
        beta, r2, _ = fit_model(x_data, y_data, degree)
        results[model_name] = r2

        # Build equation string
        eq = f"y = {beta[0]:.2f}"
        for d in range(1, degree + 1):
            eq += f"  +  ({beta[d]:.2f})*x^{d}"

        print_subsection(f"{model_name} Model  (degree {degree})")
        print(f"    Equation : {eq}")
        print(f"    R2       : {r2:.4f}")

    # -- Parsimony-based model selection --
    gain_q = results["Quadratic"] - results["Linear"]
    gain_c = results["Cubic"]    - results["Quadratic"]

    if gain_q > 0.02:
        best = "Quadratic"
        if gain_c > 0.02:
            best = "Cubic"
    else:
        best = "Linear"

    print(f"\n  >>  Model Selection: {best} recommended")
    print(f"     dR2 (Linear -> Quadratic) = {gain_q:+.4f}")
    print(f"     dR2 (Quadratic -> Cubic)  = {gain_c:+.4f}")

    return results


res_outlier = analyse_dataset("ORIGINAL DATA (with outlier)",             x,         y_with_outlier)
res_cleaned = analyse_dataset("CLEANED DATA (without outlier)", x_cleaned, y_cleaned)


# -- 3.  PREDICTIONS -----------------------------------------------------------

print_section("PREDICTIONS FOR 1997  (year = 1997 , x = 10)")

# With-outlier data - highest R2 model
_deg_map   = {"Linear": 1, "Quadratic": 2, "Cubic": 3}
best_with  = max(res_outlier, key=res_outlier.get)
deg_best_w = _deg_map[best_with]
b_best_w, r2_best_w, _ = fit_model(x, y_with_outlier, deg_best_w)
pred_with_best = np.polyval(b_best_w[::-1], 10)
print(f"  Original Data with-outlier (Cubic Model) -> {pred_with_best:,.2f} insured persons")

# Cleaned data - Quadratic (recommended)
beta_q_clean, _, _ = fit_model(x_cleaned, y_cleaned, 2)
pred_1997_clean = beta_q_clean[0] + beta_q_clean[1]*10 + beta_q_clean[2]*(10**2)
print(f"\n  Cleaned Data without-outlier (Quadratic Model) -> {pred_1997_clean:,.2f} insured persons")


# -- 4.  PLOTTING ------------------------------

x_range = np.linspace(0, 10, 100)
labels  = ["Linear", "Quadratic", "Cubic"]
colors  = ["blue", "green", "purple"]

r2_w = [res_outlier[m] for m in labels]
r2_c = [res_cleaned[m] for m in labels]


# -- 4a.  Scatter plots --------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Scatter Plots", fontsize=13, fontweight="bold")

axes[0].scatter(x, y_with_outlier, color="red", s=80)
axes[0].set_title("With Outlier")
axes[0].set_xlabel("Years since 1987 (x)")
axes[0].set_ylabel("Insured Persons (y)")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(x_cleaned, y_cleaned, color="blue", s=80)
axes[1].set_title("Without Outlier")
axes[1].set_xlabel("Years since 1987 (x)")
axes[1].set_ylabel("Insured Persons (y)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# -- 4b.  Individual model plots (Linear / Quadratic / Cubic) -----------------

for deg, color, name in zip([1, 2, 3], colors, labels):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{name} Model", fontsize=13, fontweight="bold")

    for ax, x_data, y_data, dot_color, suffix in [
        (axes[0], x,         y_with_outlier, "red",  "With Outlier"),
        (axes[1], x_cleaned, y_cleaned,      "blue", "Without Outlier"),
    ]:
        b, r2, _ = fit_model(x_data, y_data, deg)
        ax.scatter(x_data, y_data, color=dot_color, label=f"Data ({suffix})")
        ax.plot(x_range, np.polyval(b[::-1], x_range),
                color=color, linewidth=2, label=f"{name} (R2={r2:.4f})")
        ax.set_title(suffix)
        ax.set_xlabel("Years since 1987 (x)")
        ax.set_ylabel("Insured Persons (y)")
        ax.legend()
        set_ytick_step(ax)

    plt.tight_layout()
    plt.show()


# -- 4c.  All three models on one graph ----------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("All Models on Same Graph", fontsize=13, fontweight="bold")

for ax, x_data, y_data, dot_color, suffix in [
    (axes[0], x,         y_with_outlier, "red",  "With Outlier"),
    (axes[1], x_cleaned, y_cleaned,      "blue", "Without Outlier"),
]:
    ax.scatter(x_data, y_data, color=dot_color, label=f"Data ({suffix})")
    for deg, color, name in zip([1, 2, 3], colors, labels):
        b, r2, _ = fit_model(x_data, y_data, deg)
        ax.plot(x_range, np.polyval(b[::-1], x_range),
                color=color, linewidth=2, label=f"{name} (R2={r2:.4f})")
    ax.set_title(suffix)
    ax.set_xlabel("Years since 1987 (x)")
    ax.set_ylabel("Insured Persons (y)")
    ax.legend()
    set_ytick_step(ax)

plt.tight_layout()
plt.show()


# -- 4d.  Best-fit model with 1997 prediction marker --------------------------

def best_model_and_plot(xd, yd, title_suffix):
    """Plot the best-fit model and annotate the 1997 prediction."""
    if title_suffix == "Without Outlier":
        deg_best = 2                                    # quadratic recommended
        b_best, r2_best, _ = fit_model(xd, yd, deg_best)
    else:
        r2s, betas = [], []
        for deg in [1, 2, 3]:
            b, r2, _ = fit_model(xd, yd, deg)
            betas.append((deg, b, r2))
            r2s.append(r2)
        deg_best, b_best, r2_best = betas[int(np.argmax(r2s))]

    fig, ax = plt.subplots(figsize=(8, 6))
    dot_color = "orange" if title_suffix == "With Outlier" else "cyan"
    ax.scatter(xd, yd, color=dot_color, label="Data")
    ax.plot(x_range, np.polyval(b_best[::-1], x_range),
            color="black", linewidth=2,
            label=f"Best Fit: Degree {deg_best}  (R2={r2_best:.4f})")

    # Prediction marker for 1997 (x=10)
    pred_y       = np.polyval(b_best[::-1], 10)
    marker_color = "gold" if title_suffix == "With Outlier" else "darkorange"
    ax.scatter(10, pred_y, color=marker_color, marker="*",
               s=140, edgecolor="k", zorder=5, label="Prediction for 1997")
    ax.annotate(f"{pred_y:.0f}",
                xy=(10, pred_y), xytext=(10.3, pred_y + 300),
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9)

    ax.set_title(f"Best Fit Model - {title_suffix}")
    ax.set_xlabel("Years since 1987 (x)")
    ax.set_ylabel("Insured Persons (y)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    set_ytick_step(ax)
    plt.tight_layout()
    plt.show()


def best_models_side_by_side(x1, y1, label1, x2, y2, label2):
    """Plot the best-fit model for two datasets side-by-side in a single figure."""
    def choose_best(xd, yd, suffix):
        if suffix == "Without Outlier":
            deg_best = 2
            b_best, r2_best, _ = fit_model(xd, yd, deg_best)
        else:
            r2s, betas = [], []
            for deg in [1, 2, 3]:
                b, r2, _ = fit_model(xd, yd, deg)
                betas.append((deg, b, r2))
                r2s.append(r2)
            deg_best, b_best, r2_best = betas[int(np.argmax(r2s))]
        return deg_best, b_best, r2_best

    deg1, b1, r1 = choose_best(x1, y1, label1)
    deg2, b2, r2 = choose_best(x2, y2, label2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, xd, yd, b_best, r_best, deg_best, title_suffix in [
        (axes[0], x1, y1, b1, r1, deg1, label1),
        (axes[1], x2, y2, b2, r2, deg2, label2),
    ]:
        dot_color = "orange" if title_suffix == "With Outlier" else "cyan"
        ax.scatter(xd, yd, color=dot_color, label="Data")
        ax.plot(x_range, np.polyval(b_best[::-1], x_range),
                color="black", linewidth=2,
                label=f"Best Fit: Degree {deg_best}  (R2={r_best:.4f})")

        # Prediction marker for 1997 (x=10)
        pred_y = np.polyval(b_best[::-1], 10)
        marker_color = "gold" if title_suffix == "With Outlier" else "darkorange"
        ax.scatter(10, pred_y, color=marker_color, marker="*",
                   s=140, edgecolor="k", zorder=5, label="Prediction for 1997")
        ax.annotate(f"{pred_y:.0f}",
                    xy=(10, pred_y), xytext=(10.3, pred_y + 300),
                    arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9)

        ax.set_title(f"Best Fit Model - {title_suffix}")
        ax.set_xlabel("Years since 1987 (x)")
        ax.set_ylabel("Insured Persons (y)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        set_ytick_step(ax)

    plt.tight_layout()
    plt.show()


best_models_side_by_side(x, y_with_outlier, "With Outlier", x_cleaned, y_cleaned, "Without Outlier")


# -- 4e.  R2 comparison (scatter points) --------------------------------------

r2_w = [0.1345, 0.2534, 0.3829]   # with outlier
r2_c = [0.9439, 0.9725, 0.9787]   # without outlier

x_pos = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(x_pos, r2_w, color="blue",   s=100, zorder=3, label="With Outlier")
ax.scatter(x_pos, r2_c, color="orange", s=100, zorder=3, marker="s",
           label="Without Outlier")

for i, (wv, cv) in enumerate(zip(r2_w, r2_c)):
    ax.text(i, wv + 0.02, f"{wv:.4f}", ha="center", color="blue",   fontsize=9)
    ax.text(i, cv + 0.02, f"{cv:.4f}", ha="center", color="orange", fontsize=9)

ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.1)
ax.set_ylabel("R2 Value")
ax.set_title("R2 Comparison - Effect of Outlier Removal")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

print_section("DONE")
