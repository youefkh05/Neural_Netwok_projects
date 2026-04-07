import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
#  3D REGRESSION ANALYSIS -- Oil Usage vs Temperature & Insulation
#  Compares Linear / Quadratic fits with and without an outlier.
# =============================================================================


# -- 0.  HELPERS --------------------------------------------------------------

def print_section(title):
    """Print a clearly visible section header."""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def print_subsection(title):
    print(f"\n  -- {title} --")


def fit_model_3d(X_data, y_data):
    """
    Fit Linear and Quadratic models on 2-feature input.
    Returns (lin_reg, quad_reg, poly_transformer, R2_linear, R2_quadratic).
    """
    # Linear
    lin_reg = LinearRegression().fit(X_data, y_data)
    r2_lin  = r2_score(y_data, lin_reg.predict(X_data))

    # Quadratic
    poly    = PolynomialFeatures(degree=2)
    X_poly  = poly.fit_transform(X_data)
    quad_reg = LinearRegression().fit(X_poly, y_data)
    r2_quad  = r2_score(y_data, quad_reg.predict(X_poly))

    return lin_reg, quad_reg, poly, r2_lin, r2_quad


def make_surface_grid(df_plot):
    """Build a meshgrid that covers the data range for surface plotting."""
    t_surf = np.linspace(0, 80, 20)
    i_max  = 40 if df_plot['Insulation'].max() > 20 else 10
    i_surf = np.linspace(0, i_max, 20)
    T, I   = np.meshgrid(t_surf, i_surf)
    grid   = np.c_[T.ravel(), I.ravel()]
    return T, I, grid


def _print_coeffs_equations(X, y, label):
    """Print intercept, coefficients, equations, and R2 for both models."""
    # Linear
    lin         = LinearRegression().fit(X, y)
    r2_lin      = r2_score(y, lin.predict(X))

    # Quadratic (no bias column for clear coefficient ordering)
    poly_local  = PolynomialFeatures(degree=2, include_bias=False)
    Xp          = poly_local.fit_transform(X)
    quad        = LinearRegression().fit(Xp, y)
    r2_quad     = r2_score(y, quad.predict(Xp))

    feat_names  = (poly_local.get_feature_names_out(['Temp', 'Insulation'])
                   if hasattr(poly_local, 'get_feature_names_out')
                   else ['Temp', 'Insulation', 'Temp^2', 'Temp Insulation', 'Insulation^2'])

    print_subsection(label)

    print("  Linear model:")
    print(f"    Intercept : {lin.intercept_:.6f}")
    print(f"    Coef      : Temp={lin.coef_[0]:.6f},  Insulation={lin.coef_[1]:.6f}")
    print(f"    Equation  : y = {lin.intercept_:.4f}"
          f"  +  ({lin.coef_[0]:.4f})*Temp"
          f"  +  ({lin.coef_[1]:.4f})*Insulation")
    print(f"    R2        : {r2_lin:.6f}")

    print("  Quadratic model:")
    print(f"    Intercept : {quad.intercept_:.6f}")
    for name, c in zip(feat_names, quad.coef_):
        print(f"    {name:<20}: {c:.6f}")
    eq = f"y = {quad.intercept_:.4f}"
    for name, c in zip(feat_names, quad.coef_):
        eq += f"  +  ({c:.4f})*{name}"
    print(f"    Equation  : {eq}")
    print(f"    R2        : {r2_quad:.6f}")


# -- 1.  DATASET --------------------------------------------------------------

data = {
    'Oil':        [270, 362, 162,  45,  91, 233, 372, 305, 234, 122,  25, 210, 450, 325,  52],
    'Temp':       [ 40,  27,  40,  73,  65,  65,  10,   9,  24,  65,  66,  41,  22,  40,  60],
    'Insulation': [  4,   4,  10,   6,   7,  40,   6,  10,  10,   4,  10,   6,   4,   4,  10],
}
df = pd.DataFrame(data)

# Outlier: (Temp=65, Insulation=40, Oil=233)
df_cleaned = df[df['Insulation'] < 20].copy()


# -- 2.  MODEL FITTING & TERMINAL OUTPUT --------------------------------------

print_section("MODEL COEFFICIENTS & EQUATIONS")

X_orig = df[['Temp', 'Insulation']].values
y_orig = df['Oil'].values
X_cln  = df_cleaned[['Temp', 'Insulation']].values
y_cln  = df_cleaned['Oil'].values

_print_coeffs_equations(X_orig, y_orig, 'With Outlier (Original)')
_print_coeffs_equations(X_cln,  y_cln,  'Without Outlier (Cleaned)')


# -- 3.  R2 SUMMARY TABLE -----------------------------------------------------

print_section("R2 SUMMARY")

l_m,  q_m,  p_f,  r2_l,  r2_q  = fit_model_3d(X_orig, y_orig)
l_mc, q_mc, p_fc, r2_lc, r2_qc = fit_model_3d(X_cln,  y_cln)

print(f"\n  {'Model':<15} | {'Original R2':<15} | {'Cleaned R2':<15}")
print(f"  {'-'*48}")
print(f"  {'Linear':<15} | {r2_l:<15.4f} | {r2_lc:<15.4f}")
print(f"  {'Quadratic':<15} | {r2_q:<15.4f} | {r2_qc:<15.4f}")

gain_orig = r2_q  - r2_l
gain_cln  = r2_qc - r2_lc
print(f"\n  dR2 (Linear -> Quadratic)  [With Outlier]   : {gain_orig:+.4f}")
print(f"  dR2 (Linear -> Quadratic)  [Without Outlier]: {gain_cln:+.4f}")


# -- 4.  PREDICTIONS ----------------------------------------------------------

print_section("PREDICTIONS AT (Temp=15F, Insulation=5in)")

pred_orig_quad = q_m.predict(p_f.transform([[15, 5]]))[0]
pred_lin_clean = l_mc.predict([[15, 5]])[0]

print(f"\n  Original data  -- Quadratic best-fit : {pred_orig_quad:.2f} units")
print(f"  Cleaned data   -- Linear best-fit    : {pred_lin_clean:.2f} units")


# -- 5.  CONCLUSIONS ----------------------------------------------------------

print_section("CONCLUSIONS")

print("\n  Part a  (With Outlier):")
print(f"    dR2 (Linear -> Quadratic) = {gain_orig:+.4f}  -->  significant gain")
print(f"    Best-fit model: Quadratic  (R2 = {r2_q:.4f})")

print("\n  Part b  (Without Outlier):")
print(f"    dR2 (Linear -> Quadratic) = {gain_cln:+.4f}  -->  not significant")
print(f"    Best-fit model: Linear     (R2 = {r2_lc:.4f})")


# -- 6.  PLOTTING ----------------------------

def plot_regression_3d(df_plot, title, is_quad=False):
    """Plot a fitted surface (linear or quadratic) over the 3-D scatter."""
    X_plot = df_plot[['Temp', 'Insulation']].values
    y_plot = df_plot['Oil'].values
    lin_m, quad_m, poly_f, r2_lin, r2_quad = fit_model_3d(X_plot, y_plot)

    T, I, grid = make_surface_grid(df_plot)

    if is_quad:
        Z     = quad_m.predict(poly_f.transform(grid)).reshape(T.shape)
        label = f"{title}  (Quadratic  R2={r2_quad:.4f})"
    else:
        Z     = lin_m.predict(grid).reshape(T.shape)
        label = f"{title}  (Linear  R2={r2_lin:.4f})"

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, I, Z, cmap='viridis', alpha=0.6)
    ax.scatter(df_plot['Temp'], df_plot['Insulation'], df_plot['Oil'],
               color='red', s=50, label='Data')
    ax.set_title(label)
    ax.set_xlabel('Temp (F)')
    ax.set_ylabel('Insulation (inches)')
    ax.set_zlabel('Oil Usage')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_best_fit_3d(df_plot, title_prefix):
    """Select and plot the best-fit surface (by R2); force Linear for cleaned data."""
    X_plot = df_plot[['Temp', 'Insulation']].values
    y_plot = df_plot['Oil'].values
    lin_m, quad_m, poly_f, r2_lin, r2_quad = fit_model_3d(X_plot, y_plot)

    use_quad  = (r2_quad > r2_lin) and ('without' not in title_prefix.lower())
    best_name = 'Quadratic' if use_quad else 'Linear'
    best_r2   = r2_quad if use_quad else r2_lin

    T, I, grid = make_surface_grid(df_plot)
    Z = (quad_m.predict(poly_f.transform(grid)) if use_quad
         else lin_m.predict(grid)).reshape(T.shape)

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, I, Z, cmap='plasma', alpha=0.6)
    ax.scatter(df_plot['Temp'], df_plot['Insulation'], df_plot['Oil'],
               color='red', s=50, label='Data')
    ax.set_title(f"{title_prefix}  --  Best Fit: {best_name}  (R2={best_r2:.4f})")
    ax.set_xlabel('Temp (F)')
    ax.set_ylabel('Insulation (inches)')
    ax.set_zlabel('Oil Usage')
    ax.legend()
    plt.tight_layout()
    plt.show()


# -- 6a.  Scatter plots -------------------------------------------------------

# With outlier
fig = plt.figure(figsize=(10, 7))
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(df['Temp'], df['Insulation'], df['Oil'], color='blue', s=60, label='Data')
ax.scatter([65], [40], [233], color='red', s=100, label='Outlier')
ax.set_title('Scatter Plot -- With Outlier')
ax.set_xlabel('Temp (F)')
ax.set_ylabel('Insulation (inches)')
ax.set_zlabel('Oil Usage')
ax.legend()
plt.tight_layout()
plt.show()

# Without outlier
fig = plt.figure(figsize=(10, 7))
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(df_cleaned['Temp'], df_cleaned['Insulation'], df_cleaned['Oil'],
           color='blue', s=60)
ax.set_title('Scatter Plot -- Without Outlier')
ax.set_xlabel('Temp (F)')
ax.set_ylabel('Insulation (inches)')
ax.set_zlabel('Oil Usage')
plt.tight_layout()
plt.show()


# -- 6b.  Linear and Quadratic fits -------------------------------------------

plot_regression_3d(df,         "Linear Fitting (With Outlier)",    is_quad=False)
plot_regression_3d(df,         "Quadratic Fitting (With Outlier)",  is_quad=True)
plot_regression_3d(df_cleaned, "Linear Fitting (Cleaned Data)",     is_quad=False)
plot_regression_3d(df_cleaned, "Quadratic Fitting (Cleaned Data)",  is_quad=True)


# -- 6c.  Best-fit surfaces ---------------------------------------------------

plot_best_fit_3d(df,         'Best Fit (With Outlier)')
plot_best_fit_3d(df_cleaned, 'Best Fit (Without Outlier)')

print_section("DONE")