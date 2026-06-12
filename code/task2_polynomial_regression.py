import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.array([1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22])
y = np.array([100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100])

# ── Step 1: Show why linear fails ─────────────────────────────────────────────
slope_lin, int_lin, r_lin, *_ = stats.linregress(x, y)
r2_linear = r_lin ** 2
print(f"Linear R² = {r2_linear:.4f}")   # 0.1823 — very poor

plt.figure(figsize=(9, 5))
plt.scatter(x, y, color='darkorange', s=85, edgecolors='saddlebrown',
            zorder=5, label='Observed data')
x_plot = np.linspace(0, 23, 300)
plt.plot(x_plot, slope_lin * x_plot + int_lin, color='gray',
         linestyle='--', linewidth=1.5,
         label=f'Linear fit  (R² = {r2_linear:.4f})')
plt.xlabel("TV Commercials (x)");  plt.ylabel("Store Sales (y)")
plt.title("Store 2 – Linear Fit Failure")
plt.legend();  plt.grid(True, linestyle='--', alpha=0.55)
plt.tight_layout();  plt.show()

# ── Step 2: Fit cubic polynomial (degree 3) ───────────────────────────────────
# np.polyfit returns coefficients [a3, a2, a1, a0] for a3*x³ + a2*x² + a1*x + a0
coeffs = np.polyfit(x, y, 3)
p_cubic = np.poly1d(coeffs)

# Compute R² for cubic model
y_pred_cubic = p_cubic(x)
ss_res = np.sum((y - y_pred_cubic) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2_cubic = 1 - ss_res / ss_tot
print(f"Cubic polynomial R² = {r2_cubic:.4f}")   # 0.9432

print(f"Model: y = {coeffs[0]:.4f}x³ + {coeffs[1]:.4f}x² "
      f"+ {coeffs[2]:.4f}x + {coeffs[3]:.4f}")

# ── Step 3: Plot cubic regression curve ──────────────────────────────────────
plt.figure(figsize=(9, 5))
plt.scatter(x, y, color='darkorange', s=85, edgecolors='saddlebrown',
            zorder=5, label='Observed data')
plt.plot(x_plot, p_cubic(x_plot), color='mediumblue', linewidth=2.2,
         label=f'Cubic fit  (R² = {r2_cubic:.4f})')
plt.xlabel("TV Commercials (x)");  plt.ylabel("Store Sales (y)")
plt.title("Store 2 – Cubic Polynomial Regression")
plt.legend();  plt.grid(True, linestyle='--', alpha=0.55)
plt.tight_layout();  plt.show()

# ── Step 4: Compare model R² values ──────────────────────────────────────────
degrees   = [1, 2, 3, 4]
r2_values = []
for d in degrees:
    c = np.polyfit(x, y, d)
    p = np.poly1d(c)
    ssr = np.sum((y - p(x)) ** 2)
    r2_values.append(1 - ssr / ss_tot)

plt.figure(figsize=(7, 4))
plt.bar([str(d) for d in degrees], r2_values,
        color=['#d9534f','#f0ad4e','#5cb85c','#5bc0de'])
plt.xlabel("Polynomial Degree");  plt.ylabel("R²")
plt.title("Model Comparison – R² by Polynomial Degree")
plt.ylim(0, 1.05)
for i, v in enumerate(r2_values):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
plt.tight_layout();  plt.show()