import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.array([2, 5, 1, 3, 4, 1, 5, 3, 4, 2])
y = np.array([50, 57, 41, 54, 54, 38, 63, 48, 59, 49])

# ── Pearson correlation ───────────────────────────────────────────────────────
r, p_value = stats.pearsonr(x, y)
print(f"Pearson r = {r:.4f},  p-value = {p_value:.6f}")
# Output: Pearson r = 0.9203,  p-value = 0.000160

# ── Fit the linear model ──────────────────────────────────────────────────────
# scipy.stats.linregress computes slope and intercept via the
# ordinary least squares (OLS) method, minimising sum of squared residuals
slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
print(f"Regression equation: y_hat = {intercept:.2f} + {slope:.2f}x")
# Output: ŷ = 36.90 + 4.80x

y_pred = slope * x + intercept
r_squared = r_val ** 2
print(f"R² = {r_squared:.4f}")
# Output: R² = 0.8469

# ── Plot: data + regression line ──────────────────────────────────────────────
x_line = np.linspace(0.5, 5.5, 200)
y_line = slope * x_line + intercept

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='steelblue', s=90, edgecolors='navy',
            zorder=5, label='Observed data')
plt.plot(x_line, y_line, color='tomato', linewidth=2,
         label=f'ŷ = {intercept:.2f} + {slope:.2f}x   (R² = {r_squared:.4f})')
plt.xlabel("TV Commercials (x)")
plt.ylabel("Store Sales (y)")
plt.title("Store 1 – Linear Regression Fit")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.55)
plt.tight_layout()
plt.show()

# ── Residuals plot ────────────────────────────────────────────────────────────
residuals = y - y_pred
plt.figure(figsize=(8, 4))
plt.stem(range(len(residuals)), residuals,
         markerfmt='ro', linefmt='gray', basefmt='k-')
plt.axhline(0, color='black', linewidth=1)
plt.xlabel("Observation Index")
plt.ylabel("Residual")
plt.title("Residuals – Store 1 Linear Model")
plt.tight_layout()
plt.show()