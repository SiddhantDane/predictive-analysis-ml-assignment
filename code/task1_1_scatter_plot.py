import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ── Dataset ──────────────────────────────────────────────────────────────────
x = np.array([2, 5, 1, 3, 4, 1, 5, 3, 4, 2])
y = np.array([50, 57, 41, 54, 54, 38, 63, 48, 59, 49])

# ── Scatter plot ─────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='steelblue', s=90, edgecolors='navy', zorder=5)
plt.xlabel("Number of TV Commercials (x)", fontsize=12)
plt.ylabel("Weekly Store Sales in $thousands (y)", fontsize=12)
plt.title("Store 1 – TV Commercials vs Weekly Sales", fontsize=13)
plt.grid(True, linestyle='--', alpha=0.55)
plt.tight_layout()
plt.show()