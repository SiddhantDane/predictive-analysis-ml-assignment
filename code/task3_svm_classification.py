import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, ConfusionMatrixDisplay)
from mpl_toolkits.mplot3d import Axes3D

# ── 1. Load dataset and select 3 features for visualisation ──────────────────
cancer = datasets.load_breast_cancer()
X = cancer.data[:, :3]          # mean radius, mean texture, mean perimeter
y = cancer.target                # 0 = malignant, 1 = benign
feature_names = cancer.feature_names[:3]
class_names   = cancer.target_names

print(f"Samples: {X.shape[0]}  |  Features: {X.shape[1]}")
print(f"Classes: {class_names}  |  Distribution: {np.bincount(y)}")
# Samples: 569  |  Features: 3
# Classes: ['malignant' 'benign']  |  Distribution: [212 357]

# ── 2. Train / test split with stratification to preserve class balance ───────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ── 3. Feature scaling – SVM is distance-based so scaling is critical ─────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit ONLY on training data
X_test_sc  = scaler.transform(X_test)        # apply same scale to test data

# ── 4. Train SVM with RBF kernel ──────────────────────────────────────────────
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
# C controls the trade-off between margin width and misclassification penalty
# gamma='scale' sets gamma = 1 / (n_features * X.var()) automatically
svm.fit(X_train_sc, y_train)

# ── 5. Predictions and evaluation ────────────────────────────────────────────
y_pred = svm.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy:  {acc:.4f}")           # 0.9161
print(classification_report(y_test, y_pred, target_names=class_names))

# ── 6. Cross-validation for robust performance estimate ──────────────────────
X_all_sc = scaler.fit_transform(X)
cv_acc = cross_val_score(svm, X_all_sc, y, cv=5, scoring='accuracy')
print(f"5-fold CV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
# 5-fold CV Accuracy: 0.9016 ± 0.0290

# ── 7. Confusion matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=class_names)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title("Confusion Matrix – SVM (RBF Kernel)")
plt.tight_layout();  plt.show()

# ── 8. 3D scatter plot of the two classes ─────────────────────────────────────
fig = plt.figure(figsize=(11, 8))
ax  = fig.add_subplot(111, projection='3d')

class_colors = {0: '#E74C3C', 1: '#2980B9'}    # red = malignant, blue = benign
class_markers = {0: '^', 1: 'o'}

for cls in [0, 1]:
    mask = y == cls
    ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
               c=class_colors[cls],
               marker=class_markers[cls],
               label=class_names[cls],
               alpha=0.55, s=28, edgecolors='none')

ax.set_xlabel(feature_names[0], fontsize=9, labelpad=8)
ax.set_ylabel(feature_names[1], fontsize=9, labelpad=8)
ax.set_zlabel(feature_names[2], fontsize=9, labelpad=8)
ax.set_title("3D View – Breast Cancer Classes\n"
             "(mean radius / texture / perimeter)", fontsize=11)
ax.legend(loc='upper left', fontsize=10)

# ── 9. Approximate decision boundary as a surface at a fixed perimeter slice ──
# Project a 2D mesh (radius × texture) at the median perimeter value,
# query the SVM, and plot the predicted class surface
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 35),
    np.linspace(X[:, 1].min() - 2, X[:, 1].max() + 2, 35)
)
z_fixed = np.full(xx.shape, np.median(X[:, 2]))    # hold perimeter constant
grid_pts = np.c_[xx.ravel(), yy.ravel(), z_fixed.ravel()]
grid_sc  = scaler.transform(grid_pts)
zz_pred  = svm.predict(grid_sc).reshape(xx.shape)

# Map class prediction (0/1) onto the perimeter axis range for plotting
z_range  = X[:, 2].max() - X[:, 2].min()
surf_z   = zz_pred * z_range * 0.45 + X[:, 2].min() + z_range * 0.25

ax.plot_surface(xx, yy, surf_z,
                alpha=0.18, cmap='RdBu_r', linewidth=0)

plt.tight_layout();  plt.show()