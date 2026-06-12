# predictive-analysis-ml-assignment
Predictive analysis ML assignment.
# Predictive Analysis – Regression, Classification and Clustering

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

**Module:** Predictive Analysis – Classifications, Regression, Clustering  
**Degree:** BSc (Hons) Computer Science and Digitisation  
**Institution:** Berlin School of Business and Innovation (BSBI)  
**Submission Date:** 12 June 2026  

---

## Overview

This assignment applies core machine learning techniques to real-world 
business and medical datasets across three tasks of increasing complexity — 
from simple linear regression through polynomial curve fitting to 
kernel-based binary classification.

---

## Tasks at a Glance

| Task | Problem | Model | Result |
|------|---------|-------|--------|
| 1 | TV commercials vs store sales (linear) | Linear Regression | R² = 0.8469 |
| 2 | TV commercials vs store sales (non-linear) | Cubic Polynomial Regression | R² = 0.9432 |
| 3 | Tumour classification | SVM – RBF Kernel | Accuracy = 91.61% |

---

## Code Files

There are **4 Python scripts** in the `/code` folder, 
one per section of the assignment:

| File | Section | Description |
|------|---------|-------------|
| `task1_1_scatter_plot.py` | 2.1 | Task 1 – data visualisation scatter plot |
| `task1_3_linear_regression.py` | 2.3 | Task 1 – Pearson correlation + linear regression fit + residuals |
| `task2_polynomial_regression.py` | 3.4 | Task 2 – linear failure demo + cubic fit + model comparison |
| `task3_svm_classification.py` | 4.3 | Task 3 – SVM training, evaluation, confusion matrix, 3D plot |

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/[your-username]/predictive-analysis-ml-assignment.git
cd predictive-analysis-ml-assignment
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run each script individually**
```bash
# Task 1 – Scatter plot
python code/task1_1_scatter_plot.py

# Task 1 – Linear regression
python code/task1_3_linear_regression.py

# Task 2 – Polynomial regression
python code/task2_polynomial_regression.py

# Task 3 – SVM classification
python code/task3_svm_classification.py
```

> **Windows users:** If you see a `UnicodeEncodeError` in the terminal,
> add these two lines at the very top of any script:
> ```python
> import sys
> sys.stdout.reconfigure(encoding='utf-8')
> ```

---

## Task 1 – Linear Regression

Store 1 data: relationship between weekend TV commercials and weekly sales.

x = [2, 5, 1, 3, 4, 1, 5, 3, 4, 2]   # TV commercials

y = [50,57,41,54,54,38,63,48,59,49]   # Weekly sales ($thousands)
**Key Results:**
- Pearson r = **0.9203** (p = 0.000160)
- Fitted equation: **ŷ = 36.90 + 4.80x**
- R² = **0.8469**
- Each additional commercial → ~$4,800 increase in weekly sales

**Output Plots:**

| Scatter Plot | Regression Fit | Residuals |
|-------------|---------------|-----------|
| ![](figures/task1/Figure_1_1_scatter_plot.png) | ![](figures/task1/Figure_1_3_regression_fit.png) | ![](figures/task1/Figure_1_3_residuals.png) |

---

## Task 2 – Polynomial Regression

Store 2 data: U-shaped relationship — sales fall then rise again 
as commercials increase.
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]

y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
**Why Linear Fails:** R² = 0.1823 — less than 18% of variance explained.

**Degree Comparison:**

| Degree | R² |
|--------|----|
| 1 – Linear | 0.182 |
| 2 – Quadratic | 0.760 |
| **3 – Cubic** | **0.943** ✓ |
| 4 – Quartic | 0.954 |

**Fitted Cubic Model:**
ŷ = -0.0303x³ + 1.3433x² - 15.5383x + 113.768
**Output Plots:**

| Linear Failure | Cubic Fit | Model Comparison |
|---------------|-----------|-----------------|
| ![](figures/task2/Figure_2_3_linear_failure.png) | ![](figures/task2/Figure_2_3_2_cubic_fit.png) | ![](figures/task2/Figure_2_3_3_model_comparison.png) |

---

## Task 3 – SVM Binary Classification

**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** UCI Machine Learning Repository  
**Loaded via:** `sklearn.datasets.load_breast_cancer()`

| Property | Detail |
|----------|--------|
| Samples | 569 |
| Features used | Mean radius, Mean texture, Mean perimeter |
| Malignant | 212 samples |
| Benign | 357 samples |
| Kernel | RBF (`C=1.0`, `gamma='scale'`) |
| Train / Test split | 75% / 25% stratified |

**Results:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Malignant | 0.89 | 0.89 | 0.89 |
| Benign | 0.93 | 0.93 | 0.93 |
| **Overall Accuracy** | | | **91.61%** |
| **5-fold CV** | | | **90.16% ± 2.90%** |

**Confusion Matrix:**

|  | Pred: Malignant | Pred: Benign |
|--|----------------|--------------|
| **Actual: Malignant** | 47 | 6 |
| **Actual: Benign** | 6 | 84 |

**Output Plots:**

| Confusion Matrix | 3D Visualisation |
|-----------------|-----------------|
| ![](figures/task3/Figure_3_3_2_confusion_matrix.png) | ![](figures/task3/Figure_3_3_3_3d_visualization.png) |

---

## Project Structure
predictive-analysis-ml-assignment/

│

├── README.md                        # This file

├── requirements.txt                 # Python dependencies

│

├── code/

│   ├── task1_1_scatter_plot.py      # Task 1: scatter plot

│   ├── task1_3_linear_regression.py # Task 1: regression analysis

│   ├── task2_polynomial_regression.py # Task 2: polynomial regression

│   └── task3_svm_classification.py  # Task 3: SVM classifier

│

├── report/

│   └── Assignment_Report.pdf        # Final submitted report

│

├── figures/

│   ├── task1/                       # Task 1 output plots

│   ├── task2/                       # Task 2 output plots

│   └── task3/                       # Task 3 output plots

│

└── terminal_outputs/                # Console output screenshots
---

## Dependencies
numpy

scipy

matplotlib

scikit-learn

seaborn

Install all:
```bash
pip install -r requirements.txt
```

---

## References

- Bishop, C.M. (2006) *Pattern Recognition and Machine Learning*. Springer.
- Cortes, C. and Vapnik, V. (1995) 'Support-vector networks',
  *Machine Learning*, 20(3), pp. 273–297.
- Draper, N.R. and Smith, H. (1998) *Applied Regression Analysis*. Wiley.
- Géron, A. (2022) *Hands-On Machine Learning with Scikit-Learn,
  Keras & TensorFlow*. 3rd edn. O'Reilly.
- James, G. et al. (2021) *An Introduction to Statistical Learning*.
  2nd edn. Springer.
- Montgomery, D.C., Peck, E.A. and Vining, G.G. (2012)
  *Introduction to Linear Regression Analysis*. 5th edn. Wiley.
- Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python',
  *Journal of Machine Learning Research*, 12, pp. 2825–2830.
- Wolberg, W.H., Street, W.N. and Mangasarian, O.L. (1994)
  'Machine learning techniques to diagnose breast cancer',
  *Cancer Letters*, 77(2–3), pp. 163–171.

---

*Submitted as part of the BSc (Hons) Computer Science and Digitisation 
programme at BSBI, affiliated with the University for the Creative Arts.*
