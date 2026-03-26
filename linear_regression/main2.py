import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# ── 1. CREATE DATASET ──────────────────────────────────────────────────────
np.random.seed(0)
n = 100
X = np.linspace(-3, 3, n)
y = 0.5*X**3 - X**2 + 2*X + 1 + np.random.normal(0, 2.5, n)

# Split 70% train, 30% validation
split = int(0.7 * n)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

degrees = [1, 2, 3, 4, 5]

# ── 2. FIGURE 1: Regression Curves ────────────────────────────────────────
X_plot = np.linspace(-3, 3, 200)
train_errs = []
val_errs   = []

plt.figure(figsize=(18, 4))

for i, deg in enumerate(degrees):
    # Train model
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=deg)),
        ('lr',   LinearRegression())
    ])
    model.fit(X_train.reshape(-1, 1), y_train)

    # Compute errors
    tr  = mean_squared_error(y_train, model.predict(X_train.reshape(-1, 1)))
    val = mean_squared_error(y_val,   model.predict(X_val.reshape(-1, 1)))
    train_errs.append(tr)
    val_errs.append(val)

    # Plot
    plt.subplot(1, 5, i + 1)
    plt.scatter(X_train, y_train, color='blue',  s=10, alpha=0.5, label='Train')
    plt.scatter(X_val,   y_val,   color='orange', s=10, alpha=0.5, label='Val')
    plt.plot(X_plot, model.predict(X_plot.reshape(-1, 1)), color='red', lw=2)
    plt.title(f'Degree {deg}\nTrain:{tr:.1f} Val:{val:.1f}')
    plt.xlabel('x')
    if i == 0:
        plt.ylabel('y')
        plt.legend(fontsize=7)

plt.suptitle('Polynomial Regression Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig1_regression_curves.png', dpi=120)
plt.show()
print("Saved fig1_regression_curves.png")

# ── 3. FIGURE 2: Train vs Validation Error ────────────────────────────────
plt.figure(figsize=(7, 5))
plt.plot(degrees, train_errs, 'o-', color='green', label='Train MSE')
plt.plot(degrees, val_errs,   's-', color='red',   label='Val MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.title('Train vs Validation Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('fig2_error_vs_degree.png', dpi=120)
plt.show()
print("Saved fig2_error_vs_degree.png")

# ── 4. FIGURE 3: Learning Curves ──────────────────────────────────────────
plt.figure(figsize=(18, 4))
n_train = len(X_train)
sizes   = list(range(5, n_train + 1, 5))   # 5, 10, 15, ... up to full train set

for i, deg in enumerate(degrees):
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=deg)),
        ('lr',   LinearRegression())
    ])

    tr_curve  = []
    val_curve = []

    for sz in sizes:
        Xt, yt = X_train[:sz].reshape(-1, 1), y_train[:sz]
        model.fit(Xt, yt)
        tr_curve.append( mean_squared_error(yt,    model.predict(Xt)))
        val_curve.append(mean_squared_error(y_val, model.predict(X_val.reshape(-1, 1))))

    plt.subplot(1, 5, i + 1)
    plt.plot(sizes, tr_curve,  color='green', label='Train MSE')
    plt.plot(sizes, val_curve, color='red',   label='Val MSE')
    plt.title(f'Degree {deg}')
    plt.xlabel('Training size')
    if i == 0:
        plt.ylabel('MSE')
        plt.legend(fontsize=7)
    plt.grid(True)

plt.suptitle('Learning Curves (MSE vs Training Size)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig3_learning_curves.png', dpi=120)
plt.show()
print("Saved fig3_learning_curves.png")

# ── 5. ERROR TABLE ────────────────────────────────────────────────────────
print("\nDegree | Train MSE | Val MSE  | Diagnosis")
print("-" * 45)
for d, tr, val in zip(degrees, train_errs, val_errs):
    if d <= 2:
        note = "Underfitting"
    elif d == 5:
        note = "Overfitting"
    else:
        note = "Good Fit"
    print(f"  {d}    |  {tr:6.2f}   |  {val:6.2f}  | {note}")