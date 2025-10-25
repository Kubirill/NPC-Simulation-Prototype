# npc_benchmark_plot.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# === Загрузка данных ===
df = pd.read_csv("npc_benchmark_grid.csv")
print(f"Загружено {len(df)} строк данных")

# === Аппроксимация функции и построение графиков ===
def fit_and_plot(x_col, y_col, xlabel, ylabel, title, filename):
    X = df[[x_col]].values
    y = df[y_col].values
    poly = PolynomialFeatures(degree=2)
    model = LinearRegression().fit(poly.fit_transform(X), y)
    x_pred = np.linspace(0, 2000, 400).reshape(-1, 1)
    y_pred = model.predict(poly.transform(x_pred))

    plt.figure()
    plt.scatter(X, y, s=10, alpha=0.5, label="данные")
    plt.plot(x_pred, y_pred, color="red", linewidth=2, label="аппроксимация (до 2000)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.show()

    # Вывести уравнение и R²
    r2 = model.score(poly.transform(X), y)
    print(f"{title}\n  y = {model.coef_[2]:.4f}·x² + {model.coef_[1]:.4f}·x + {model.intercept_:.4f}  (R²={r2:.3f})\n")

# --- Графики ---
fit_and_plot(
    "n_cities", "build_distance_matrix_ms",
    "Количество городов", "Время build_distance_matrix, мс",
    "build_distance_matrix vs число городов (аппроксимация до 2000)",
    "plot_build_distance_matrix_grid_fit.png"
)

fit_and_plot(
    "n_npcs", "trio_per_step_ms",
    "Количество NPC", "Время (select+update+shaker) на шаг, мс",
    "(select+update+shaker) vs NPC (аппроксимация до 2000)",
    "plot_trio_vs_npcs_grid_fit.png"
)
