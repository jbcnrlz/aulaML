import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# 1. CARREGAMENTO DOS DADOS REAIS
data = fetch_california_housing()
# Usaremos apenas a coluna 0 (MedInc - Renda Média) como X
X_real = data.data[:, 0] 
y_real = data.target     # MedHouseVal (Valor da Casa)

# 2. CÁLCULO DOS COEFICIENTES (MATEMÁTICA PURA)
x_mean = np.mean(X_real)
y_mean = np.mean(y_real)

# B1 = sum((xi - x_mean) * (yi - y_mean)) / sum((xi - x_mean)^2)
numerador = np.sum((X_real - x_mean) * (y_real - y_mean))
denominador = np.sum((X_real - x_mean)**2)

beta_1 = numerador / denominador
beta_0 = y_mean - (beta_1 * x_mean)

# 3. PREVISÕES E MÉTRICAS
y_pred = beta_0 + beta_1 * X_real

# Métricas calculadas do zero
mse = np.mean((y_real - y_pred)**2)
rmse = np.sqrt(mse)

ss_tot = np.sum((y_real - y_mean)**2)
ss_res = np.sum((y_real - y_pred)**2)
r2 = 1 - (ss_res / ss_tot)

# R² Ajustado: n é o total de linhas, p é o número de variáveis (1)
n, p = len(y_real), 1
r2_adj = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

# 4. EXIBIÇÃO
print("--- REGRESSÃO SIMPLES: CALIFORNIA HOUSING (1D) ---")
print(f"Variável Independente: Renda Média (MedInc)")
print(f"Equação: y = {beta_0:.4f} + {beta_1:.4f} * x")
print("-" * 50)
print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f} | R² Ajustado: {r2_adj:.4f}")

# Visualização para os alunos
plt.figure(figsize=(10, 6))
plt.scatter(X_real[:500], y_real[:500], alpha=0.5, label="Dados Reais (Amostra de 500)")
plt.plot(X_real[:500], y_pred[:500], color="red", linewidth=3, label="Reta OLS")
plt.title("Renda Média vs Valor da Casa (California)")
plt.xlabel("Renda Média")
plt.ylabel("Valor Médio da Casa (x100k)")
plt.legend()
plt.show()