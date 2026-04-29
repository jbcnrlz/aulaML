import numpy as np
from sklearn.datasets import fetch_california_housing

# 1. CARREGAMENTO E PREPARAÇÃO MATRICIAL
data = fetch_california_housing()
X_multi = data.data  # Todas as 8 colunas (MedInc, HouseAge, AveRooms, etc.)
y_multi = data.target

# Adicionando a coluna de '1's para o intercepto (Bias)
n_amostras = X_multi.shape[0]
X_b = np.c_[np.ones((n_amostras, 1)), X_multi]

# 2. RESOLUÇÃO PELA EQUAÇÃO NORMAL (OLS)
# Beta = (X^T * X)^-1 * X^T * y
# Nota didática: np.linalg.inv resolve a inversa matricial de (X^T * X)
beta_hat = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_multi)

# 3. PREVISÕES
y_pred_m = X_b.dot(beta_hat)

# 4. CÁLCULO DE TODAS AS MÉTRICAS DO ZERO
y_mean_m = np.mean(y_multi)

# Erros
mse_m = np.mean((y_multi - y_pred_m)**2)
rmse_m = np.sqrt(mse_m)
mae_m = np.mean(np.abs(y_multi - y_pred_m))

# R² e R² Ajustado
ss_tot_m = np.sum((y_multi - y_mean_m)**2)
ss_res_m = np.sum((y_multi - y_pred_m)**2)
r2_m = 1 - (ss_res_m / ss_tot_m)

p_m = X_multi.shape[1] # 8 variáveis
r2_adj_m = 1 - ((1 - r2_m) * (n_amostras - 1) / (n_amostras - p_m - 1))

# 5. EXIBIÇÃO DOS RESULTADOS
print("--- REGRESSÃO MÚLTIPLA: CALIFORNIA HOUSING (8 FEATURES) ---")
print("-" * 60)
print(f"MAE:  {mae_m:.4f}")
print(f"MSE:  {mse_m:.4f}")
print(f"RMSE: {rmse_m:.4f}")
print(f"R²:   {r2_m:.4f}")
print(f"R² Ajustado: {r2_adj_m:.4f}")

print("\n--- PESOS DOS COEFICIENTES (BETA) ---")
features = ["Intercepto"] + list(data.feature_names)
for name, coef in zip(features, beta_hat):
    print(f"{name:12}: {coef:.6f}")