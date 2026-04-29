import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# ==========================================
# 1. CARREGAMENTO DOS DADOS REAIS
# ==========================================
data = fetch_california_housing()
X_raw = data.data  # 8 variáveis
y = data.target    # Variável alvo (Valor da casa)

m = len(y)         # Número de amostras (linhas)
n = X_raw.shape[1] # Número de features (colunas)

# ==========================================
# 2. ESCALONAMENTO DE DADOS (CRÍTICO PARA O GD)
# ==========================================
# Padronização (Z-score): subtrai a média e divide pelo desvio padrão.
# Isso coloca todas as features na mesma escala (média 0, variância 1).
X_mean = np.mean(X_raw, axis=0)
X_std = np.std(X_raw, axis=0)
X_scaled = (X_raw - X_mean) / X_std

# Adicionando a coluna de '1's para o intercepto (b0)
X_b = np.c_[np.ones((m, 1)), X_scaled]

# ==========================================
# 3. HIPERPARÂMETROS DO GRADIENTE DESCENDENTE
# ==========================================
taxa_aprendizado = 0.05  # Alpha (passo de descida)
n_iteracoes = 500        # Quantidade de épocas (loops)

# Inicializando os pesos (theta) aleatoriamente
np.random.seed(42)
theta = np.random.randn(n + 1) # n features + 1 intercepto

# Lista para armazenar o histórico do MSE e criar a curva de aprendizado
historico_mse = []

# ==========================================
# 4. LOOP DO GRADIENTE DESCENDENTE
# ==========================================
for iteracao in range(n_iteracoes):
    # 4.1. Calcular as previsões atuais do modelo
    y_pred = X_b.dot(theta)
    
    # 4.2. Calcular o erro (resíduo)
    erro = y_pred - y
    
    # 4.3. Calcular o custo atual (MSE) para registro
    mse_atual = np.mean(erro**2)
    historico_mse.append(mse_atual)
    
    # 4.4. Calcular o Gradiente
    # A fórmula da derivada do MSE em relação aos pesos é: (2/m) * X^T * Erro
    gradientes = (2/m) * X_b.T.dot(erro)
    
    # 4.5. Atualizar os pesos (dar o passo ladeira abaixo)
    theta = theta - taxa_aprendizado * gradientes

# 5. CÁLCULO DAS MÉTRICAS E EXIBIÇÃO
# ==========================================
# 5.1 Fazendo a previsão final com os pesos (theta) otimizados
y_pred_final = X_b.dot(theta)

# 5.2 Calculando o R²
y_mean = np.mean(y)
ss_tot = np.sum((y - y_mean)**2)
ss_res = np.sum((y - y_pred_final)**2)

r2_gd = 1 - (ss_res / ss_tot)

# 5.3 Calculando o R² Ajustado
p = X_raw.shape[1] # Número de features (8)
r2_adj_gd = 1 - ((1 - r2_gd) * (m - 1) / (m - p - 1))

# Exibindo tudo
print("--- RESULTADOS: GRADIENTE DESCENDENTE (MIN-MAX) ---")
print(f"MSE Final   : {historico_mse[-1]:.4f}")
print(f"R²          : {r2_gd:.4f}")
print(f"R² Ajustado : {r2_adj_gd:.4f}")

print("\n--- PESOS FINAIS (THETA) ---")
features = ["Intercepto"] + list(data.feature_names)
for nome, peso in zip(features, theta):
    print(f"{nome:12}: {peso:.6f}")

# Plotando a Curva de Aprendizado
plt.figure(figsize=(10, 5))
plt.plot(range(n_iteracoes), historico_mse, color='purple', linewidth=2)
plt.title('Curva de Aprendizado do GD (Normalização Min-Max)')
plt.xlabel('Iterações (Épocas)')
plt.ylabel('Custo (MSE)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()