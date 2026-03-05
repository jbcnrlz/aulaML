import numpy as np
import matplotlib.pyplot as plt

# 1. Dados da Porta AND
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_true = np.array([0, 0, 0, 1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# 2. Inicialização (Busca Aleatória / Hill Climbing)
np.random.seed(42)
weights = np.random.randn(2)
bias = np.random.randn(1)

# Cálculo inicial para o histórico
z_init = np.dot(X, weights) + bias
y_pred_init = sigmoid(z_init)
best_mse = compute_mse(y_true, y_pred_init)

mse_history_no_grad = [best_mse]

# 3. Otimização Estocástica (Sem Gradiente)
epochs = 5000
step_size = 0.1

for i in range(epochs):
    # Propor nova solução (perturbação aleatória)
    w_candidate = weights + np.random.uniform(-step_size, step_size, size=2)
    b_candidate = bias + np.random.uniform(-step_size, step_size, size=1)
    
    # Forward Pass
    z = np.dot(X, w_candidate) + b_candidate
    y_pred = sigmoid(z)
    
    # Calcular Loss (MSE)
    current_mse = compute_mse(y_true, y_pred)
    
    # Critério de Aceitação: Se melhorou, mantém
    if current_mse < best_mse:
        best_mse = current_mse
        weights = w_candidate
        bias = b_candidate
    
    mse_history_no_grad.append(best_mse)

# 4. Plot da Convergência Sem Gradiente
plt.figure(figsize=(10, 6))
plt.plot(range(len(mse_history_no_grad)), mse_history_no_grad, color='red', linewidth=2)
plt.title('Convergência do MSE (Sem Gradiente - Busca Aleatória)', fontsize=14)
plt.xlabel('Iterações (Tentativas)', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.yscale('log') # Escala logarítmica para comparação justa
plt.show()

print(f"MSE Final (Sem Gradiente): {best_mse:.6f}")