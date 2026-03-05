import numpy as np
import matplotlib.pyplot as plt

def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# 1. Preparação dos Dados (Porta AND)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_true = np.array([[0], [0], [0], [1]])

# Funções Auxiliares
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(p):
    return p * (1 - p)

# 2. Inicialização de Parâmetros
np.random.seed(42)
weights = np.random.uniform(-1, 1, (2, 1))
bias = np.random.uniform(-1, 1, (1, 1))
learning_rate = 0.5
epochs = 5000

# Lista para armazenar o histórico do erro
mse_history = []

# 3. Loop de Otimização (Gradiente Descendente)
for epoch in range(epochs):
    # --- Forward Pass ---
    z = np.dot(X, weights) + bias
    y_pred = sigmoid(z)
    
    # Cálculo e armazenamento do MSE
    mse = np.mean((y_true - y_pred)**2)
    mse_history.append(mse)
    
    # --- Backward Pass (Regra da Cadeia) ---
    error = y_pred - y_true
    d_prediction = error * sigmoid_derivative(y_pred)
    
    # Gradientes
    weights_gradient = np.dot(X.T, d_prediction)
    bias_gradient = np.sum(d_prediction, axis=0, keepdims=True)
    
    # --- Atualização dos Parâmetros ---
    weights -= learning_rate * weights_gradient
    bias -= learning_rate * bias_gradient

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


plt.plot(mse_history, label='Gradiente Descendente', color='blue')
plt.plot(mse_history_no_grad, label='Busca Aleatória (Sem Gradiente)', color='red')
plt.yscale('log')
plt.title('Eficiência: Gradiente vs. Busca Aleatória')
plt.legend()
plt.show()

# Verificação Final
print(f"MSE Final: {mse_history[-1]:.6f}")