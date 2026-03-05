import numpy as np

# 1. Dados da Porta AND
# Entradas (x1, x2) e Alvos (y)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_true = np.array([0, 0, 0, 1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# 2. Inicialização
weights = np.random.randn(2)
bias = np.random.randn(1)
best_mse = float('inf')

print(f"MSE Inicial: {best_mse}")

# 3. Ciclo de Aprendizado (Sem Gradiente)
epochs = 2000
for i in range(epochs):
    # Gerar uma pequena perturbação nos pesos (Candidato)
    step_size = 0.1
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

    if i % 400 == 0:
        print(f"Época {i}: MSE = {best_mse:.4f}")

# 4. Teste Final
print("\n--- Resultado Final ---")
final_preds = sigmoid(np.dot(X, weights) + bias)
final_preds = np.round(final_preds)  # Arredondar para 0 ou 1
for i in range(len(X)):
    print(f"Input: {X[i]} -> Pred: {final_preds[i]:.4f} (Alvo: {y_true[i]})")