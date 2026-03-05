import numpy as np

# 1. Preparação dos Dados (Porta AND)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_true = np.array([[0], [0], [0], [1]])

# Função de Ativação Sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivada da Sigmoide (necessária para a Regra da Cadeia no neurônio)
def sigmoid_derivative(p):
    return p * (1 - p)

# 2. Inicialização de Parâmetros
np.random.seed(42)
weights = np.random.uniform(-1, 1, (2, 1))
bias = np.random.uniform(-1, 1, (1, 1))
learning_rate = 0.5
epochs = 5000

print("Treinando Neurônio com Gradiente Descendente...")

# 3. Loop de Otimização
for epoch in range(epochs):
    # --- Forward Pass ---
    z = np.dot(X, weights) + bias
    y_pred = sigmoid(z)
    
    # Cálculo do MSE para monitoramento
    mse = np.mean((y_true - y_pred)**2)
    
    # --- Backward Pass (Regra da Cadeia) ---
    # 1. Derivada da Externa (MSE): 2/n * (y_pred - y_true)
    # 2. Derivada da Ativação (Sigmoide): sigmoid_derivative
    # 3. Derivada da Interna (Linear): X
    error = y_pred - y_true
    d_prediction = error * sigmoid_derivative(y_pred)
    
    # Gradientes (∇J)
    weights_gradient = np.dot(X.T, d_prediction)
    bias_gradient = np.sum(d_prediction, axis=0, keepdims=True)
    
    # --- Atualização dos Parâmetros ---
    # theta = theta - eta * gradiente
    weights -= learning_rate * weights_gradient
    bias -= learning_rate * bias_gradient
    
    if epoch % 1000 == 0:
        print(f"Época {epoch} | MSE: {mse:.6f}")

# 4. Verificação dos Resultados
print("\n--- Teste Final ---")
final_out = np.round(sigmoid(np.dot(X, weights) + bias))
for i in range(len(X)):
    print(f"Input: {X[i]} -> Pred: {int(final_out[i][0])} (Alvo: {y_true[i][0]})")