import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(p):
    return p * (1 - p)

def mse(y_verdadeiro, y_predito):
    somatorio = 0
    for i in range(len(y_verdadeiro)):
        somatorio += (y_verdadeiro[i] - y_predito[i]) ** 2
    return somatorio / len(y_verdadeiro)
    
    #return np.mean((y_verdadeiro - y_predito) ** 2)

def mse_derivative(y_verdadeiro, y_predito):
    return 2 * (y_predito - y_verdadeiro) / len(y_verdadeiro)

def main():
    #               
    X = np.array([
        #C1, C2
        [0 , 0], 
        [0 , 1], 
        [1 , 0], 
        [1 , 1]
    ])
    y_True = np.array([0, 0, 0, 1])

    #Criando os pesos e o bias
    pesos = np.random.randn(2)
    bias = np.random.randn(1)

    for i in range(len(X)):
        print(f"Entrada: {X[i]}, Saída Esperada: {y_True[i]}")
        andResult = sigmoid(np.dot(X[i], pesos) + bias)
        andResult = round(andResult[0])
        print(f"Resultado do AND lógico: {andResult}")

    melhorMSE = float('inf')
    epocas = 1000
    learningRate = 0.1
    for i in range(epocas):
        #predicao
        z = 0        
        for i in range(len(X)):
            zCalc = np.dot(X[i], pesos) + bias
            #print(f"({X[i][0]} * {pesos[0]} + {X[i][1]} * {pesos[1]}) + {bias} = {zCalc}")
            z += zCalc
        z = np.dot(X, pesos) + bias #neuronio
        y_pred = sigmoid(z) #ativacao
        #print(f"Predicao: {y_pred}")

        erro = mse(y_True, y_pred)
        #print(f"MSE: {erro}")


        d_predicao = mse_derivative(y_True, y_pred) * sigmoid_derivative(y_pred)
        wGrad = np.dot(X.T, d_predicao)
        bGrad = np.sum(d_predicao, axis=0, keepdims=True)

        pesos -= learningRate * wGrad
        bias -= learningRate * bGrad
        print(f"Época {i} | MSE: {erro:.6f}")

    for i in range(len(X)):
        print(f"Entrada: {X[i]}, Saída Esperada: {y_True[i]}")
        andResult = sigmoid(np.dot(X[i], pesos) + bias)
        andResult = round(andResult[0])
        print(f"Resultado do AND lógico: {andResult}")

if __name__ == "__main__":
    main()

"""
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
"""