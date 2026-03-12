import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidDerivada(x):
    return x * (1-x)

def mse(y_verdadeiro, y_predito):
    somatorio = 0
    for i in range(len(y_verdadeiro)):
        somatorio += (y_verdadeiro[i] - y_predito[i]) ** 2
    return somatorio / len(y_verdadeiro)
    
    #return np.mean((y_verdadeiro - y_predito) ** 2)

def mseDerivada(y_verdadeiro, y_predito):
    return 2 * (y_predito - y_verdadeiro) / len(y_verdadeiro)

def main():
    #               
    X = np.array([
        #C1, C2
        #[0 , 0], 
        [0 , 1], 
        [1 , 0], 
        [1 , 1]
    ])
    y_True = np.array([0, 0, 1])

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
    lr = 0.1
    for ep in range(epocas):
        z = np.dot(X, pesos) + bias #neuronio
        y_pred = sigmoid(z) #ativacao

        erro = mse(y_True, y_pred)
        
        derivadaPredicao = mseDerivada(y_True, y_pred) * sigmoidDerivada(y_pred)
        wGradiente = np.dot(X.T, derivadaPredicao)
        bGradiente = np.sum(derivadaPredicao, axis=0, keepdims=True)
        
        pesos = pesos - lr * wGradiente
        bias = bias - lr * bGradiente
        print(f"Época: {ep}| MSE: {erro}")

    X = np.array([
        #C1, C2
        [0 , 0], 
        [0 , 1], 
        [1 , 0], 
        [1 , 1]
    ])
    for i in range(len(X)):
        #print(f"Entrada: {X[i]}, Saída Esperada: {y_True[i]}")
        andResult = sigmoid(np.dot(X[i], pesos) + bias)
        andResult = round(andResult[0])
        print(f"Entrada: {X[i]} - Resultado do AND lógico: {andResult}")

if __name__ == "__main__":
    main()