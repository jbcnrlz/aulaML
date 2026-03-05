import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse(y_verdadeiro, y_predito):
    somatorio = 0
    for i in range(len(y_verdadeiro)):
        somatorio += (y_verdadeiro[i] - y_predito[i]) ** 2
    return somatorio / len(y_verdadeiro)
    
    #return np.mean((y_verdadeiro - y_predito) ** 2)

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
    epocas = 100
    for i in range(epocas):
        tamanhoPasso = 0.1
        wCandidato = pesos + np.random.uniform(-tamanhoPasso, tamanhoPasso, size=2)
        bCandidato = bias + np.random.uniform(-tamanhoPasso, tamanhoPasso, size=1)

        #predicao
        z = 0
        for i in range(len(X)):
            zCalc = np.dot(X[i], wCandidato) + bCandidato
            print(f"({X[i][0]} * {wCandidato[0]} + {X[i][1]} * {wCandidato[1]}) + {bCandidato} = {zCalc}")
            z += zCalc
        z = np.dot(X, wCandidato) + bCandidato #neuronio
        y_pred = sigmoid(z) #ativacao
        print(f"Predicao: {y_pred}")

        erro = mse(y_True, y_pred)
        print(f"MSE: {erro}")
        if erro < melhorMSE:
            melhorMSE = erro
            pesos = wCandidato
            bias = bCandidato
            print(f"Novo melhor MSE: {melhorMSE}")

    for i in range(len(X)):
        print(f"Entrada: {X[i]}, Saída Esperada: {y_True[i]}")
        andResult = sigmoid(np.dot(X[i], pesos) + bias)
        andResult = round(andResult[0])
        print(f"Resultado do AND lógico: {andResult}")

if __name__ == "__main__":
    main()