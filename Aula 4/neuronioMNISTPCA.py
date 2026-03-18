import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ReLU: f(x) = max(0, x)
def relu(x):
    return np.maximum(0, x)

# Derivada da ReLU: 1 se x > 0, caso contrário 0
def relu_derivative(x):
    return (x > 0).astype(float)

def mse(y_verdadeiro, y_predito):
    return np.mean((y_verdadeiro - y_predito) ** 2)

def main():
    # Carregamento
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y_True = mnist.data, mnist.target
    y_True = y_True.astype(np.float64)

    # Escalonamento
    # 2. Padronização (Essencial)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=150)
    X_scaled = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_True, test_size=0.2, random_state=42, stratify=y_True
    )

    # Inicialização (He initialization aproximada para ReLU)
    # Pesos menores são cruciais para a ReLU não explodir
    pesos = np.random.randn(X_train.shape[1]) * 0.001 
    bias = np.zeros(1)

    epocas = 10000
    # Reduzi o Learning Rate; ReLU converge mais rápido mas é sensível
    learningRate = 0.0001 

    progresso = tqdm(range(epocas), desc="Treinando com ReLU")

    for i in progresso:
        # Neuronio estimando y
        z = np.dot(X_train, pesos) + bias 
        y_pred = relu(z) 

        # Cálculo do erro
        erro = mse(y_train, y_pred)

        # Backpropagation
        # Gradiente do MSE * Gradiente da ReLU
        d_predicao = 2 * (y_pred - y_train) / y_train.size
        d_predicao *= relu_derivative(z)
        
        wGrad = np.dot(X_train.T, d_predicao)
        bGrad = np.sum(d_predicao)

        # Atualização
        pesos -= learningRate * wGrad
        bias -= learningRate * bGrad

        progresso.set_postfix(MSE=f"{erro:.6f}")

    print("\n--- Resultados de Teste (Amostras) ---")
    acertou = 0
    for i in range(X_test.shape[0]):
        z_test = np.dot(X_test[i], pesos) + bias
        pred = relu(z_test)
        acertou += (round(pred[0]) == y_test[i])

    print(f"\nAcurácia ({X_test.shape[0]} amostras): {acertou}/{X_test.shape[0]} ({(acertou/X_test.shape[0])*100:.2f}%)")

if __name__ == "__main__":
    main()