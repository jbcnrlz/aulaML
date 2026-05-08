import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA

def main():
    dados = fetch_california_housing()
    X = dados.data
    y = dados.target

    #Normalização por Z-Score
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    X = (X - x_mean) / x_std #dados normalizados

    X_b = np.c_[np.ones((len(X), 1)), X] #Adicionando o termo de interceptação

    lr = 0.05
    n_iteracoes = 500

    #criando modelo inicial randomico
    np.random.seed(42)
    beta = np.random.rand(X_b.shape[1])

    mseHist = []
    for i in range(n_iteracoes):
        y_pred = X_b.dot(beta)

        erro = y_pred - y

        mseAtual = np.mean(erro ** 2)
        mseHist.append(mseAtual)

        gradiente = (2 / len(X_b)) * X_b.T.dot(erro)

        beta = beta - lr * gradiente

    y_pred = X_b.dot(beta)

    #Calculando o erro quadrático médio
    mse = np.mean((y - y_pred) ** 2) #MSE = (1/n) * Σ(y_i - y_pred_i)^2
    rmse = np.sqrt(mse) #RMSE = sqrt(MSE)

    #Calculando o R²
    ss_total = np.sum((y - np.mean(y)) ** 2) #SS_total = Σ(y_i - y_media)^2
    ss_residual = np.sum((y - y_pred) ** 2) #SS_residual = Σ(y_i - y_pred_i)^2
    r2 = 1 - (ss_residual / ss_total) #R² = 1 - (SS_residual / SS_total)
    r2_ajustado = 1 - ((1 - r2) * (len(X) - 1) / (len(X) - X.shape[1] - 1)) #R² ajustado    
    print(f"MSE: {mse}") #Quanto menor melhor
    print(f"RMSE: {rmse}")#Quanto menor melhor
    print(f"R²: {r2}") #Quanto maior melhor
    print(f"R² Ajustado: {r2_ajustado}") #Quanto maior melhor


def mainAnalitico():
    dados = fetch_california_housing()
    X = dados.data
    y = dados.target

    n_amostras = len(X)
    X_b = np.c_[np.ones((n_amostras, 1)), X]

    #Resolvendo pela equação normal (OLS)
    #Beta = (X^T * X)^(-1) * X^T * y
    beta_hat = np.linalg.inv(X_b.T.dot(X_b)) #(X^T * X)^(-1)
    beta_hat = beta_hat.dot(X_b.T) # (X^T * X)^(-1) * X^T
    beta_hat = beta_hat.dot(y) #(X^T * X)^(-1) * X^T * y
    print("Beta estimado:", beta_hat)

    y_pred = X_b.dot(beta_hat)

    #Calculando o erro quadrático médio
    mse = np.mean((y - y_pred) ** 2) #MSE = (1/n) * Σ(y_i - y_pred_i)^2
    rmse = np.sqrt(mse) #RMSE = sqrt(MSE)

    #Calculando o R²
    ss_total = np.sum((y - np.mean(y)) ** 2) #SS_total = Σ(y_i - y_media)^2
    ss_residual = np.sum((y - y_pred) ** 2) #SS_residual = Σ(y_i - y_pred_i)^2
    r2 = 1 - (ss_residual / ss_total) #R² = 1 - (SS_residual / SS_total)

    p = X.shape[1] #Número de características
    r2_ajustado = 1 - ((1 - r2) * (n_amostras - 1) / (n_amostras - p - 1)) #R² ajustado   

    print(f"MSE: {mse}") #Quanto menor melhor
    print(f"RMSE: {rmse}")#Quanto menor melhor
    print(f"R²: {r2}") #Quanto maior melhor
    print(f"R² Ajustado: {r2_ajustado}") #Quanto maior melhor

if __name__ == "__main__":
    mainAnalitico()
    main()