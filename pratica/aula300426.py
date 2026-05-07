import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

def main():
    data = fetch_california_housing()
    x_real = data.data[:, 0]  # MedInc
    y_real = data.target  # MedHouseVal

    # Cálculo dos coeficientes beta_0 e beta_1 usando OLS
    #ŷ = beta_0 + beta_1 * x
    x_media = np.mean(x_real)
    y_media = np.mean(y_real)

    numerador = np.sum((x_real - x_media) * (y_real - y_media))
    denominador = np.sum((x_real - x_media) ** 2)
    #beta_1 = sum((x_real - x_media) * (y_real - y_media)) / sum((x_real - x_media) ** 2)
    beta_1 = numerador / denominador
    #beta_0 = y_media - beta_1 * x_media
    beta_0 = y_media - beta_1 * x_media
    print(f"Modelo: ŷ = {beta_0} + {beta_1} * x")

    y_pred = beta_0 + beta_1 * x_real

    #Calculo do MSE
    mse = np.mean((y_real - y_pred) ** 2) #MSE = (1/n) * sum((y_real - y_pred) ** 2)
    rmse = np.sqrt(mse) #RMSE = sqrt(MSE)

    #Cálculo do R²
    ss_tot = np.sum((y_real - y_media) ** 2)
    ss_res = np.sum((y_real - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    #Cálculo do R² ajustado
    n = len(y_real)
    p = 1
    r2_ajustado = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    print(f"MSE: {mse}") #Quanto menor melhor
    print(f"RMSE: {rmse}")#Quanto menor melhor
    print(f"R²: {r2}") #Quanto maior melhor
    print(f"R² Ajustado: {r2_ajustado}") #Quanto maior melhor

    # Visualização para os alunos
    plt.figure(figsize=(10, 6))
    plt.scatter(x_real, y_real, alpha=0.5, label="Dados Reais (Amostra de 500)")
    plt.plot(x_real, y_pred, color="red", linewidth=3, label="Reta OLS")
    plt.title("Renda Média vs Valor da Casa (California)")
    plt.xlabel("Renda Média")
    plt.ylabel("Valor Médio da Casa (x100k)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()