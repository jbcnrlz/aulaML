import numpy as np

# Gerando 1 milhão de rótulos reais (0 ou 1)
y_real = np.random.randint(0, 2, size=1000000)

# Simulando as previsões de duas hipóteses h1 e h2 pertencentes a H
h1_preds = np.random.randint(0, 2, size=1000000)
h2_preds = np.random.randint(0, 2, size=1000000)

def calcular_erro_amostra(y_true, y_pred):
    # Vetorização: compara todos os elementos de uma vez
    erros = (y_true != y_pred) 
    return np.mean(erros) # Retorna a fração de erros (P)

erro_h1 = calcular_erro_amostra(y_real, h1_preds)
erro_h2 = calcular_erro_amostra(y_real, h2_preds)

print(f"Desempenho P (Erro) da Hipótese h1: {erro_h1:.4f}")
print(f"Desempenho P (Erro) da Hipótese h2: {erro_h2:.4f}")