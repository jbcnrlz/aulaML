import math
import random
import urllib.request
import matplotlib.pyplot as plt

# 1. Carregar o dataset real (focando em 2 features para permitir o gráfico 2D)
def carregar_iris_2d():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    resposta = urllib.request.urlopen(url)
    linhas = resposta.read().decode('utf-8').splitlines()
    
    dataset = []
    for linha in linhas:
        if not linha: continue
        valores = linha.split(',')
        features = [float(valores[2]), float(valores[3])] # Comprimento e Largura da Pétala
        classe = valores[-1]
        dataset.append((features, classe))
    return dataset

# 2. Divisão de Treino e Teste (Holdout)
def separar_treino_teste(dataset, proporcao_treino=0.8):
    random.seed(42) 
    random.shuffle(dataset)
    indice_corte = int(len(dataset) * proporcao_treino)
    return dataset[:indice_corte], dataset[indice_corte:]

# 3. Distância Euclidiana
def distancia_euclidiana(ponto1, ponto2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(ponto1, ponto2)))

# 4. KNN Ponderado (A Mágica Acontece Aqui)
def knn_ponderado(base_treino, novo_ponto, k):
    distancias = []
    for features_treino, classe in base_treino:
        dist = distancia_euclidiana(novo_ponto, features_treino)
        distancias.append((dist, classe))
    
    # Ordena e pega os K vizinhos mais próximos
    distancias.sort(key=lambda x: x[0])
    k_vizinhos = distancias[:k]
    
    # Votação Ponderada
    votos_ponderados = {}
    epsilon = 1e-10 # Evita divisão por zero se a distância for exatamente 0
    
    for dist, classe in k_vizinhos:
        # Peso = 1 / d^2
        peso = 1.0 / ((dist ** 2) + epsilon)
        
        # Acumula o peso para a respectiva classe
        votos_ponderados[classe] = votos_ponderados.get(classe, 0.0) + peso
        
    # Retorna a classe que acumulou a maior soma de pesos
    classe_vencedora = max(votos_ponderados, key=votos_ponderados.get)
    return classe_vencedora

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    dataset = carregar_iris_2d()
    treino, teste = separar_treino_teste(dataset, 0.8)
    
    k = 5
    acertos = 0
    predicoes = []
    
    # Avaliando a base de teste com o KNN Ponderado
    for features_teste, classe_real in teste:
        classe_predita = knn_ponderado(treino, features_teste, k)
        predicoes.append((features_teste, classe_predita, classe_real))
        
        if classe_predita == classe_real:
            acertos += 1
            
    acuracia = (acertos / len(teste)) * 100
    print(f"Acurácia KNN Ponderado (K={k}): {acuracia:.2f}%\n")

    # --- VISUALIZAÇÃO COM MATPLOTLIB ---
    cores_classes = {
        'Iris-setosa': 'red',
        'Iris-versicolor': 'green',
        'Iris-virginica': 'blue'
    }

    plt.figure(figsize=(10, 6))

    # Treino
    for features, classe in treino:
        plt.scatter(features[0], features[1], c=cores_classes[classe], marker='o', alpha=0.3, s=30)

    # Teste
    for features, classe_predita, classe_real in predicoes:
        cor = cores_classes[classe_predita]
        edge_color = 'black' if classe_predita != classe_real else cor 
        plt.scatter(features[0], features[1], c=cor, marker='*', edgecolors=edge_color, s=150)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, alpha=0.3, label='Treino (Todas as Classes)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=15, label='Teste (Classificado Ponderado)')
    ]

    plt.title(f"Classificação KNN Ponderado (K={k}) - Iris Dataset")
    plt.xlabel("Comprimento da Pétala")
    plt.ylabel("Largura da Pétala")
    plt.legend(handles=legend_elements)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()