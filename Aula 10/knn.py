import math
import random
import urllib.request
from collections import Counter
import matplotlib.pyplot as plt

# 1. Carregar o dataset real (focando em 2 features para permitir o gráfico 2D)
def carregar_iris_2d():
    """Baixa o Iris Dataset e extrai apenas o Comprimento e Largura da Pétala."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    print("Baixando o dataset Iris...")
    resposta = urllib.request.urlopen(url)
    linhas = resposta.read().decode('utf-8').splitlines()
    
    dataset = []
    for linha in linhas:
        if not linha: continue
        valores = linha.split(',')
        # Índices 2 e 3 correspondem ao Comprimento e Largura da Pétala
        features = [float(valores[2]), float(valores[3])]
        classe = valores[-1]
        dataset.append((features, classe))
    return dataset

# 2. Divisão de Treino e Teste (Holdout)
def separar_treino_teste(dataset, proporcao_treino=0.8):
    """Embaralha os dados e separa em treino e teste."""
    random.seed(42) # Semente para garantir que o resultado seja reproduzível
    random.shuffle(dataset)
    
    indice_corte = int(len(dataset) * proporcao_treino)
    treino = dataset[:indice_corte]
    teste = dataset[indice_corte:]
    
    return treino, teste

# 3. Distância Euclidiana
def distancia_euclidiana(ponto1, ponto2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(ponto1, ponto2)))

# 4. KNN Puro
def knn(base_treino, novo_ponto, k):
    distancias = []
    for features_treino, classe in base_treino:
        dist = distancia_euclidiana(novo_ponto, features_treino)
        distancias.append((dist, classe))
    
    distancias.sort(key=lambda x: x[0])
    k_vizinhos = distancias[:k]
    
    votos = Counter([classe for _, classe in k_vizinhos])
    return votos.most_common(1)[0][0]

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    dataset = carregar_iris_2d()
    
    # Separando 80% para treino e 20% para teste
    treino, teste = separar_treino_teste(dataset, 0.8)
    
    k = 5
    acertos = 0
    predicoes = []
    
    print("-" * 40)
    print(f"Total de dados : {len(dataset)}")
    print(f"Base de Treino : {len(treino)}")
    print(f"Base de Teste  : {len(teste)}")
    print("-" * 40)
    
    # Avaliando a base de teste
    for features_teste, classe_real in teste:
        classe_predita = knn(treino, features_teste, k)
        predicoes.append((features_teste, classe_predita, classe_real))
        
        if classe_predita == classe_real:
            acertos += 1
            
    acuracia = (acertos / len(teste)) * 100
    print(f"Acurácia (K={k}): {acuracia:.2f}%\n")
    print("Gerando gráfico...")

    # --- VISUALIZAÇÃO COM MATPLOTLIB ---
    cores_classes = {
        'Iris-setosa': 'red',
        'Iris-versicolor': 'green',
        'Iris-virginica': 'blue'
    }

    plt.figure(figsize=(10, 6))

    # Plotar pontos de TREINO (bolinhas menores e um pouco transparentes)
    for features, classe in treino:
        plt.scatter(features[0], features[1], c=cores_classes[classe], marker='o', alpha=0.3, s=30)

    # Plotar pontos de TESTE (Estrelas opacas)
    for features, classe_predita, classe_real in predicoes:
        cor = cores_classes[classe_predita]
        # Borda preta se errou, borda da mesma cor se acertou
        edge_color = 'black' if classe_predita != classe_real else cor 
        plt.scatter(features[0], features[1], c=cor, marker='*', edgecolors=edge_color, s=150)

    # Legendas customizadas
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, alpha=0.3, label='Treino (Todas as Classes)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=15, label='Teste (Classificado pelo KNN)')
    ]

    plt.title(f"Classificação KNN (K={k}) - Iris Dataset (Pétalas)")
    plt.xlabel("Comprimento da Pétala")
    plt.ylabel("Largura da Pétala")
    plt.legend(handles=legend_elements)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()