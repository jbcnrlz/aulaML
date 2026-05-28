import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

def euclideana(p1,p2):
    dist = 0
    for i in range(len(p1)):
        dist += (p1[i] - p2[i]) ** 2
    return math.sqrt(dist)

def knn(treino, novoPonto, k):
    #Gerar distancias entre o novo ponto e os pontos de treino
    distancias = []
    for ft, classe in treino:
        dist = euclideana(ft, novoPonto)
        distancias.append((dist, classe))

    #Ordenar as distancias e pegar os k mais proximos
    distancias.sort(key=lambda x : x[0])
    kVizinhos = distancias[:k]

    #Contar os votos dos vizinhos e retornar a classe mais comum
    votos = Counter([classe for _, classe in kVizinhos])
    return votos.most_common(1)[0][0]

def main():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X = MinMaxScaler().fit_transform(X)

    print(X[:5])

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    k = 3
    acertos = 0
    for i in range(len(x_test)):
        predicao = knn(list(zip(x_train, y_train)), x_test[i], k)

        if predicao == y_test[i]:
            acertos += 1

    print(f"Acurácia: {acertos / len(x_test):.2f}")

if __name__ == "__main__":
    main()