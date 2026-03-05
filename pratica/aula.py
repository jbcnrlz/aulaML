import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler

def simCoss(v1,v2):
    dprod = np.dot(v1,v2)
    mod1 = np.linalg.norm(v1)
    mod2 = np.linalg.norm(v2)
    return dprod/(mod1*mod2)

if __name__ == "__main__":
    print("Carregando o dataset MNIST...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data, mnist.target
    print(X)
    print(y)

    minMax = MinMaxScaler()
    X_scaled = minMax.fit_transform(X)

    idx_5a =np.where(y == "5")[0][0]
    idx_5b =np.where(y == "5")[0][1]
    idx_0 = np.where(y == "0")[0][0]

    feats_5a = X_scaled[idx_5a]
    feats_5b = X_scaled[idx_5b]
    feats_0 = X_scaled[idx_0]

    sim_mesmaclasse = simCoss(feats_5a, feats_5b)
    sim_diferentesclasses = simCoss(feats_5a, feats_0)
    print(f"Similaridade entre as duas imagens de 5: {sim_mesmaclasse:.4f}")
    print(f"Similaridade entre imagem de 5 e imagem de 0: {sim_diferentesclasses:.4f}")