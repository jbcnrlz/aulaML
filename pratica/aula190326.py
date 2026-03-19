import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class NaiveBayes:
    def fazerFit(self, X, y):
        nSamples, nFeatures = X.shape
        self.classes = np.unique(y)
        nClasses = len(self.classes)

        self.media = np.zeros((nClasses, nFeatures), dtype=np.float64)
        self.variancia = np.zeros((nClasses, nFeatures), dtype=np.float64)
        self.prior = np.zeros(nClasses, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.media[idx, :] = X_c.mean(axis=0)
            self.variancia[idx, :] = X_c.var(axis=0)
            self.prior[idx] = X_c.shape[0] / float(nSamples)
        
    def predizer(self,X):
        posteriores = []
        for idx, c in enumerate(self.classes):
            priori = np.log(self.prior[idx])
            verossimilhanca = -0.5 * np.sum(np.log(2*np.pi*self.variancia[idx, :]) + ((X - self.media[idx,:])**2 /  self.variancia[idx, :]))

            posteriores.append(priori + verossimilhanca)
        print("Posteriores:", posteriores)
        return self.classes[np.argmax(posteriores)] 
        

def main():
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

    modelo = NaiveBayes()
    modelo.fazerFit(X_train, y_train)

    print("Predição:", modelo.predizer(X_test[0].reshape(1, -1)))
    print("Valor Real:", y_test[0])

if __name__ == "__main__":
    main()
