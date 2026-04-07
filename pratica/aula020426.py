import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
class NaiveBayesMultinomial:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def efetuarFit(self, X, y):
        nExemplos, nAtributos = X.shape
        self.classes = np.unique(y)
        nClasses = len(self.classes)

        self.contagemFeatures = np.zeros((nClasses, nAtributos))
        self.totalPalavrasPorClasse = np.zeros(nClasses)
        self.classesPrior = np.zeros(nClasses)

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.contagemFeatures[i, :] = np.sum(X_c, axis=0)
            self.totalPalavrasPorClasse[i] = np.sum(self.contagemFeatures[i, :])
            self.classesPrior[i] = X_c.shape[0] / nExemplos

    def _preverUnico(self, x):
        posteriori = []
        nFeatures = self.contagemFeatures.shape[1]

        for i, c in enumerate(self.classes):
            logPrior = np.log(self.classesPrior[i])

            numerador = self.contagemFeatures[i, :] + self.alpha
            denominador = self.totalPalavrasPorClasse[i] + self.alpha * nFeatures
            logVerosimilhanca = np.sum(x * np.log(numerador / denominador))
            posteriori.append(logPrior + logVerosimilhanca)

        return self.classes[np.argmax(posteriori)], posteriori

if __name__ == "__main__":
    documentos = [
        "Eu amo programação",
        "A programação é divertida",
        "Eu odeio bugs",
        "Bugs são irritantes",
        "Eu adoro aprender",
        "Ai meu deus do céu, roubaram meu fusquinha, eu tô maluco",
        "Não importa o tamanho da varinha, o importante é a magia que ela tem"
    ]

    vectorizer = CountVectorizer(max_features=5000)
    X = vectorizer.fit_transform(documentos).toarray()
    y = np.array([1, 1, 1, 1, 1, 0, 1])
    print("Matriz de características (X):")
    print(X)

    nbm = NaiveBayesMultinomial(alpha=1.0)
    nbm.efetuarFit(X, y)
    print("Contagem de características por classe:")
    print(nbm.contagemFeatures)
    print("Total de palavras por classe:") 
    print(nbm.totalPalavrasPorClasse)
    print("Probabilidades a priori das classes:")
    print(nbm.classesPrior)

    teste = [
        "Eu quero aprender mais sobre programação",
        "Mais vale um passarinho na mão do que dois voando",
    ]
    X_teste = vectorizer.transform(teste).toarray()
    previsao, poteriori = nbm._preverUnico(X_teste[0])
    print(f"Previsão para o teste: {teste[0]} (Classe: {previsao})")
    print(f"Posteriori para o teste: {teste[0]}: {poteriori}")
    previsao, poteriori = nbm._preverUnico(X_teste[1])
    print(f"Previsão para o teste: {teste[1]} (Classe: {previsao})")
    print(f"Posteriori para o teste: {teste[1]}: {poteriori}")
    