import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn import metrics

class NaiveBayesMultinomial:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def efetuarFit(self, X, y):
        nExemplos, nAtributos = X.shape
        self.classes = np.unique(y)
        nClasses = len(self.classes)

        self.contagemFeatures = np.zeros((nClasses, nAtributos))
        self.totalPalavrasPorClasse = np.zeros(nClasses)
        self.classesPrioris = np.zeros(nClasses)

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            # Somas das contagens de cada termo para a classe c
            self.contagemFeatures[i, :] = np.array(X_c.sum(axis=0)) 
            # Total de termos na classe (para o denominador)
            self.totalPalavrasPorClasse[i] = self.contagemFeatures[i, :].sum()
            # CORREÇÃO: Usar .shape[0] em vez de len()
            self.classesPrioris[i] = X_c.shape[0] / nExemplos

    def _preverUnico(self, x):
        # Se x for esparso (uma linha da matriz), transformamos em denso para o cálculo
        if hasattr(x, "toarray"):
            x = x.toarray().flatten()
            
        posterioris = []
        n_features = self.contagemFeatures.shape[1]

        for i, c in enumerate(self.classes):
            logPrior = np.log(self.classesPrioris[i])
            
            # Suavização de Laplace na verossimilhança
            numerador = self.contagemFeatures[i, :] + self.alpha
            denominador = self.totalPalavrasPorClasse[i] + (self.alpha * n_features)
            
            logVerossimilhanca = np.log(numerador / denominador)
            
            # Somatório dos logs (equivalente à multiplicação das probabilidades)
            logPosteriori = logPrior + np.sum(x * logVerossimilhanca)
            posterioris.append(logPosteriori)
            
        return self.classes[np.argmax(posterioris)]
    
    def predizer(self, X):
        # Itera sobre as linhas da matriz esparsa
        return [self._preverUnico(X[i]) for i in range(X.shape[0])]

if __name__ == '__main__':
    # 1. Dados
    categorias = ['sci.med', 'sci.electronics']
    dados = fetch_20newsgroups(subset='all', categories=categorias, remove=('headers', 'footers', 'quotes'))

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(dados.data, dados.target, test_size=0.25, random_state=42)

    # 2. Vetorização
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X_train_counts = vectorizer.fit_transform(X_train_raw)
    X_test_counts = vectorizer.transform(X_test_raw)

    # 3. Treino e Predição
    modelo = NaiveBayesMultinomial()
    modelo.efetuarFit(X_train_counts, y_train)
    
    # Chamada corrigida para o nome do método 'predizer'
    y_pred = modelo.predizer(X_test_counts)

    # 4. Resultados
    print(f"Acurácia: {metrics.accuracy_score(y_test, y_pred):.2%}")
    print("\nRelatório de Classificação:")
    print(metrics.classification_report(y_test, y_pred, target_names=dados.target_names))