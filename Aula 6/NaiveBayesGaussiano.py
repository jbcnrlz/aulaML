import numpy as np, math
from sklearn.datasets import load_iris

class NaiveBayesGaussiano:
    def efetuarFit(self, X, y):
        self.classes = np.unique(y)
        self.parametros = []

        for c in self.classes:
            X_c = X[y == c]
            stats = []
            for col in X_c.T:
                media = np.mean(col)
                variancia = np.var(col)
                stats.append({'media': media, 'variancia': variancia})

            self.parametros.append({
                'stats': stats,
                'prior': len(X_c) / len(X)
            })

    def calcularProbabilidade(self, x, media, variancia):
        eps = 1e-6  # Para evitar divisão por zero
        coeficiente = 1.0 / np.sqrt(2.0 * np.pi * variancia + eps)
        expoente = np.exp(-(math.pow(x - media, 2) / (2 * variancia + eps)))
        return coeficiente * expoente
    
    def prever(self, X):
        previsoes = []
        todaPosteriores = []
        for x in X:
            posteriores = []
            for i, c in enumerate(self.classes):
                probabilidade_c = self.parametros[i]['prior']
                for j, stat in enumerate(self.parametros[i]['stats']):
                    probabilidade_c *= self.calcularProbabilidade(x[j], stat['media'], stat['variancia'])
                posteriores.append(probabilidade_c)
            previsoes.append(self.classes[np.argmax(posteriores)])
            todaPosteriores.append(posteriores)
        return np.array(previsoes), np.array(todaPosteriores)
    

if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target

    modelo = NaiveBayesGaussiano()
    modelo.efetuarFit(X, y)
    previsoes, probabilidades = modelo.prever(X)

    print("Previsões:", previsoes)
    print("Probabilidades:", probabilidades)