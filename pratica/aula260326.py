import numpy as np, math

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
                stats.append({"media" : media, "variancia" : variancia})

            self.parametros.append({
                'stats' : stats,
                'prior' : len(X_c) / len(X)
            })

    def calcularProbabilidade(self, x, media, variancia):
        eps = 1e-6
        coeficiente = 1.0 / np.sqrt(2*np.pi*variancia+eps)
        expoente = np.exp(-(math.pow(x - media,2) / (2* variancia + eps)))
        return coeficiente * expoente

    def previsao(self,X):
        previsoes = []
        todasPosteriores = []
        for x in X:
            for i, c in enumerate(self.classes):
                probPrior = self.parametros[i]['prior']
                for j, stat in enumerate(self.parametros[i]['stats']):
                    probPrior *= self.calcularProbabilidade(x[j],stat['media'],stat['variancia'])                
            todasPosteriores.append(probPrior)
            previsoes.append(self.classes[np.argmax(probPrior)])

        return np.array(previsoes), np.array(todasPosteriores)