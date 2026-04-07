import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class NaiveBayesGaussianoVetorizado:
    def efetuarFit(self, X, y):
        self.classes = np.unique(y)
        self.parametros = {} # Usar um dicionário facilita o acesso aos parâmetros de cada classe

        for c in self.classes:
            # Filtramos as instâncias da classe atual
            X_c = X[y == c]
            
            # VETORIZAÇÃO: O parâmetro axis=0 diz ao NumPy para calcular 
            # a média e a variância de todas as 784 colunas (pixels) de uma só vez!
            media = np.mean(X_c, axis=0) 
            variancia = np.var(X_c, axis=0)
            prior = len(X_c) / len(X)

            self.parametros[c] = {
                'media': media,
                'variancia': variancia,
                'prior': prior
            }

    def previsao(self, X):
        # Aqui vamos guardar as previsões finais e as matrizes de score
        todasPosteriores = []
        eps = 1e-6

        # Em vez de um loop para cada registro, iteramos apenas sobre as 10 classes
        for c in self.classes:
            media = self.parametros[c]['media']
            variancia = self.parametros[c]['variancia']
            prior = self.parametros[c]['prior']

            # MATEMÁTICA VETORIZADA PARA TODOS OS REGISTROS SIMULTANEAMENTE:
            # np.sum(..., axis=1) soma os valores ao longo das 784 colunas, 
            # gerando um único array 1D com um score para cada linha do dataset de teste.
            
            termo1 = -0.5 * np.sum(np.log(2 * np.pi * variancia + eps))
            termo2 = -0.5 * np.sum(((X - media) ** 2) / (variancia + eps), axis=1)
            
            # log_prob é um array com milhares de pontuações (uma para cada registro em X)
            log_prob = termo1 + termo2 + np.log(prior)
            
            todasPosteriores.append(log_prob)

        # Transformamos a lista em array. O formato atual é (10_classes, Num_Registros).
        # Usamos .T (Transposta) para girar a matriz para (Num_Registros, 10_classes).
        todasPosteriores = np.array(todasPosteriores).T

        # Pega a classe com a maior pontuação para cada linha de uma vez só
        indices_max = np.argmax(todasPosteriores, axis=1)
        previsoes = self.classes[indices_max]

        return previsoes, todasPosteriores

if __name__ == '__main__':
    print("Carregando o dataset MNIST...")
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data, mnist.target

    print("Pré-processando os dados...")
    scal = StandardScaler()
    X_scaled = scal.fit_transform(X)
    
    pca = PCA(n_components=20)
    X_scaled = pca.fit_transform(X_scaled)

    print("Dividindo os dados em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("Treinando o modelo...")
    nbg = NaiveBayesGaussianoVetorizado()
    nbg.efetuarFit(X_train, y_train) 
    
    print("Fazendo previsões vetorizadas (agora vai ser rápido!)...")
    previsoes, todasPosteriores = nbg.previsao(X_test)

    print("Calculando a acurácia...")
    # Avaliação usando operações vetorizadas do NumPy
    y_test_array = np.array(y_test)
    acertos = np.sum(previsoes == y_test_array)
    acuracia = acertos / len(y_test_array)
    
    print(f"Acurácia: {acuracia * 100:.2f}%")