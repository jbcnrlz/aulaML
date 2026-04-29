import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score
from scipy.stats import t

def main():
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data, mnist.target

    X_scaled = MinMaxScaler().fit_transform(X)

    K = 5
    kf = KFold(n_splits=K, shuffle=True, random_state=42)

    gnb = GaussianNB()
    mnb = MultinomialNB()

    diferencas = []
    for i, (tIdx,testIdx) in enumerate(kf.split(X_scaled)):
        X_train, y_train = X_scaled[tIdx], y[tIdx]
        X_test, y_test = X_scaled[testIdx], y[testIdx]

        gnb.fit(X_train, y_train)
        mnb.fit(X_train, y_train)

        y_pred_gnb = gnb.predict(X_test)
        y_pred_mnb = mnb.predict(X_test)

        acc_gnb = accuracy_score(y_test, y_pred_gnb)
        acc_mnb = accuracy_score(y_test, y_pred_mnb)

        diferenca = acc_gnb - acc_mnb
        diferencas.append(diferenca)
        print(f"Fold {i+1}: GNB Accuracy = {acc_gnb:.4f}, MNB Accuracy = {acc_mnb:.4f}, Difference = {diferenca:.4f}")

    diferencas = np.array(diferencas)

    #Teste t de Student para Amostras Pareadas
    d = np.mean(diferencas)
    S2 = np.var(diferencas, ddof=1)
    var_padrao = S2 / K
    t_padrao = d / np.sqrt(var_padrao)
    p_value = 2 * (1 - t.cdf(abs(t_padrao), df=K-1))

    print("===Resultados do Teste t de Student para Amostras Pareadas===")
    print(f" Média das Diferenças: {d:.4f}")
    print(f" Variância das Diferenças: {S2:.4f}")
    print(f" Estatística t: {t_padrao:.4f}")
    print(f" Valor-p: {p_value}")

    #Teste t de Student corrigido para Amostras Pareadas
    n_test = len(X) / K
    n_train = len(X) - n_test
    fator_correcao = (1/K) + (n_test/n_train)
    var_corrigida = S2 * fator_correcao
    t_corrigida = d / np.sqrt(var_corrigida)
    p_value_corrigida = 2 * (1 - t.cdf(abs(t_corrigida), df=K-1))

    print("===Resultados do Teste t de Student corrigido para Amostras Pareadas===")
    print(f" Média das Diferenças: {d:.4f}")
    print(f" Variância das Diferenças: {var_corrigida:.4f}")
    print(f" Estatística t: {t_corrigida:.4f}")
    print(f" Valor-p: {p_value_corrigida}")

if __name__ == "__main__":
    main()