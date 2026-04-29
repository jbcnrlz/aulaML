import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB # Importação do modelo do Sklearn

def criarRoc(y_true, y_scores):
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    thresholds = np.unique(y_scores)[::-1]
    thresholds = np.insert(thresholds, 0, thresholds[0] + 1)

    tpr = []
    fpr = []

    for t in thresholds:
        previoes = (y_scores >= t).astype(int)
        tp = np.sum((previoes == 1) & (y_true == 1))
        fp = np.sum((previoes == 1) & (y_true == 0))
        tpr.append(tp / P)
        fpr.append(fp / N)

    return np.array(fpr), np.array(tpr), thresholds

def main():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    y = np.where(y == '5', 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    gnb = MultinomialNB() # Criação do modelo
    gnb.fit(X_train, y_train) # Treinamento do modelo
    y_pred = gnb.predict_proba(X_test)[:,1] # Previsão com o modelo treinado
    fpr, tpr, thresholds = criarRoc(y_test, y_pred)
    print(fpr)
    print(tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multinomial Naive Bayes on MNIST')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()