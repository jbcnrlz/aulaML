from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

if __name__ == "__main__":
    dados = load_breast_cancer()
    X = dados.data
    y = dados.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_treino, x_teste, y_treino, y_teste = train_test_split(X_scaled, y, test_size=0.5, random_state=42)

    modelos = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'Perceptron': Perceptron(random_state=42,max_iter=1000)
    }

    print("Resultados dos Modelos:")
    for nome, modelo in modelos.items():
        modelo.fit(x_treino, y_treino) #treinando o modelo
        y_pred = modelo.predict(x_teste) #fazendo as predições
        acuracia = accuracy_score(y_teste, y_pred) #calculando a acurácia

        print(f"{nome}: Acurácia = {acuracia:.4f}")

        cm = confusion_matrix(y_teste, y_pred)
        print(f"Matriz de Confusão para {nome}:\n{cm}\n")