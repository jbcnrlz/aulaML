import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. CARREGAR E PREPARAR OS DADOS
dados = load_breast_cancer()
X = dados.data
y = dados.target

# O Neurônio (Perceptron) é muito sensível à escala dos dados.
# Árvores e Naive Bayes não ligam para isso, mas padronizar é essencial para o Perceptron.
scaler = StandardScaler()
X_escalado = scaler.fit_transform(X)

X_treino, X_teste, y_treino, y_teste = train_test_split(X_escalado, y, test_size=0.3, random_state=42)

# 2. INICIALIZAR OS MODELOS
modelos = {
    "Árvore de Decisão": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Naive Bayes (Gaussian)": GaussianNB(),
    "Neurônio (Perceptron)": Perceptron(random_state=42, max_iter=1000)
}

# 3. TREINAR E AVALIAR CADA MODELO
print("=== RESULTADOS DA COMPARAÇÃO ===\n")

for nome, modelo in modelos.items():
    inicio = time.time()
    
    # Treinamento
    modelo.fit(X_treino, y_treino)
    
    # Previsão
    previsoes = modelo.predict(X_teste)
    
    fim = time.time()
    
    # Avaliação
    acuracia = accuracy_score(y_teste, previsoes)
    tempo_execucao = (fim - inicio) * 1000 # Convertendo para milissegundos
    
    print(f"Modelo: {nome}")
    print(f"Acurácia: {acuracia * 100:.2f}%")
    print(f"Tempo de treino/previsão: {tempo_execucao:.2f} ms")
    print("-" * 30)