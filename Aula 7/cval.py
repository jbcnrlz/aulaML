import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from scipy.stats import t
import time

print("Baixando o dataset MNIST... (isso pode levar um minuto)")
# Pegando o MNIST inteiro (70.000 imagens, 784 pixels)
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

# O Multinomial trabalha melhor com features inteiras (contagens). 
# Opcional, mas comum no MNIST: normalizar os pixels de 0-255 para 0-1
X = X / 255.0

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

gnb = GaussianNB()
mnb = MultinomialNB()

differences = []

print("\n=== Iniciando o 5-Fold CV ===")
start_time = time.time()

for i, (train_idx, test_idx) in enumerate(kf.split(X)):
    fold_start = time.time()
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Gaussian NB
    gnb.fit(X_train, y_train)
    acc_gnb = accuracy_score(y_test, gnb.predict(X_test))
    
    # Multinomial NB
    mnb.fit(X_train, y_train)
    acc_mnb = accuracy_score(y_test, mnb.predict(X_test))
    
    # Diferença (Neste caso, vamos olhar Multinomial - Gaussiano, pois esperamos que o MNB ganhe)
    diff = acc_mnb - acc_gnb
    differences.append(diff)
    
    print(f"Fold {i+1} | MNB: {acc_mnb:.3f} | GNB: {acc_gnb:.3f} | Diferença: {diff:.3f} | Tempo: {time.time()-fold_start:.1f}s")

differences = np.array(differences)

d_bar = np.mean(differences)
S2 = np.var(differences, ddof=1)

n_test = len(X) / K          # 14.000
n_train = len(X) - n_test    # 56.000

# Teste t Padrão
var_padrao = S2 / K
t_padrao = d_bar / np.sqrt(var_padrao)
p_padrao = 2 * (1 - t.cdf(abs(t_padrao), df=K-1))

# Teste t Corrigido (Nadeau-Bengio)
fator_correcao = (1/K) + (n_test / n_train) # 0.2 + (14000/56000) = 0.45
var_corrigida = S2 * fator_correcao
t_corrigido = d_bar / np.sqrt(var_corrigida)
p_corrigido = 2 * (1 - t.cdf(abs(t_corrigido), df=K-1))

print("\n=== Resultados Estatísticos (MNB vs GNB) ===")
print(f"Diferença Média de Acurácia: {d_bar:.3f} ({d_bar*100:.1f}%) a favor do MNB")
print(f"Variância Amostral (S²):     {S2:.7f}")
print("-" * 45)
print("Teste t Padrão:")
print(f"Valor t: {t_padrao:.4f} | Valor p: {p_padrao:.10f}")
print("-" * 45)
print("Teste t Corrigido (Nadeau-Bengio):")
print(f"Valor t: {t_corrigido:.4f} | Valor p: {p_corrigido:.10f}")
print(f"Tempo Total: {time.time()-start_time:.1f}s")