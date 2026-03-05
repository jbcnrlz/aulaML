import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler

# 1. Carregando o MNIST (pode demorar alguns segundos na primeira vez)
print("Carregando dados...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data, mnist.target

# 2. Normalização (Essencial em ML)
# Transformamos os pixels (0-255) para o intervalo [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

def similaridade_cosseno(v1, v2):
    """Implementação da fórmula apresentada em aula"""
    # Produto interno <x, y> [cite: 4, 11]
    dot_product = np.dot(v1, v2)
    
    # Normas L2 (Euclidianas) [cite: 9, 212]
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    return dot_product / (norm_v1 * norm_v2)

# 3. Seleção de exemplos para teste
# Vamos pegar dois '5's e um '0' para comparar
idx_5a = np.where(y == '5')[0][0]
idx_5b = np.where(y == '5')[0][1]
idx_0  = np.where(y == '0')[0][0]

v_5a = X_scaled[idx_5a]
v_5b = X_scaled[idx_5b]
v_0  = X_scaled[idx_0]

# 4. Cálculo das Similaridades
sim_mesma_classe = similaridade_cosseno(v_5a, v_5b)
sim_classes_dif = similaridade_cosseno(v_5a, v_0)

# 5. Visualização dos Resultados
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
titles = [f"Referência (Dígito {y[idx_5a]})", f"Mesma Classe (Dígito {y[idx_5b]})", f"Outra Classe (Dígito {y[idx_0]})"]
images = [v_5a, v_5b, v_0]

for i, ax in enumerate(axes):
    ax.imshow(images[i].reshape(28, 28), cmap='gray')
    ax.set_title(titles[i])
    ax.axis('off')

plt.tight_layout()
plt.show()

print(f"Similaridade entre dois '5's: {sim_mesma_classe:.4f}")
print(f"Similaridade entre um '5' e um '0': {sim_classes_dif:.4f}")