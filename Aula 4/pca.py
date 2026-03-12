import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Carregar os dados (usando uma amostra menor para rapidez em sala)
print("Carregando MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X = mnist.data[:2000]  # Usando 2000 amostras para visualização clara
y = mnist.target[:2000].astype(int)

# 2. Padronização (Essencial para PCA)
# Centraliza na média e escala para variância unitária
X_scaled = StandardScaler().fit_transform(X)

# 3. Aplicar PCA para reduzir de 784 para 2 dimensões
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Visualização Antes (Exemplo de Imagem) e Depois (Gráfico 2D)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Lado Esquerdo: Como o dado "parece" (28x28)
img_idx = 0
ax[0].imshow(X[img_idx].reshape(28, 28), cmap='gray')
ax[0].set_title(f"Dado Original (Alta Dimensão: 784)\nDígito: {y[img_idx]}")
ax[0].axis('off')

# Lado Direito: Projeção de todos os dados no plano 2D
scatter = ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7, s=10)
legend1 = ax[1].legend(*scatter.legend_elements(), title="Dígitos", loc="best")
ax[1].add_artist(legend1)
ax[1].set_title("Dados Projetados via PCA (2 Dimensões)")
ax[1].set_xlabel("Componente Principal 1")
ax[1].set_ylabel("Componente Principal 2")
ax[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# 5. Variância Explicada
var_exp = pca.explained_variance_ratio_
print(f"Variância capturada pelo PC1: {var_exp[0]*100:.2f}%")
print(f"Variância capturada pelo PC2: {var_exp[1]*100:.2f}%")
print(f"Total de informação preservada em 2D: {sum(var_exp)*100:.2f}%")