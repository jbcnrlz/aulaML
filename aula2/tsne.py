# -*- coding: utf-8 -*-
"""
Aplicação de t‑SNE no MNIST usando fetch_openml
- Carrega o MNIST (versão 'mnist_784') do OpenML
- Reduz a dimensionalidade para 2D via t‑SNE
- Visualiza os dígitos com cores correspondentes aos rótulos
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
import time

# 1. Carregar o MNIST do OpenML
print("Carregando MNIST do OpenML...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
# X: array de forma (70000, 784) com valores entre 0 e 255
# y: array de strings, vamos converter para inteiros
y = y.astype(np.uint8)

# Normalizar os valores dos pixels para o intervalo [0, 1]
X = X / 255.0

print(f"Forma dos dados completos: {X.shape}")
print(f"Rótulos presentes: {np.unique(y)}")

# 2. Subamostrar para acelerar o t‑SNE (opcional, mas recomendado)
n_samples = 5000   # Número de pontos a serem usados (ajuste conforme necessário)
np.random.seed(42)
indices = np.random.choice(X.shape[0], n_samples, replace=False)
X_sample = X[indices]
y_sample = y[indices]

print(f"Usando amostra de {n_samples} pontos: {X_sample.shape}")

# 3. Aplicar t‑SNE
print("Executando t‑SNE... (pode levar alguns minutos)")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, verbose=1)
start = time.time()
X_tsne = tsne.fit_transform(X_sample)
end = time.time()
print(f"t‑SNE concluído em {end - start:.2f} segundos.")

# 4. Visualização
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='tab10', s=5, alpha=0.7)
plt.colorbar(scatter, ticks=range(10), label='Dígito')
plt.title(f'Visualização t‑SNE do MNIST (amostra de {n_samples} imagens)')
plt.xlabel('Componente t‑SNE 1')
plt.ylabel('Componente t‑SNE 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()