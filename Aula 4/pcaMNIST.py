import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Carregar os dados (MNIST Completo para melhor cálculo do PCA)
print("Carregando MNIST (aguarde)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X = mnist.data
y = mnist.target.astype(int)

# 2. Padronização (Essencial)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Escolher um dígito de exemplo (ex: índice 0, que é um '5')
indice_exemplo = 0
imagem_original = X[indice_exemplo]
imagem_original_scaled = X_scaled[indice_exemplo].reshape(1, -1)
digito = y[indice_exemplo]

# 4. Configurar diferentes níveis de compressão (número de componentes)
n_componentes_lista = [2, 10, 50, 150] # 784 original
reconstrucoes = []

for n in n_componentes_lista:
    # Treinar o PCA com 'n' componentes
    pca = PCA(n_components=n)
    pca.fit(X_scaled) # Treina com todos os dados para definir o espaço
    
    # Projetar o exemplo para o espaço reduzido
    X_reduzido = pca.transform(imagem_original_scaled)
    
    # Projetar de volta para o espaço original (Reconstrução)
    X_reconstruido_scaled = pca.inverse_transform(X_reduzido)
    
    # Desfazer a padronização para visualizar corretamente
    X_reconstruido = scaler.inverse_transform(X_reconstruido_scaled)
    
    # Calcular variância explicada acumulada
    var_acumulada = np.sum(pca.explained_variance_ratio_)
    
    reconstrucoes.append((n, X_reconstruido, var_acumulada))

# 5. Visualização: Antes e Depois da Projeção Inverse
fig, axes = plt.subplots(1, len(n_componentes_lista) + 1, figsize=(20, 4))

# Imagem Original (Todos os 784 atributos)
axes[0].imshow(imagem_original.reshape(28, 28), cmap='gray')
axes[0].set_title(f"Original\n(784 Atributos)")
axes[0].axis('off')

# Imagens Reconstruídas
for i, (n, img_rec, var) in enumerate(reconstrucoes):
    axes[i+1].imshow(img_rec.reshape(28, 28), cmap='gray')
    axes[i+1].set_title(f"{n} Componentes\n({var*100:.1f}% Variância)")
    axes[i+1].axis('off')

plt.suptitle(f"Reconstrução PCA do Dígito {digito}", fontsize=16)
plt.tight_layout()
plt.show()