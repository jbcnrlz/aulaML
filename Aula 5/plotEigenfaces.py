import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

# ==========================================
# 1. Carregar o banco de dados LFW
# ==========================================
print("Baixando/Carregando o dataset LFW...")
print("(Isso pode levar alguns minutos na primeira execução)")

# Filtramos para incluir apenas pessoas com pelo menos 60 imagens
# O resize reduz o tamanho da imagem para acelerar o processamento
lfw_people = fetch_lfw_people(min_faces_per_person=60, resize=0.4)

n_samples, h, w = lfw_people.images.shape
X = lfw_people.data # Vetores das imagens achatadas (1D)


print(f"\nTotal de imagens carregadas: {n_samples}")
print(f"Dimensões originais de cada imagem: {h}x{w} pixels")
print(f"Total de características originais por imagem: {X.shape[1]}")

# ==========================================
# 2. Aplicar PCA para extrair as características
# ==========================================
# Escolhemos extrair as 150 características principais (redução de dimensionalidade)
n_components = 150 
print(f"\nExtraindo os {n_components} principais componentes (Eigenfaces)...")

# O parâmetro 'whiten=True' ajuda a normalizar as características extraídas
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X)

# As 'eigenfaces' são os componentes principais matemáticos remodelados para o formato de imagem
eigenfaces = pca.components_.reshape((n_components, h, w))

# ==========================================
# 3. Visualizar as características extraídas
# ==========================================
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Função auxiliar para plotar uma galeria de imagens"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()
    
# Visualizar as 12 características mais fundamentais (Eigenfaces)
eigenface_titles = [f"Característica {i+1}" for i in range(eigenfaces.shape[0])]
print("\nAbrindo visualização das características fundamentais...")
plot_gallery(eigenfaces, eigenface_titles, h, w)