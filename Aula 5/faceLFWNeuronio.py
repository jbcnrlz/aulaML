import numpy as np
from sklearn.datasets import fetch_lfw_people, fetch_openml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==========================================
# Funções Matemáticas para Classificação
# ==========================================
def softmax(z):
    """
    Transforma as saídas brutas em probabilidades (somam 1).
    Subtraímos o máximo de z por estabilidade numérica (evita overflow no exp).
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    """Calcula a perda (erro) logarítmica entre a previsão e o real."""
    # Clip para evitar log(0) que resulta em NaN
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    # Cálculo vetorizado da Entropia Cruzada
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def to_one_hot(y, num_classes):
    """Converte labels inteiros (ex: 3) em vetores (ex: [0,0,0,1,0])."""
    return np.eye(num_classes)[y.astype(int)]

def main():
    # 1. Carregamento
    print("Carregando LFW...")
    #lfw = fetch_lfw_people(min_faces_per_person=60, resize=0.4)
    lfw = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y_True = lfw.data, lfw.target
    
    #num_classes = len(lfw.target_names)
    num_classes = 10
    m_amostras = X.shape[0]
    print(f"Total de imagens: {m_amostras} | Total de Pessoas (Classes): {num_classes}")

    # 2. Padronização e PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=150)
    X_scaled = pca.fit_transform(X_scaled)

    # 3. Divisão de Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_True, test_size=0.2, random_state=42, stratify=y_True
    )
    
    # Converte o y_train para One-Hot Encoding para facilitar o backpropagation
    y_train_one_hot = to_one_hot(y_train, num_classes)
    for idx, _ in enumerate(y_train[:5]):
        print(f"Exemplo {idx}: Classe Original = {y_train[idx]} | One-Hot = {y_train_one_hot[idx]}")

    input("Pressione Enter para continuar...")
    # 4. Inicialização de Pesos
    # Agora temos uma MATRIZ de pesos: (n_features, n_classes)
    pesos = np.random.randn(X_train.shape[1], num_classes) * 0.01 
    print(pesos)
    input("Pesos inicializados. Pressione Enter para continuar...")
    # E um VETOR de bias: um para cada classe
    bias = np.zeros(num_classes)

    epocas = 5000
    learningRate = 0.1 # Softmax/Cross-Entropy aceita LR maior que a ReLU com MSE

    progresso = tqdm(range(epocas), desc="Treinando Softmax")

    # 5. Loop de Treinamento (Batch Gradient Descent)
    m_train = X_train.shape[0]
    
    for i in progresso:
        # Forward Pass
        z = np.dot(X_train, pesos) + bias
        y_pred = softmax(z) 

        # Cálculo da perda (apenas para monitoramento)
        erro = cross_entropy(y_train_one_hot, y_pred)

        # Backpropagation
        # A derivada da Entropia Cruzada com Softmax simplifica maravilhosamente para (Predição - Real)
        dZ = y_pred - y_train_one_hot
        
        wGrad = np.dot(X_train.T, dZ) / m_train
        bGrad = np.sum(dZ, axis=0) / m_train

        # Atualização dos parâmetros
        pesos -= learningRate * wGrad
        bias -= learningRate * bGrad

        progresso.set_postfix(Loss=f"{erro:.4f}")

    # 6. Avaliação no Teste
    print("\n--- Resultados de Teste ---")
    
    # Processa todas as imagens de teste de uma vez (Vetorização)
    z_test = np.dot(X_test, pesos) + bias
    pred_probs = softmax(z_test)
    
    # Pega o índice da maior probabilidade (quem a rede acha que é)
    pred_classes = np.argmax(pred_probs, axis=1)
    
    acertos = np.sum(pred_classes == y_test)
    total = X_test.shape[0]
    
    print(f"Acurácia: {acertos}/{total} ({(acertos/total)*100:.2f}%)")

if __name__ == "__main__":
    main()