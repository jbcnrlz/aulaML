import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB # Importação do modelo do Sklearn
import time

# ==========================================
# FUNÇÕES DA CURVA ROC (IMPLEMENTADAS DO ZERO)
# ==========================================

def calcular_roc_do_zero(y_true, y_scores):
    """
    Calcula os pontos da Curva ROC simulando o deslizamento do limiar.
    """
    # 1. Total de positivos e negativos na realidade
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    
    # 2. Extrair limiares únicos e ordenar do maior (mais conservador) para o menor
    limiares = np.unique(y_scores)[::-1]
    
    # Adicionar um limiar infinitamente alto para garantir o ponto inicial (0,0)
    limiares = np.insert(limiares, 0, limiares[0] + 1.0)
    
    fpr_lista = []
    tpr_lista = []
    
    # 3. Testar cada limiar
    for t in limiares:
        # Se o score for maior ou igual ao limiar, a predição é 1 (Positivo)
        previsoes = (y_scores >= t).astype(int)
        
        # Contar TP e FP usando operações booleanas
        TP = np.sum((previsoes == 1) & (y_true == 1))
        FP = np.sum((previsoes == 1) & (y_true == 0))
        
        # Calcular taxas
        TPR = TP / P
        FPR = FP / N
        
        fpr_lista.append(FPR)
        tpr_lista.append(TPR)
        
    return np.array(fpr_lista), np.array(tpr_lista), limiares

def calcular_auc_do_zero(fpr, tpr):
    """
    Calcula a Área sob a Curva usando a Regra do Trapézio.
    Fórmula: Somatório de (Base * Altura Média)
    """
    area = 0.0
    for i in range(1, len(fpr)):
        base = fpr[i] - fpr[i-1]
        altura_media = (tpr[i] + tpr[i-1]) / 2.0
        area += base * altura_media
    return area

# ==========================================
# PREPARAÇÃO E EXECUÇÃO
# ==========================================
print("Baixando MNIST...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Reduzindo para 5000 amostras para execução rápida do loop Python
idx = np.random.choice(len(X), 5000, replace=False)
X, y = X[idx], y[idx]

X = X / 255.0 

# Problema Binário: Classe 1 (Dígito 5) vs Classe 0 (Resto)
y_binario = np.where(y == '5', 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y_binario, test_size=0.3, random_state=42)

print("Treinando o modelo GaussianNB do scikit-learn...")
# Instanciando o modelo do scikit-learn
modelo = GaussianNB()
modelo.fit(X_train, y_train) # Usando .fit() em vez de .efetuarFit()

print("Calculando scores para o teste...")
# predict_proba retorna uma matriz com a probabilidade para cada classe.
# Pegamos a coluna 1, que representa as probabilidades da classe Positiva (Dígito 5)
y_scores = modelo.predict_proba(X_test)[:, 1] 

print("Gerando Curva ROC matemática do zero...")
start_time = time.time()
fpr_custom, tpr_custom, limiares_custom = calcular_roc_do_zero(y_test, y_scores)
auc_custom = calcular_auc_do_zero(fpr_custom, tpr_custom)

# Imprima os resultados da sua função calcular_roc_do_zero
print("Total de pontos na curva:", len(fpr_custom))
print("FPR (5 primeiros):", fpr_custom[:5])
print("TPR (5 primeiros):", tpr_custom[:5])

print(f"Cálculos finalizados em {time.time() - start_time:.2f}s")

print(f"\nPreparando gráfico... AUC calculada: {auc_custom:.3f}")

# ==========================================
# PLOTAGEM
# ==========================================
# 1. Cria a figura
plt.figure(figsize=(8, 6))

# 2. PINTA AS LINHAS PRIMEIRO
plt.plot(fpr_custom, tpr_custom, color='crimson', lw=2, label=f'ROC Matemática (AUC = {auc_custom:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatório (AUC = 0.500)')

# 3. Configura o visual
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
plt.title('Curva ROC e AUC calculadas do zero (Dígito 5 vs Resto)', fontsize=14)
plt.grid(True, alpha=0.3)

# 4. CHAMA A LEGENDA SÓ NO FINAL
plt.legend(loc="lower right")

# 5. Exibe a janela na sua tela
print("Abrindo a janela do gráfico...")
plt.show()