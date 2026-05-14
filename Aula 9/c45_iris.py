import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# =====================================================================
# 1. CARREGAMENTO E DIVISÃO DO DATASET (Iris)
# =====================================================================
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Especie'] = [iris.target_names[i] for i in iris.target] 

# Partição Treino/Teste (80% treino, 20% teste)
df_treino, df_teste = train_test_split(df, test_size=0.2, random_state=42)

# =====================================================================
# 2. FUNÇÕES MATEMÁTICAS: ÍNDICE GINI
# =====================================================================
def calcular_gini(coluna_alvo):
    """
    Calcula a Impureza de Gini: 1 - sum(p_i^2)
    """
    proporcoes = coluna_alvo.value_counts(normalize=True)
    return 1.0 - sum(p**2 for p in proporcoes)

def encontrar_melhor_limiar_gini(df, atributo, alvo):
    """
    Testa limiares contínuos buscando a maior Redução de Impureza de Gini.
    """
    valores_unicos = np.sort(df[atributo].unique())
    melhor_reducao_gini = -1
    melhor_limiar = None
    gini_base = calcular_gini(df[alvo])
    
    for i in range(len(valores_unicos) - 1):
        limiar = (valores_unicos[i] + valores_unicos[i+1]) / 2.0
        
        subset_esq = df[df[atributo] <= limiar][alvo]
        subset_dir = df[df[atributo] > limiar][alvo]
        
        peso_esq = len(subset_esq) / len(df)
        peso_dir = len(subset_dir) / len(df)
        
        # Gini ponderado dos subconjuntos
        gini_ponderado = (peso_esq * calcular_gini(subset_esq)) + \
                         (peso_dir * calcular_gini(subset_dir))
        
        # Redução de Impureza (análogo ao Ganho de Informação)
        reducao_gini = gini_base - gini_ponderado
        
        if reducao_gini > melhor_reducao_gini:
            melhor_reducao_gini = reducao_gini
            melhor_limiar = limiar
            
    return melhor_limiar, melhor_reducao_gini

# =====================================================================
# 3. CONSTRUÇÃO DA ÁRVORE (CART Style)
# =====================================================================
def construir_arvore_gini(df, atributos, alvo, profundidade_atual=0, max_profundidade=4):
    # CASOS BASE
    if len(df[alvo].unique()) == 1: # Nó puro
        return df[alvo].iloc[0]
    if profundidade_atual >= max_profundidade:
        return df[alvo].mode()[0]
        
    melhor_reducao_global = -1
    melhor_atributo_global = None
    melhor_limiar_global = None
    
    for atr in atributos:
        limiar, reducao_gini = encontrar_melhor_limiar_gini(df, atr, alvo)
        if reducao_gini > melhor_reducao_global:
            melhor_reducao_global = reducao_gini
            melhor_atributo_global = atr
            melhor_limiar_global = limiar
            
    # Critério de parada: se não há redução de impureza
    if melhor_reducao_global <= 0:
        return df[alvo].mode()[0]
        
    arvore = {melhor_atributo_global: {'limiar': melhor_limiar_global, '<=': {}, '>': {}}}
    
    df_esq = df[df[melhor_atributo_global] <= melhor_limiar_global]
    df_dir = df[df[melhor_atributo_global] > melhor_limiar_global]
    
    arvore[melhor_atributo_global]['<='] = construir_arvore_gini(df_esq, atributos, alvo, profundidade_atual + 1, max_profundidade)
    arvore[melhor_atributo_global]['>']  = construir_arvore_gini(df_dir, atributos, alvo, profundidade_atual + 1, max_profundidade)
    
    return arvore

def prever_amostra(arvore, amostra):
    if not isinstance(arvore, dict):
        return arvore
    atributo = list(arvore.keys())[0]
    limiar = arvore[atributo]['limiar']
    if amostra[atributo] <= limiar:
        return prever_amostra(arvore[atributo]['<='], amostra)
    else:
        return prever_amostra(arvore[atributo]['>'], amostra)

def avaliar_modelo(arvore, df_teste, alvo):
    predicoes = [prever_amostra(arvore, linha) for _, linha in df_teste.iterrows()]
    acuracia = sum(predicoes == df_teste[alvo]) / len(df_teste)
    return acuracia

# =====================================================================
# 4. VISUALIZAÇÃO COM MATPLOTLIB
# =====================================================================
def desenhar_no_binario(ax, texto, centro, pai_coord, rotulo_ramo, is_folha=False):
    if is_folha:
        estilo_caixa = dict(boxstyle="round,pad=0.4", fc="#b2df8a", ec="black", lw=1.2)
    else:
        estilo_caixa = dict(boxstyle="square,pad=0.4", fc="#fb9a99", ec="black", lw=1.2) # Mudei para vermelho claro para diferenciar

    if pai_coord is not None:
        ax.annotate(texto, xy=pai_coord, xytext=centro,
                    arrowprops=dict(arrowstyle="<-", color="black", lw=1.5),
                    bbox=estilo_caixa, ha='center', va='center', fontsize=9, weight='bold')
        
        x_meio = (centro[0] + pai_coord[0]) / 2
        y_meio = (centro[1] + pai_coord[1]) / 2
        ax.text(x_meio, y_meio, rotulo_ramo, ha='center', va='center', 
                fontsize=8, color="#d95f02", weight='bold',
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.9))
    else:
        ax.text(centro[0], centro[1], texto, ha='center', va='center', 
                fontsize=10, weight='bold', bbox=estilo_caixa)

def plotar_arvore_recursiva(arvore, ax, x, y, dx, dy, pai_coord=None, rotulo_ramo=""):
    if not isinstance(arvore, dict):
        desenhar_no_binario(ax, str(arvore).capitalize(), (x, y), pai_coord, rotulo_ramo, is_folha=True)
        return

    atributo = list(arvore.keys())[0]
    limiar = arvore[atributo]['limiar']
    
    texto_no = atributo.replace(" (cm)", "")
    desenhar_no_binario(ax, texto_no, (x, y), pai_coord, rotulo_ramo, is_folha=False)
    
    ramo_esq = arvore[atributo]['<=']
    ramo_dir = arvore[atributo]['>']
    
    plotar_arvore_recursiva(ramo_esq, ax, x - dx, y - dy, dx / 1.6, dy, 
                            pai_coord=(x, y), rotulo_ramo=f"<= {limiar:.2f}")
    
    plotar_arvore_recursiva(ramo_dir, ax, x + dx, y - dy, dx / 1.6, dy, 
                            pai_coord=(x, y), rotulo_ramo=f"> {limiar:.2f}")

# =====================================================================
# 5. EXECUÇÃO
# =====================================================================
if __name__ == "__main__":
    alvo = 'Especie'
    atributos_disponiveis = iris.feature_names
    
    print("Treinando modelo com Critério GINI...")
    modelo_gini = construir_arvore_gini(df_treino, atributos_disponiveis, alvo, max_profundidade=4)
    
    acuracia = avaliar_modelo(modelo_gini, df_teste, alvo)
    print(f"Acurácia no Teste: {acuracia * 100:.2f}%\nGerando o gráfico...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')
    
    plotar_arvore_recursiva(modelo_gini, ax, x=0.5, y=1.0, dx=0.25, dy=0.15)
    
    plt.title(f"Árvore de Decisão (Critério: Impureza de Gini)\nAcurácia Teste: {acuracia*100:.2f}%", 
              fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.show()