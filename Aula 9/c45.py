import pandas as pd
import numpy as np
import math
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
# 2. FUNÇÕES MATEMÁTICAS E TRATAMENTO CONTÍNUO (C4.5)
# =====================================================================
def calcular_entropia(coluna_alvo):
    proporcoes = coluna_alvo.value_counts(normalize=True)
    return -sum(p * math.log2(p) for p in proporcoes)

def encontrar_melhor_limiar(df, atributo, alvo):
    valores_unicos = np.sort(df[atributo].unique())
    melhor_ganho = -1
    melhor_limiar = None
    entropia_base = calcular_entropia(df[alvo])
    
    for i in range(len(valores_unicos) - 1):
        limiar = (valores_unicos[i] + valores_unicos[i+1]) / 2.0
        
        subset_esq = df[df[atributo] <= limiar][alvo]
        subset_dir = df[df[atributo] > limiar][alvo]
        
        peso_esq = len(subset_esq) / len(df)
        peso_dir = len(subset_dir) / len(df)
        entropia_ponderada = (peso_esq * calcular_entropia(subset_esq)) + \
                             (peso_dir * calcular_entropia(subset_dir))
        
        ganho = entropia_base - entropia_ponderada
        
        if ganho > melhor_ganho:
            melhor_ganho = ganho
            melhor_limiar = limiar
            
    return melhor_limiar, melhor_ganho

# =====================================================================
# 3. ALGORITMO C4.5 (Construção da Árvore)
# =====================================================================
def c45(df, atributos, alvo, profundidade_atual=0, max_profundidade=4):
    # CASOS BASE
    if len(df[alvo].unique()) == 1:
        return df[alvo].iloc[0]
    if profundidade_atual >= max_profundidade:
        return df[alvo].mode()[0]
        
    melhor_ganho_global = -1
    melhor_atributo_global = None
    melhor_limiar_global = None
    
    for atr in atributos:
        limiar, ganho = encontrar_melhor_limiar(df, atr, alvo)
        if ganho > melhor_ganho_global:
            melhor_ganho_global = ganho
            melhor_atributo_global = atr
            melhor_limiar_global = limiar
            
    if melhor_ganho_global <= 0:
        return df[alvo].mode()[0]
        
    # Inicializa o Nó (Guardando o atributo e o valor de corte)
    arvore = {melhor_atributo_global: {'limiar': melhor_limiar_global, '<=': {}, '>': {}}}
    
    df_esq = df[df[melhor_atributo_global] <= melhor_limiar_global]
    df_dir = df[df[melhor_atributo_global] > melhor_limiar_global]
    
    arvore[melhor_atributo_global]['<='] = c45(df_esq, atributos, alvo, profundidade_atual + 1, max_profundidade)
    arvore[melhor_atributo_global]['>']  = c45(df_dir, atributos, alvo, profundidade_atual + 1, max_profundidade)
    
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
# 4. VISUALIZAÇÃO COM MATPLOTLIB (Adaptado para C4.5 / Binário)
# =====================================================================
def desenhar_no_c45(ax, texto, centro, pai_coord, rotulo_ramo, is_folha=False):
    """Desenha as caixas e as setas, ajustando as cores para folhas e nós."""
    if is_folha:
        estilo_caixa = dict(boxstyle="round,pad=0.4", fc="#b2df8a", ec="black", lw=1.2)
    else:
        estilo_caixa = dict(boxstyle="square,pad=0.4", fc="#a6cee3", ec="black", lw=1.2)

    if pai_coord is not None:
        ax.annotate(texto, xy=pai_coord, xytext=centro,
                    arrowprops=dict(arrowstyle="<-", color="black", lw=1.5),
                    bbox=estilo_caixa, ha='center', va='center', fontsize=9, weight='bold')
        
        # Posiciona o rótulo da regra (ex: <= 2.45) no meio da seta
        x_meio = (centro[0] + pai_coord[0]) / 2
        y_meio = (centro[1] + pai_coord[1]) / 2
        ax.text(x_meio, y_meio, rotulo_ramo, ha='center', va='center', 
                fontsize=8, color="#d95f02", weight='bold',
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.9))
    else:
        ax.text(centro[0], centro[1], texto, ha='center', va='center', 
                fontsize=10, weight='bold', bbox=estilo_caixa)

def plotar_arvore_c45_recursiva(arvore, ax, x, y, dx, dy, pai_coord=None, rotulo_ramo=""):
    """Caminha pela árvore binária do C4.5."""
    # CASO BASE: Nó folha
    if not isinstance(arvore, dict):
        desenhar_no_c45(ax, str(arvore).capitalize(), (x, y), pai_coord, rotulo_ramo, is_folha=True)
        return

    # Nó de Decisão (Atributo)
    atributo = list(arvore.keys())[0]
    limiar = arvore[atributo]['limiar']
    
    # Desenha o nó atual com uma quebra de linha para ficar mais limpo
    texto_no = atributo.replace(" (cm)", "")
    desenhar_no_c45(ax, texto_no, (x, y), pai_coord, rotulo_ramo, is_folha=False)
    
    # Bifurcação binária (Esquerda: <= limiar | Direita: > limiar)
    ramo_esq = arvore[atributo]['<=']
    ramo_dir = arvore[atributo]['>']
    
    # Ajustamos o dx para que a pirâmide não cruze os nós inferiores
    plotar_arvore_c45_recursiva(ramo_esq, ax, x - dx, y - dy, dx / 1.6, dy, 
                                pai_coord=(x, y), rotulo_ramo=f"<= {limiar:.2f}")
    
    plotar_arvore_c45_recursiva(ramo_dir, ax, x + dx, y - dy, dx / 1.6, dy, 
                                pai_coord=(x, y), rotulo_ramo=f"> {limiar:.2f}")

# =====================================================================
# 5. EXECUÇÃO
# =====================================================================
if __name__ == "__main__":
    alvo = 'Especie'
    atributos_disponiveis = iris.feature_names
    
    print("Treinando modelo C4.5 (from scratch)...")
    modelo_c45 = c45(df_treino, atributos_disponiveis, alvo, max_profundidade=4)
    
    acuracia = avaliar_modelo(modelo_c45, df_teste, alvo)
    print(f"Acurácia no Teste: {acuracia * 100:.2f}%\nGerando o gráfico...")
    
    # Configuração do Canvas do Matplotlib
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')
    
    # Ponto de partida: x=0.5 (meio da tela horizontal), y=1.0 (topo da tela)
    plotar_arvore_c45_recursiva(modelo_c45, ax, x=0.5, y=1.0, dx=0.25, dy=0.15)
    
    plt.title(f"Árvore de Decisão C4.5 - Dataset Iris\nAcurácia Teste: {acuracia*100:.2f}%", 
              fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.show()