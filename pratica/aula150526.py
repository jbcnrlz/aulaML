import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Especie'] = iris.target

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

def entropia(colunaAlvo):
    proportions = colunaAlvo.value_counts(normalize=True)
    ent = []
    for p in proportions:
        ent.append(p * math.log2(p))
    return -sum(ent)

def encontrarLimiar(dados, atributo, alvo):
    valoresUnicos = np.sort(dados[atributo].unique())
    melhorGanho = -1
    melhorLimiar= None
    entropiaBase = entropia(dados[alvo])

    for i in range(len(valoresUnicos) - 1):
        limiar = (valoresUnicos[i] + valoresUnicos[i + 1]) / 2.0

        subsetEsquerda = dados[dados[atributo] <= limiar][alvo]
        subsetDireita = dados[dados[atributo] > limiar][alvo]

        pesoEsquerda = len(subsetEsquerda) / len(dados)
        pesoDireita = len(subsetDireita) / len(dados)
        entropiaPonderada = (pesoEsquerda * entropia(subsetEsquerda)) + (pesoDireita * entropia(subsetDireita))
        ganho = entropiaBase - entropiaPonderada

        if ganho > melhorGanho:
            melhorGanho = ganho
            melhorLimiar = limiar

    return melhorLimiar, melhorGanho


def c45(dados, atributos, alvo, profundidadeAtual=0, profundidadeMaxima=4):

    if len(dados[alvo].unique()) == 1:
        return dados[alvo].iloc[0]
    
    if profundidadeAtual >= profundidadeMaxima or len(atributos) == 0:
        return dados[alvo].mode()[0]
    
    melhorGanhoGlobal = -1
    melhorAtributoGlobal = None
    melhorLimiarGlobal = None

    for a in atributos:
        limiar, ganho = encontrarLimiar(dados, a, alvo)
        if ganho > melhorGanhoGlobal:
            melhorGanhoGlobal = ganho
            melhorAtributoGlobal = a
            melhorLimiarGlobal = limiar

    arvore = {melhorAtributoGlobal: { "limiar": melhorLimiarGlobal, '<=' :  {}, '>' : {}}}

    dadosEsquerda = dados[ dados[melhorAtributoGlobal] <= melhorLimiarGlobal]
    dadosDireita = dados[ dados[melhorAtributoGlobal] > melhorLimiarGlobal]

    novosAtributos = []
    for a in atributos:
        if a != melhorAtributoGlobal:
            novosAtributos.append(a)

    arvore[melhorAtributoGlobal]['<='] = c45(dadosEsquerda, novosAtributos, alvo, profundidadeAtual + 1, profundidadeMaxima)
    arvore[melhorAtributoGlobal]['>'] = c45(dadosDireita, novosAtributos, alvo, profundidadeAtual + 1, profundidadeMaxima)

    return arvore

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

if __name__ == "__main__":
    alvo = 'Especie'
    atributos = iris.feature_names
    arvore_c45 = c45(df_train, atributos, alvo)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')

    plotar_arvore_c45_recursiva(arvore_c45, ax, x=0.5, y=1.0, dx=0.25, dy=0.15)
    plt.show()