import pandas as pd, math
import matplotlib.pyplot as plt

dados = {
    'Historico':['Ruim', 'Desconhecido', 'Desconhecido', 'Bom', 'Bom', 'Bom', 'Ruim', 'Ruim', 'Bom', 'Desconhecido'],
    'Divida':   ['Alta', 'Alta', 'Baixa', 'Baixa', 'Baixa', 'Alta', 'Alta', 'Baixa', 'Baixa', 'Baixa'],
    'Garantia': ['Nenhuma', 'Nenhuma', 'Nenhuma', 'Nenhuma', 'Adequada', 'Adequada', 'Nenhuma', 'Adequada', 'Nenhuma', 'Adequada'],
    'Renda':    ['Baixa', 'Alta', 'Alta', 'Alta', 'Alta', 'Alta', 'Baixa', 'Baixa', 'Baixa', 'Alta'],
    'Risco':    ['Alto', 'Alto', 'Baixo', 'Baixo', 'Baixo', 'Baixo', 'Alto', 'Alto', 'Baixo', 'Baixo']
}
df = pd.DataFrame(dados)

def entropia(colunaAlvo):
    #entropia = -p1*log2(p1) - p2*log2(p2) - ... - pk*log2(pk)
    valorEnt = 0
    proporcao = colunaAlvo.value_counts(normalize=True)
    for p in proporcao:
        valorEnt += p * math.log2(p)
    return -valorEnt

def calculoGanhoInformacao(dados,atributo,alvo):
    #ganho de informação = entropia do dataset - entropia ponderada dos subconjuntos
    enttropiaOriginal = entropia(dados[alvo])
    valorAtributo = dados[atributo].value_counts(normalize=True)
    print(valorAtributo)
    entropiaPonderada = 0
    for valor, proporcao in valorAtributo.items():
        subconjunto = dados[dados[atributo] == valor][alvo]
        entropiaPonderada += proporcao * entropia(subconjunto)

    return enttropiaOriginal - entropiaPonderada

def id3(dados, atributos, alvo):
    if (len(dados[alvo].unique()) == 1):
        return dados[alvo].iloc[0]
    if (len(atributos) == 0):
        return dados[alvo].mode()[0]
    
    ganhos = {}
    for atributo in atributos:
        ganhos[atributo] = calculoGanhoInformacao(dados, atributo, alvo)

    print(ganhos)

    melhorAtributo = max(ganhos, key=ganhos.get)

    arvore = {melhorAtributo: {}}
    atributosRestantes = []
    for atributo in atributos:
        if atributo != melhorAtributo:
            atributosRestantes.append(atributo)

    for valor in dados[melhorAtributo].unique():
        subconjunto = dados[dados[melhorAtributo] == valor]
        arvore[melhorAtributo][valor] = id3(subconjunto, atributosRestantes, alvo)

    return arvore

def desenhar_no(ax, texto, centro, pai_coord, rotulo_ramo, is_folha=False):
    """Desenha um nó (caixa) e uma seta conectando-o ao pai."""
    # Estilos diferentes para nós de decisão e folhas
    if is_folha:
        estilo_caixa = dict(boxstyle="round,pad=0.4", fc="lightgreen", ec="black")
    else:
        estilo_caixa = dict(boxstyle="square,pad=0.4", fc="lightblue", ec="black")

    # Se tiver um pai, desenha a seta vindo dele
    if pai_coord is not None:
        ax.annotate(texto, xy=pai_coord, xytext=centro,
                    arrowprops=dict(arrowstyle="<-", color="black", lw=1.5),
                    bbox=estilo_caixa, ha='center', va='center', fontsize=10, weight='bold')
        
        # Adiciona o texto na linha (ex: 'Baixa', 'Alta')
        x_meio = (centro[0] + pai_coord[0]) / 2
        y_meio = (centro[1] + pai_coord[1]) / 2
        ax.text(x_meio, y_meio, rotulo_ramo, ha='center', va='center', 
                fontsize=9, color="darkred", 
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.9))
    else:
        # Nó Raiz (sem pai)
        ax.text(centro[0], centro[1], texto, ha='center', va='center', 
                fontsize=10, weight='bold', bbox=estilo_caixa)

def plotar_arvore_recursiva(arvore, ax, x, y, dx, dy, pai_coord=None, rotulo_ramo=""):
    """Caminha pela árvore calculando as coordenadas e desenhando."""
    # CASO BASE: Nó folha
    if not isinstance(arvore, dict):
        desenhar_no(ax, str(arvore), (x, y), pai_coord, rotulo_ramo, is_folha=True)
        return

    # Nó de Decisão (Atributo)
    atributo = list(arvore.keys())[0]
    ramos = arvore[atributo]
    
    # Desenha o nó atual
    desenhar_no(ax, atributo, (x, y), pai_coord, rotulo_ramo, is_folha=False)
    
    # Calcula a posição dos filhos
    num_ramos = len(ramos)
    # Ajusta o espaçamento horizontal (dx) baseado na quantidade de filhos para evitar sobreposição
    inicio_x = x - dx * (num_ramos - 1) / 2
    
    for i, (valor_ramo, sub_arvore) in enumerate(ramos.items()):
        filho_x = inicio_x + i * dx
        filho_y = y - dy
        # Reduz o dx para a próxima camada para caber na tela
        plotar_arvore_recursiva(sub_arvore, ax, filho_x, filho_y, dx / 1.5, dy, 
                                pai_coord=(x, y), rotulo_ramo=str(valor_ramo))


if __name__ == "__main__":
    atributos_disponiveis = ['Historico', 'Divida', 'Garantia', 'Renda']
    alvo = 'Risco'

    # Treinando o modelo
    modelo_arvore = id3(df, atributos_disponiveis, alvo)
    print(modelo_arvore)

    # Configurando o Canvas do Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off') # Remove os eixos (grid)
    
    # Renderiza a árvore (começando no topo, centro)
    plotar_arvore_recursiva(modelo_arvore, ax, x=0.5, y=1.0, dx=0.3, dy=0.2)
    
    plt.title("Árvore de Decisão ID3 (Risco de Crédito)", fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.show()