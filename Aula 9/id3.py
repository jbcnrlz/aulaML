import pandas as pd
import math
import matplotlib.pyplot as plt

# =====================================================================
# 1. CONJUNTO DE DADOS (Dataset de Risco de Crédito)
# =====================================================================
dados = {
    'Historico': ['Ruim', 'Desconhecido', 'Desconhecido', 'Bom', 'Bom', 'Bom', 'Ruim', 'Ruim', 'Bom', 'Desconhecido'],
    'Divida': ['Alta', 'Alta', 'Baixa', 'Baixa', 'Baixa', 'Alta', 'Alta', 'Baixa', 'Baixa', 'Baixa'],
    'Garantia': ['Nenhuma', 'Nenhuma', 'Nenhuma', 'Nenhuma', 'Adequada', 'Adequada', 'Nenhuma', 'Adequada', 'Nenhuma', 'Adequada'],
    'Renda': ['Baixa', 'Alta', 'Alta', 'Alta', 'Alta', 'Alta', 'Baixa', 'Baixa', 'Baixa', 'Alta'],
    'Risco': ['Alto', 'Alto', 'Baixo', 'Baixo', 'Baixo', 'Baixo', 'Alto', 'Alto', 'Baixo', 'Baixo']
}
df = pd.DataFrame(dados)

# =====================================================================
# 2. FUNÇÕES MATEMÁTICAS 
# =====================================================================
def calcular_entropia(coluna_alvo):
    proporcoes = coluna_alvo.value_counts(normalize=True)
    return -sum(p * math.log2(p) for p in proporcoes)

def calcular_ganho_informacao(df, atributo, alvo):
    entropia_original = calcular_entropia(df[alvo])
    valores_atributo = df[atributo].value_counts(normalize=True)
    entropia_ponderada = 0
    for valor, fracao in valores_atributo.items():
        subconjunto = df[df[atributo] == valor][alvo]
        entropia_ponderada += fracao * calcular_entropia(subconjunto)
    return entropia_original - entropia_ponderada

# =====================================================================
# 3. ALGORITMO ID3
# =====================================================================
def id3(df, atributos, alvo):
    if len(df[alvo].unique()) == 1:
        return df[alvo].iloc[0] 
    if len(atributos) == 0:
        return df[alvo].mode()[0]
    
    ganhos = {atr: calcular_ganho_informacao(df, atr, alvo) for atr in atributos}
    melhor_atributo = max(ganhos, key=ganhos.get)
    
    arvore = {melhor_atributo: {}}
    atributos_restantes = [atr for atr in atributos if atr != melhor_atributo]
    
    for valor in df[melhor_atributo].unique():
        sub_df = df[df[melhor_atributo] == valor]
        if sub_df.empty:
            arvore[melhor_atributo][valor] = df[alvo].mode()[0]
        else:
            arvore[melhor_atributo][valor] = id3(sub_df, atributos_restantes, alvo)
            
    return arvore

# =====================================================================
# 4. VISUALIZAÇÃO COM MATPLOTLIB
# =====================================================================
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

# =====================================================================
# 5. EXECUÇÃO
# =====================================================================
if __name__ == "__main__":
    atributos_disponiveis = ['Historico', 'Divida', 'Garantia', 'Renda']
    alvo = 'Risco'

    # Treinando o modelo
    modelo_arvore = id3(df, atributos_disponiveis, alvo)
    
    # Configurando o Canvas do Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off') # Remove os eixos (grid)
    
    # Renderiza a árvore (começando no topo, centro)
    plotar_arvore_recursiva(modelo_arvore, ax, x=0.5, y=1.0, dx=0.3, dy=0.2)
    
    plt.title("Árvore de Decisão ID3 (Risco de Crédito)", fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.show()