import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

quantidade_individuos = 50
quantidade_maxima_geracoes = 100
valor_otimo = 10
numero_elites = 5
numero_execucoes = 100

def ler_pontos_do_arquivo(caminho_do_arquivo):
    dataframe = pd.read_csv(caminho_do_arquivo, header=None)
    return dataframe.values

def criar_populacao(tamanho, numero_de_pontos):
    return [np.random.permutation(numero_de_pontos) for _ in range(tamanho)]

def calcular_aptidao(cromossomo, pontos):
    soma_das_distancias = 0
    for indice in range(len(cromossomo)):
        ponto_atual = pontos[cromossomo[indice]]
        ponto_proximo = pontos[cromossomo[(indice + 1) % len(cromossomo)]]
        distancia = np.linalg.norm(ponto_atual - ponto_proximo)
        soma_das_distancias += distancia
    return soma_das_distancias

def selecao_por_torneio(populacao, aptidoes, tamanho_torneio=3):
    selecionados = []
    while len(selecionados) < len(populacao):
        competidores = np.random.choice(len(populacao), tamanho_torneio, replace=False)
        vencedor = competidores[np.argmin(aptidoes[competidores])]
        selecionados.append(populacao[vencedor])
    return selecionados

def crossover_de_dois_pontos(pai1, pai2):
    tamanho = len(pai1)
    filho1, filho2 = pai1.copy(), pai2.copy()
    ponto1, ponto2 = sorted(np.random.choice(range(1, tamanho-1), 2, replace=False))
    filho1[ponto1:ponto2], filho2[ponto1:ponto2] = pai2[ponto1:ponto2], pai1[ponto1:ponto2]
    return filho1, filho2

def mutacao_por_troca(cromossomo, taxa_de_mutacao=0.05):
    cromossomo = np.array(cromossomo)
    for indice in range(len(cromossomo)):
        if np.random.rand() < taxa_de_mutacao:
            j = np.random.randint(len(cromossomo))
            cromossomo[indice], cromossomo[j] = cromossomo[j], cromossomo[indice]
    return cromossomo.tolist()

def aplicar_elitismo(populacao, aptidoes, numero_de_elites):
    indices_das_elites = np.argsort(aptidoes)[:numero_de_elites]
    return [populacao[indice] for indice in indices_das_elites]

def plotar_tsp_3d(pontos, cromossomo):
    figura = plt.figure(figsize=(10, 8))
    ax = figura.add_subplot(111, projection='3d')
    coordenadas_x, coordenadas_y, coordenadas_z = pontos[:, 0], pontos[:, 1], pontos[:, 2]
    ax.scatter(coordenadas_x, coordenadas_y, coordenadas_z, color='blue', s=100, zorder=5)
    ax.scatter(coordenadas_x[0], coordenadas_y[0], coordenadas_z[0], color='red', s=200, zorder=5, label='Origem')
    for indice in range(len(cromossomo) - 1):
        posicao_inicial = pontos[cromossomo[indice]]
        posicao_final = pontos[cromossomo[indice + 1]]
        ax.plot([posicao_inicial[0], posicao_final[0]], [posicao_inicial[1], posicao_final[1]], [posicao_inicial[2], posicao_final[2]], 'k-', zorder=1)
    ax.plot([pontos[cromossomo[-1]][0], pontos[0][0]], [pontos[cromossomo[-1]][1], pontos[0][1]], [pontos[cromossomo[-1]][2], pontos[0][2]], 'k-', zorder=1)
    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.set_zlabel('Coordenada Z')
    ax.legend()
    ax.grid(True)
    plt.show()

def executar_algoritmo_genetico(pontos):
    populacao = criar_populacao(quantidade_individuos, len(pontos))
    todas_as_aptidoes = []
    melhor_solucao = np.inf
    melhor_cromossomo = None

    for geracao in range(quantidade_maxima_geracoes):
        aptidoes = np.array([calcular_aptidao(individuo, pontos) for individuo in populacao])
        todas_as_aptidoes.extend(aptidoes)
        if np.min(aptidoes) < melhor_solucao:
            melhor_solucao = np.min(aptidoes)
            melhor_cromossomo = populacao[np.argmin(aptidoes)]
            print(f"Geracao {geracao}: Melhor aptidao = {melhor_solucao}")

        if melhor_solucao <= valor_otimo:
            print("Condição de parada atingida.")
            break

        elites = aplicar_elitismo(populacao, aptidoes, numero_elites)
        selecionados = selecao_por_torneio(populacao, aptidoes)
        descendentes = [crossover_de_dois_pontos(selecionados[i], selecionados[(i + 1) % len(selecionados)]) for i in range(0, len(selecionados), 2)]
        populacao = [mutacao_por_troca(filho) for par in descendentes for filho in par]
        populacao.extend(elites)

    if melhor_cromossomo is not None:
        plotar_tsp_3d(pontos, melhor_cromossomo)
    else:
        print("Nenhuma solução válida encontrada.")

    menor_aptidao = np.min(todas_as_aptidoes)
    maior_aptidao = np.max(todas_as_aptidoes)
    media_aptidao = np.mean(todas_as_aptidoes)
    desvio_padrao_aptidao = np.std(todas_as_aptidoes)

    print("Menor Valor de Aptidão:", menor_aptidao)
    print("Maior Valor de Aptidão:", maior_aptidao)
    print("Média de Valor de Aptidão:", media_aptidao)
    print("Desvio-Padrão de Valor de Aptidão:", desvio_padrao_aptidao)

    return melhor_solucao

caminho_do_arquivo = 'C:\\Users\\av80076\\Documents\\IA\\CaixeiroSimples.csv'
pontos = ler_pontos_do_arquivo(caminho_do_arquivo)

melhor_aptidao = executar_algoritmo_genetico(pontos)
print("Melhor aptidão encontrada:", melhor_aptidao)
