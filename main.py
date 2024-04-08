import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython.display import HTML
import io
import os
import base64
from gym.wrappers.monitoring.video_recorder import VideoRecorder

def criar_bins(num_bins_por_observacao):
    # Criação de bins para discretização do espaço de observação
    bins = np.array([
        np.linspace(-1.5, 1.5, num_bins_por_observacao),
        np.linspace(-1.5, 1.5, num_bins_por_observacao),
        np.linspace(-5.0, 5.0, num_bins_por_observacao),
        np.linspace(-5.0, 5.0, num_bins_por_observacao),
        np.linspace(-math.pi, math.pi, num_bins_por_observacao),
        np.linspace(-5.0, 5.0, num_bins_por_observacao),
        np.linspace(0.0, 1.0, num_bins_por_observacao),
        np.linspace(0.0, 1.0, num_bins_por_observacao)
    ])
    return bins 

def discretizar_observacao(observacoes, bins):
    # Discretização das observações usando os bins
    observacoes_discretizadas = [np.digitize(observacao, bins[i]) for i, observacao in enumerate(observacoes)]
    return tuple(observacoes_discretizadas)

def selecionar_acao_epsilon_guloso(epsilon, qTabela, estado_discreto):
    # Seleção de ação usando a estratégia epsilon-gulosa
    if np.random.random() > epsilon:
        acao = np.argmax(qTabela[estado_discreto])
    else:
        acao = np.random.randint(0, env.action_space.n)
    return acao

def calcularProximoQValor(valorQAntigo, recompensa, proximoValorQOtimo, ALPHA, GAMMA):
    # Cálculo do próximo valor Q para atualização
    return valorQAntigo + ALPHA * (recompensa + GAMMA * proximoValorQOtimo - valorQAntigo)

def reduzirEpsilon(epsilon, REDUCAO_EPSILON, EPSILON_MINIMO):
    # Redução do epsilon para diminuir a exploração ao longo do tempo
    return max(EPSILON_MINIMO, epsilon - REDUCAO_EPSILON)

def mostrar_video():
    # Exibição do vídeo gravado
    mp4list = list(filter(lambda x: x.endswith('.mp4'), os.listdir('.')))
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        return HTML(data='''<video alt="test" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))
    else:
        print("Nenhum vídeo encontrado.")

# Configuração inicial
env = gym.make("LunarLander-v2")
NUM_BINS = 30
BINS = criar_bins(NUM_BINS)
qShape = (NUM_BINS + 1,) * 8 + (env.action_space.n,)
qTabela = np.zeros(qShape)

# Parâmetros
EPOCAS = 10000
epsilon = 1.0
REDUCAO_EPSILON = 0.0001
EPSILON_MINIMO = 0.01
ALPHA = 0.001
GAMMA = 0.999
recompensas = []

for epoca in range(EPOCAS):
    estadoInicial = env.reset()
    estado_discreto = discretizar_observacao(estadoInicial, BINS)
    concluido = False
    recompensaTotal = 0.0

    while not concluido:
        acao = selecionar_acao_epsilon_guloso(epsilon, qTabela, estado_discreto)
        proximoEstado, recompensa, concluido, _ = env.step(acao)
        recompensaTotal += recompensa

        proximoEstadoDiscretizado = discretizar_observacao(proximoEstado, BINS)
        valorQAntigo = qTabela[estado_discreto + (acao,)]
        proximoValorQOtimo = np.max(qTabela[proximoEstadoDiscretizado])
        qTabela[estado_discreto + (acao,)] = calcularProximoQValor(valorQAntigo, recompensa, proximoValorQOtimo, ALPHA, GAMMA)

        estado_discreto = proximoEstadoDiscretizado

    recompensas.append(recompensaTotal)
    epsilon = reduzirEpsilon(epsilon, REDUCAO_EPSILON, EPSILON_MINIMO)
    if epoca % 100 == 0:
        print(f"Episódio: {epoca}, Recompensa Total: {recompensaTotal}, Epsilon: {epsilon}")

# Plotando gráfico de recompensa por episódio
plt.plot(recompensas)
plt.title('Recompensa X Episódio')
plt.xlabel('Episódio')
plt.ylabel('Recompensa Total')
plt.show()

# Demonstração do agente treinado
env_demo = gym.make("LunarLander-v2")
caminho_video = './video_demo.mp4'
gravador_video = VideoRecorder(env_demo, caminho_video, enabled=True)

estado = env_demo.reset()
concluido = False
while not concluido:
    estado_discreto = discretizar_observacao(estado, BINS)
    acao = np.argmax(qTabela[estado_discreto])  # Sempre escolhe a melhor ação
    estado, _, concluido, _ = env_demo.step(acao)
    gravador_video.capture_frame()
gravador_video.close()
env_demo.close()

mostrar_video()
