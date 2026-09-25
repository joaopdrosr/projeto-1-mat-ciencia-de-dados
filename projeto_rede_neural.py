from sklearn.decomposition import PCA
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
from keras.optimizers.schedules import InverseTimeDecay
from keras.utils import set_random_seed


NUM_NEURONIOS = 4
EPOCAS = 2000
TAXA_INICIAL = 10.0
DECAIMENTO = 0.001

np.random.seed(42)
set_random_seed(42)
dados = np.loadtxt("dados_ionosphere/ionosphere.data", delimiter=",", dtype=str)
X = PCA(n_components=2).fit_transform(dados[:, :-1].astype(float))
Y = (dados[:, -1] == "g").astype(int)
X = (X - X.mean(axis=0)) / X.std(axis=0)
NUM_AMOSTRAS = len(X)

cores = ["blue" if classe == 0 else "red" for classe in Y]
plt.scatter(X[:, 0], X[:, 1], c=cores)
plt.xlabel("Primeira componente principal (z-score)")
plt.ylabel("Segunda componente principal (z-score)")
plt.title("Base Ionosphere")
plt.savefig("figuras/ionosphere.svg")
plt.close()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def run_neural_net(x, w0, b0, w1, b1, w2, b2):
    y0 = sigmoid(w0 @ x + b0)
    y1 = sigmoid(w1 @ y0 + b1)
    y2 = sigmoid(w2 @ y1 + b2)
    return 1 if y2[0] > 0.5 else 0


def neural_net(x, d, w0, b0, w1, b1, w2, b2):
    # forward
    y0 = sigmoid(w0 @ x + b0)
    y1 = sigmoid(w1 @ y0 + b1)
    y2 = sigmoid(w2 @ y1 + b2)
    erro = y2[0] - d
    loss = erro**2

    # backward
    grad_v2 = erro * y2[0] * (1 - y2[0])
    grad_w2 = grad_v2 * y1
    grad_b2 = np.array([grad_v2])

    grad_v1 = grad_v2 * w2 * y1 * (1 - y1)
    grad_w1 = np.outer(grad_v1, y0)
    grad_b1 = grad_v1

    grad_v0 = (w1.T @ grad_v1) * y0 * (1 - y0)
    grad_w0 = np.outer(grad_v0, x)
    grad_b0 = grad_v0

    return grad_w0, grad_b0, grad_w1, grad_b1, grad_w2, grad_b2, loss


def calcular_acuracia(w0, b0, w1, b1, w2, b2):
    acertos = 0
    for i in range(NUM_AMOSTRAS):
        acertos += run_neural_net(X[i], w0, b0, w1, b1, w2, b2) == Y[i]
    return acertos / NUM_AMOSTRAS


def main():
    # inicialização aleatória
    w0 = np.random.rand(NUM_NEURONIOS, 2)
    w1 = np.random.rand(NUM_NEURONIOS, NUM_NEURONIOS)
    w2 = np.random.rand(NUM_NEURONIOS)
    b0 = np.random.rand(NUM_NEURONIOS)
    b1 = np.random.rand(NUM_NEURONIOS)
    b2 = np.random.rand(1)
    parametros_iniciais = (
        w0.copy(), b0.copy(), w1.copy(), b1.copy(), w2.copy(), b2.copy()
    )

    # taxa de aprendizado
    taxa = TAXA_INICIAL

    print(
        f"Acurácia antes do treinamento: "
        f"{calcular_acuracia(w0, b0, w1, b1, w2, b2):.2%}"
    )

    # gradiente descendente
    historico = []
    for epoca in range(EPOCAS):
        taxa = TAXA_INICIAL / (1 + DECAIMENTO * epoca)
        loss = 0.0
        grad_w0 = np.zeros(w0.shape)
        grad_w1 = np.zeros(w1.shape)
        grad_w2 = np.zeros(w2.shape)
        grad_b0 = np.zeros(b0.shape)
        grad_b1 = np.zeros(b1.shape)
        grad_b2 = np.zeros(b2.shape)

        for x, d in zip(X, Y):
            g_w0, g_b0, g_w1, g_b1, g_w2, g_b2, erro = neural_net(
                x, d, w0, b0, w1, b1, w2, b2
            )
            grad_w0 += g_w0
            grad_w1 += g_w1
            grad_w2 += g_w2
            grad_b0 += g_b0
            grad_b1 += g_b1
            grad_b2 += g_b2
            loss += erro

        escala_mse = 2 / NUM_AMOSTRAS
        w0 -= taxa * escala_mse * grad_w0
        w1 -= taxa * escala_mse * grad_w1
        w2 -= taxa * escala_mse * grad_w2
        b0 -= taxa * escala_mse * grad_b0
        b1 -= taxa * escala_mse * grad_b1
        b2 -= taxa * escala_mse * grad_b2
        historico.append(loss / NUM_AMOSTRAS)

        if epoca % 100 == 0:
            print(f"Época {epoca:5d} | perda: {loss / NUM_AMOSTRAS:.6f}")

    # acurácia após o treinamento
    acuracia_manual = calcular_acuracia(w0, b0, w1, b1, w2, b2)
    print(f"Acurácia após o treinamento: {acuracia_manual:.2%}")

    model = Sequential()
    model.add(Input(shape=(2,), dtype="float64"))
    model.add(Dense(NUM_NEURONIOS, activation="sigmoid", dtype="float64"))
    model.add(Dense(NUM_NEURONIOS, activation="sigmoid", dtype="float64"))
    model.add(Dense(1, activation="sigmoid", dtype="float64"))

    w0_inicial, b0_inicial, w1_inicial, b1_inicial, w2_inicial, b2_inicial = (
        parametros_iniciais
    )
    model.set_weights([
        w0_inicial.T,
        b0_inicial,
        w1_inicial.T,
        b1_inicial,
        w2_inicial.reshape(-1, 1),
        b2_inicial,
    ])

    taxa_keras = InverseTimeDecay(
        TAXA_INICIAL, decay_steps=1, decay_rate=DECAIMENTO
    )
    opt = SGD(learning_rate=taxa_keras)
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])
    model.fit(
        X,
        Y,
        epochs=EPOCAS,
        verbose=False,
        batch_size=NUM_AMOSTRAS,
        shuffle=False,
    )

    _, acuracia = model.evaluate(X, Y, verbose=False)
    print(f"Acurácia com Keras: {acuracia:.2%}")

    np.savez(
        "resultados_rede.npz",
        X=X,
        Y=Y,
        w0=w0,
        b0=b0,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        historico=np.array(historico),
        acuracia_manual=acuracia_manual,
        acuracia_keras=acuracia,
    )


if __name__ == "__main__":
    main()
