import random

import numpy as np

from network import FlexibleNetwork, sigmoid, sigmoid_prime, tanh, tanh_prime


def AND(x):
    return np.all(x)


def OR(x):
    return np.any(x)


def XOR(x):
    return np.sum(x) % 2


def generate_data(n, function):
    data = []
    for i in range(2 ** n):
        inputs = np.array([int(x) for x in format(i, f"0{n}b")]).reshape(n, 1)
        output = function(inputs)
        data.append((inputs, output))
    random.shuffle(data)
    split_index = int(0.8 * len(data))
    training_data = data[:split_index]
    test_data = data[split_index:]
    return training_data, test_data


if __name__ == '__main__':
    learning_rates = [0.1, 0.5, 1.0, 5.0]
    activations = [("Sigmoid", sigmoid, sigmoid_prime), ("Tanh", tanh, tanh_prime)]
    functions = [("AND", AND), ("OR", OR), ("XOR", XOR)]

    for function_name, function in functions:
        for use_bias in [True, False]:
            for n in [3]:
                for eta in learning_rates:
                    for activation_name, activation, activation_prime in activations:
                        print(
                            f"Função: {function_name} com {n} entradas, Taxa de Aprendizado: {eta}," +
                            f" Função de Ativação: {activation_name}, Uso de bias: {use_bias}")
                        training_data, test_data = generate_data(n, function)

                        # Usar a classe FlexibleNetwork e passar a função de ativação e sua derivada,
                        # e o parâmetro use_bias
                        net = FlexibleNetwork([n, n, 1], activation, activation_prime, use_bias)

                        net.SGD(training_data, 8, 10, eta, test_data=test_data)
                        print()
