import math
import random
import datetime
import pickle
from datasets import Datasets
from numpy import random as rm


# TODO by printing value by value from both analyzers and comparing
# TODO more OOP
# TODO continue from layer1_error
# TODO save() and load()

class Neuron:
    def __init__(self, inputs):
        self.weights = [2 * random.uniform(-1, 1) for _ in range(inputs)]  # '_' - doesn't save the iteration

    def __getitem__(self, item):
        return self.weights[item]

    def __setitem__(self, key, value):
        self.weights[key] = value

    def __len__(self):
        return len(self.weights)

    def __iter__(self):
        return iter(self.weights)

    def __repr__(self):
        return str([round(weight, 4) for weight in self.weights])


class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron, name):
        self.neurons = [Neuron(number_of_inputs_per_neuron) for _ in range(number_of_neurons)]
        self.name = name

    def __getitem__(self, item):
        return self.neurons[item]

    def __setitem__(self, key, value):
        self.neurons[key] = value

    def __len__(self):
        return len(self.neurons)

    def __iter__(self):
        return iter(self.neurons)

    def __str__(self):
        p1 = "    {} ({} neurons, each with {} inputs): \n".format(self.name, len(self), len(self[0]))
        return p1 + str(self.neurons)


class NeuralNetwork():
    def __init__(self, layer1, layer2, name="Default"):
        self.layer1 = layer1
        self.layer2 = layer2
        self.scale_n = 1
        self.name = name

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def sigmoid_der(self, x):
        return x * (1 - x)

    def sum_weights(self, inputs, weights):
        return sum(val * weight for val, weight in zip(inputs, weights))

    def save(self, filename="network.d"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
            print("Saved.")

    def load(filename="network.d"):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def adjust(self, neurons, inputs, deltas):
        for i, neuron in enumerate(neurons):
            for j, weight in enumerate(neuron):
                adjustment = inputs[j] * deltas[i]
                neurons[i][j] += adjustment

    def scale(self, ls, down=True):
        scale = self.scale_n
        if down:
            return [x / scale for x in ls]
        else:
            return [x * scale for x in ls]

    def get_scale(self, inputs, outputs):
        biggest = outputs[0]
        for x in inputs:
            biggest = sorted(x)[-1] if biggest < sorted(x)[-1] else biggest
        biggest = sorted(outputs)[-1] if biggest < sorted(outputs)[-1] else biggest
        return 10**len(str(biggest))

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations, fl='network.d'):
        self.scale_n = self.get_scale(training_set_inputs, training_set_outputs)
        self.start_time = datetime.datetime.now()
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            if iteration % 5000 == 0:
                print("Currently at iteration " + str(iteration))
                delta_time = datetime.datetime.now() - self.start_time
                print("Time elapsed: " + str(delta_time))
                if iteration == 5000:
                    print("Approximate training time:", delta_time*(number_of_training_iterations/5000))

            if iteration % 50000 == 0 and iteration > 0:
                self.save(fl)

            for s_input, s_output in zip(training_set_inputs, training_set_outputs):
                s_input = self.scale(s_input)
                s_output = s_output / self.scale_n
                # exit()
                output, neuron_outputs = self.think(s_input)
                # Calculate the error for layer 2 (The difference between the desired output
                # and the predicted output).
                output_error = s_output - output
                output_delta = output_error * self.sigmoid_der(output)

                hidden_errors = [output_delta * weight for weight in self.layer2[0]]
                hidden_deltas = [self.sigmoid_der(x) * hidden_errors[i] for i, x in enumerate(neuron_outputs)]

                self.adjust(self.layer1, s_input, hidden_deltas)
                self.adjust(self.layer2, neuron_outputs, [output_delta])
        end_time = datetime.datetime.now() - self.start_time
        print("Done {} iterations on {} datasets for ".format(iteration + 1, len(training_set_inputs)) + str(end_time))
        self.save(fl)

    # The neural network thinks.
    def think(self, inputs):
        neuron_outputs = []
        for i, weights in enumerate(self.layer1):
            neuron_outputs.append(self.sigmoid(self.sum_weights(inputs, weights)))
        output = self.sigmoid(self.sum_weights(neuron_outputs, self.layer2[0]))
        return output, neuron_outputs

    def predict(self, inputs):
        res, _ = self.think(self.scale(inputs))
        return res * self.scale_n

    # The neural network prints its weights
    def print_layers(self):
        print(self.layer1)
        print(self.layer2)



