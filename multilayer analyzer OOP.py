import math
import random
import datetime
import pickle
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
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2
        self.scale_n = 1

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
        return sum(val * weights[i] for i, val in enumerate(inputs))

    def save(self, filename="network.d"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
            print("Saved.")

    def load(filename="network.d"):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def rotation_sum(self, inputs, weights):
        res = []
        for inp in inputs:
            tmp = []
            for x in weights:
                tmp.append(inp * x)
            res.append(tmp)
        return res

    def adjust(self, neurons, inputs, deltas):
        for i, neuron in enumerate(neurons):
            for j, weight in enumerate(neuron):
                layer1_adjustment = inputs[j] * deltas[i]
                neurons[i][j] += layer1_adjustment

    def scale(self, ls, down=True):
        biggest = int(sorted(ls)[-1])
        scale = 10**(len(str(biggest))) if biggest > 1 else 1
        self.scale_n = scale if self.scale_n < scale else self.scale_n
        if down:
            return [x/scale for x in ls]
        else:
            return [x*scale for x in ls]

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        self.start_time = datetime.datetime.now()
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            if iteration % 5000 == 0:
                print("Currently at iteration " + str(iteration))
                delta_time = datetime.datetime.now() - self.start_time
                print("Time elapsed: " + str(delta_time))
            for i in range(len(training_set_inputs)):
                s_input = self.scale(training_set_inputs[i])
                # exit()
                output, neuron_outputs = self.think(s_input)
                # Calculate the error for layer 2 (The difference between the desired output
                # and the predicted output).
                output_error = training_set_outputs[i] - output
                output_delta = output_error * self.sigmoid_der(output)

                hidden_error = [output_delta * weight for weight in self.layer2[0]]
                hidden_delta = [self.sigmoid_der(x) * hidden_error[i] for i, x in enumerate(neuron_outputs)]

                self.adjust(self.layer1, s_input, hidden_delta)
                self.adjust(self.layer2, neuron_outputs, [output_delta])
        end_time = datetime.datetime.now() - self.start_time
        print("Done {} iterations on {} datasets for ".format(iteration + 1, len(training_set_inputs)) + str(end_time))
        self.save()

    # The neural network thinks.
    def think(self, inputs):
        neuron_outputs = []
        for i, weights in enumerate(self.layer1):
            neuron_outputs.append(self.sigmoid(self.sum_weights(inputs, weights)))
        output = self.sigmoid(self.sum_weights(neuron_outputs, self.layer2[0]))
        return output, neuron_outputs

    def predict(self, inputs):
        res = self.think(inputs)
        return res*self.scale_n


    # The neural network prints its weights
    def print_layers(self):
        print(self.layer1)
        print(self.layer2)


if __name__ == "__main__":
    # Seed the random number generator
    # Create layer 1 (4 neurons, each with 3 inputs)
    #layer1 = NeuronLayer(4, 3, "Layer 1")

    # Create layer 2 (a single neuron with 4 inputs)
    #layer2 = NeuronLayer(1, 4, "Layer 2")

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork.load()

    # print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_layers()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]]
    training_set_outputs = [0, 1, 1, 1, 1, 0, 0]

    # training_set_inputs = numpy.array(training_set_inputs)

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 80000)

    # print("Stage 2) New synaptic weights after training: ")
    neural_network.print_layers()

    # Test the neural network with a new situation.
    # print("Stage 3) Considering a new situation [1, 1, 0] -> ?: ")
    result, hidden_state = neural_network.think([1, 1, 0])
    print(result)

    print("Accurancy: " + str((abs(0 - result) / result)*100) + "%")
