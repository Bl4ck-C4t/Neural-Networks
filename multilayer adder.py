from neural import *
from datasets import Datasets

neural_network = NeuralNetwork(NeuronLayer(4, 3, "Hidden layer"), NeuronLayer(1, 4, "Output layer"))

training_inputs, training_outputs = Datasets.Sums.get(15)
neural_network.train(training_inputs, training_outputs, 100000, "sums.d")

en = input("Enter 3 inputs: ").split(" ")
en = [int(x) for x in en]
result = neural_network.predict(en)
exp = Datasets.Sums.perform(en)
accurancy = 100-(abs(exp - result) / exp) * 100
print("Prediction:", round(result, 2))
print("Accurancy: " + str(round(accurancy, 3)) + "%")
if accurancy < 96:
    print("Accurancy low. Adjusting...")
    neural_network.train([en], [exp], 15000, "sums.d")
