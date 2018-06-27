import random


class Datasets:
    class Sums:
        def perform(inputs):
            return sum(inputs)

        def get(lenght):
            inputs = [[random.randint(-9, 9) for _ in range(3)] for _ in range(lenght)]
            outs = [Datasets.Sums.perform(x) for x in inputs]
            return inputs, outs

