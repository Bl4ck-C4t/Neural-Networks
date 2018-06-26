import random


class Datasets:
    class Sums:
        def get(lenght):
            inputs = [[random.randint(-9, 9) for _ in range(3)] for _ in range(lenght)]
            outs = [sum(x) for x in inputs]
            return inputs, outs

        def perform(inputs):
            return sum(inputs)
