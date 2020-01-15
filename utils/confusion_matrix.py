import numpy as np


class ConfusionMatrix:
    def __init__(self):
        self.names_list = ["1", "2", "3", "4", "5", "6", "z"]
        self.mat = np.zeros((len(self.names_list), len(self.names_list)))

    def add_multi_results(self, target, output):
        for idx in range(target.size(0)):
            self.add_result(target[idx], output[idx])

    def add_result(self, target, output):
        for idx, name in enumerate(self.names_list):
            print(self.mat)
            print(target,output)
            self.mat[self.names_list.index(target[name]), self.names_list.index(output[name])] += 1

    def normalize(self):
        self.mat = self.mat / self.mat.sum(1)[:, np.newaxis]

    def __str__(self):
        print(self.mat)
