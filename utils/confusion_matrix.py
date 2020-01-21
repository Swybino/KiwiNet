import numpy as np
import pandas as pd


class ConfusionMatrix:
    def __init__(self):
        self.names_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "z"]
        self.mat = np.zeros((len(self.names_list), len(self.names_list)))
        zeros = np.zeros((len(self.names_list), len(self.names_list)))
        self.matrix = pd.DataFrame(zeros, index=self.names_list, columns=self.names_list, dtype=float)

    def add_results(self, target, output):
        for idx, v in enumerate(target):
            self.matrix[output[idx]][v] += 1

    def normalize(self):
        for i in self.names_list:
            t = 0
            for j in self.names_list:
                t += self.matrix[j][i]

            for j in self.names_list:
                self.matrix[j][i] = round(self.matrix[j][i]/max(t,1), 3)

    def __str__(self):
        return str(self.matrix)


if __name__ == "__main__":
    mat = ConfusionMatrix()
    mat.add_results(["a","b","c","z"], ["a","z","e","z"])
    print(mat)
