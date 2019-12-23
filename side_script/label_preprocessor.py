import pandas as pd
import numpy as np

class LabelPreprocessor:
    def __init__(self, csv_file):
        self.labels = pd.read_csv(csv_file, index_col=0)
        self.labels = pd.DataFrame(self.labels)
        # print(self.labels)

    def extend_data(self):
        df = pd.DataFrame(columns=self.labels.columns)


        last_row = None
        for index, row in self.labels.iterrows():
            if last_row is not None and np.array_equal(np.array(last_row), np.array(row)):
                for i in range(int(last_row.name) + 1, int(row.name)):
                    last_row.name = i
                    df = df.append(last_row)
            df = df.append(row)
            last_row = row
            # row_next = self.labels.iloc[index+1]


        print(df, "#####")
        df.to_csv("path")
        return

    def save(self):

        return

if __name__ == "__main__":
    lp = LabelPreprocessor("data/labels/171214_1.csv")
    lp.extend_data()
