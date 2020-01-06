import pandas as pd
import numpy as np


class LabelPreprocessor:
    def __init__(self, csv_file):
        self.labels = pd.read_csv(csv_file)
        self.labels = pd.DataFrame(self.labels)
        # print(self.labels)

    def extend_data(self):
        df = pd.DataFrame(columns=self.labels.columns)
        last_row = None
        for index, row in self.labels.iterrows():
            if last_row is not None and np.array_equal(np.array(last_row)[1:], np.array(row)[1:]):
                for i in range(int(last_row["frame"]) + 1, int(row["frame"])):
                    last_row["frame"] = i
                    df = df.append(last_row)
            df = df.append(row)
            last_row = row
            # row_next = self.labels.iloc[index+1]

        df.to_csv("data/labels/171214_2.csv", index=False)
        return


if __name__ == "__main__":
    lp = LabelPreprocessor("data/labels/171214_2_original.csv")
    lp.extend_data()
