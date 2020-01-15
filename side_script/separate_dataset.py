import pandas as pd

df = pd.read_csv("data/labels/dataset.csv")

df_copy = df.copy()
train_set = df_copy.sample(frac=0.85, random_state=0)
test_set = df_copy.drop(train_set.index)
train_set.to_csv(r'data/labels/train_dataset.csv', index=False)
test_set.to_csv(r'data/labels/test_dataset.csv', index=False)
