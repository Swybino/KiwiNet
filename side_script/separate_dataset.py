import pandas as pd
import numpy.random as rdm

df = pd.read_csv("data/labels/labels.csv")

patch_size = 40
sep_factor = 0.8
df_copy = df.copy()

t1 = df_copy.sample(frac=(1-sep_factor)/(patch_size*5.2), random_state=0)
t2 = pd.DataFrame(columns=t1.columns)
for index, row in t1.iterrows():
    for k in range(0, patch_size):
        tmp_row = df.iloc[int(row.name) + k]
        rows_list = df.loc[(df['video'] == tmp_row["video"]) & (df['frame'] == tmp_row['frame'])]
        t2 = t2.append(rows_list)
t2.drop_duplicates()

train = df_copy.drop(t2.index)
print(len(t2), len(train), len(df))
train.to_csv(r'data/labels/train_labels_frame_patches.csv', index=False)
t2.to_csv(r'data/labels/test_labels_frame_patches.csv', index=False)
