import pandas as pd


def separate_dataset(src_file, train_file, test_file, *, sep_factor=0.8, patch_size=1):
    df = pd.read_csv(src_file)
    df_copy = df.copy()
    if patch_size <= 1:
        t2 = df_copy.sample(frac=(1 - sep_factor), random_state=0)
    else:
        t1 = df_copy.sample(frac=(1 - sep_factor) / (patch_size * 5), random_state=0)
        t2 = pd.DataFrame(columns=t1.columns)
        for index, row in t1.iterrows():
            for k in range(0, patch_size):
                tmp_row = df.iloc[int(row.name) + k]
                rows_list = df.loc[(df['video'] == tmp_row["video"]) & (df['frame'] == tmp_row['frame'])]
                t2 = t2.append(rows_list)
        t2.drop_duplicates()

    train = df_copy.drop(t2.index)
    print(len(t2), len(train), len(df))
    train.to_csv(train_file, index=False)
    t2.to_csv(test_file, index=False)


if __name__ == '__main__':
    separate_dataset("data/labels/labels.csv", "data/labels/train_labels_frame_patches100.csv",
                     "data/labels/test_labels_frame_patches100.csv", patch_size=100)
