import pandas as pd
import utils.utils as utils
import os
import config


def separate_dataset(src_file, dst_file1, dst_file2):
    df = pd.read_csv(src_file)

    t1 = pd.DataFrame(columns=df.columns)
    t2 = pd.DataFrame(columns=df.columns)

    for index, row in df.iterrows():
        file = "%s_%s.json" % (row['video'], row['frame'])

        data = utils.read_input(os.path.join(config.inputs_dir, file))

        if int(data[row["name"]][config.CONFIDENCE_KEY]) == 1:
            t1 = t1.append(row)
        else:
            t2 = t2.append(row)

    print(len(t1), len(t2))
    t1.to_csv(dst_file1, index=False)
    t2.to_csv(dst_file2, index=False)


if __name__ == '__main__':
    separate_dataset("data/labels/train_labels_frame_patches100.csv",
                     "data/labels/train_labels_frame_patches100_good.csv",
                     "data/labels/train_labels_frame_patches100_bad.csv")
