import pandas as pd
import os


def convert_raw_labels(src_folder, dst_file):
    df = pd.DataFrame(columns=["video", "frame", "name", "target"])
    max_frames = [9900, 8000, 10000, 12000]
    max_frames = []
    for file in os.listdir(src_folder):
        tmp_df = pd.DataFrame(columns=["video", "frame", "name", "target"])
        video = file[:-4]
        data = pd.read_csv(os.path.join(src_folder, file))
        if len(max_frames) > 0:
            data = data.loc[data["frame"] < max_frames.pop(0)]
        for name in data.columns[1:]:
            name_df = data[["frame", name]]
            name_df.columns = ["frame", "target"]
            name_df.insert(0, "video", video)
            name_df.insert(2, "name", name)

            df = df.append(name_df)
    df.to_csv(dst_file, index=False)


if __name__ == "__main__":
    # convert_raw_labels("data/raw_labels", "data/labels/labels_originals.csv")
    convert_raw_labels("data/171220_1_2/labels", "data/171220_1_2/labels/171220_1_2_labels.csv")
