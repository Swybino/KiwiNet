import pandas as pd
import os


if __name__ == "__main__":
    data_folder = "data/raw_labels"
    df = pd.DataFrame(columns=["video", "frame", "name", "target"])
    max_frames = [9900, 8000, 10000, 12000]
    for file in os.listdir(data_folder):
        tmp_df = pd.DataFrame(columns=["video", "frame", "name", "target"])
        video = file[:-4]
        data = pd.read_csv(os.path.join(data_folder, file))
        if len(max_frames) > 0:
            data = data.loc[data["frame"] < max_frames.pop(0)]
        for name in data.columns[1:]:
            name_df = data[["frame", name]]
            name_df.columns= ["frame", "target"]
            name_df.insert(0, "video", video)
            name_df.insert(2, "name", name)

            df = df.append(name_df)

    df.to_csv("data/labels/labels.csv", index=False)
