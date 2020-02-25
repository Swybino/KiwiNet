import os

import pandas as pd
from utils.confusion_matrix import ConfusionMatrix
import utils.utils as utils
from utils.video import Video
from utils.viewer import Viewer
import config
import argparse
import numpy as np


def display_file_data(video, frame, df=None):
    file = "%s_%s.json" % (video, frame)
    if args.inputs is not None:
        input_dir = args.inputs
    else:
        input_dir = config.inputs_dir

    data = utils.read_input(os.path.join(input_dir, file))
    # print(os.path.join(config.video_root, "%s.MP4" % video))
    print(frame)
    img = Video(os.path.join(config.video_root, "%s.MP4" % video))[int(frame)]
    viewer = Viewer(img)
    viewer.plt_frame_idx(frame)

    if args.anonymize:
        for name, item in data.items():
            bbox = item[config.BBOX_KEY]
            bbox = list(utils.get_roi(img, bbox, scale=1.3))
            bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            viewer.blur(bbox)

    if args.bbox:
        for name, item in data.items():
            bbox = item[config.BBOX_KEY]
            viewer.plt_bbox(bbox, name, thickness=2)

    if args.landmarks:
        for name, item in data.items():
            landmarks = item[config.LANDMARKS_KEY]
            viewer.plt_landmarks(landmarks)

    if args.pose:
        for name, item in data.items():
            bbox = item[config.BBOX_KEY]
            pose = item[config.POSE_KEY]
            viewer.plt_axis(pose[0], pose[1], pose[2], bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2,
                            conf=int(item[config.CONFIDENCE_KEY]))

    if df is not None:
        for name, item in data.items():
            bbox = item[config.BBOX_KEY]
            target = df.loc[(df['video'] == video) & (df['frame'] == frame) & (df['name'] == name)]
            if len(target) > 0:
                target = target['target'].values[0]
                if target == "z":
                    bbox_target = bbox
                else:
                    bbox_target = data[target][config.BBOX_KEY]
                viewer.plt_results([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2],
                                   [bbox_target[0] + bbox_target[2] / 2, bbox_target[1] + bbox_target[3] / 2],
                                   color=config.kids_color[name])

    if args.show:
        viewer.show()
    if args.out is not None:
        viewer.save_img(os.path.join(args.out, "%s_%s.jpg" % (video, frame)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kiwi Training')
    parser.add_argument('-b', '--bbox', action='store_true', help='')
    parser.add_argument('-l', '--landmarks', action='store_true', help='')
    parser.add_argument('-p', '--pose', action='store_true', help='')
    parser.add_argument('-a', '--anonymize', action='store_true', help='')
    parser.add_argument('-s', '--show', action='store_true', help='')
    parser.add_argument('-o', '--out', type=str)
    parser.add_argument('-v', '--videos', nargs='+', type=str, help='')
    parser.add_argument('-f', '--frames', nargs='+', type=int, help='')
    parser.add_argument('-r', '--frame_range', nargs='+', type=int, help='')
    parser.add_argument('-d', '--data', type=str, help='')
    parser.add_argument('-i', '--inputs', type=str, help='')
    parser.add_argument('--labels', type=str, help='')
    args = parser.parse_args()

    if args.data is not None and args.labels is not None:
        df = pd.read_csv(args.data)
        df2 = pd.DataFrame(columns=df.columns)
        labels = pd.read_csv(args.labels)
        for index, row in labels.iterrows():
            df2 = df2.append(df.loc[(df["video"] == row["video"]) & (df["frame"] == row["frame"]) & (df["name"] == row["name"])])

        prediction_list, labels_list = np.array(df2["target"]), np.array(labels["target"])
        cm = ConfusionMatrix()
        cm.add_results(labels_list, prediction_list)
        cm.normalize()
        accuracy = (prediction_list == labels_list).sum() / prediction_list.shape[0]
        print(accuracy, cm, sep="\n")

    if args.show or args.out is not None:
        if args.videos is not None and (args.frames is not None or args.frame_range is not None):
            if args.frames is not None:
                frames_list = args.frames
            elif args.frame_range is not None:
                frames_list = range(args.frame_range[0], args.frame_range[1])
            else:
                frames_list = []
            if args.data is not None:
                df = pd.read_csv(args.data)
            else:
                df = None
            for v in args.videos:
                for f in frames_list:
                    display_file_data(v, f, df)

        elif args.data is not None:
            df = pd.read_csv(args.data)
            for v in df.video.unique():
                print("#######", v)
                for f in df.loc[(df["video"] == v)].frame.unique():
                    display_file_data(v, f, df)
        else:
            for file in os.listdir(config.inputs_dir):
                tmp = file[:-5].split("_")
                f = tmp[-1]
                v = "_".join(tmp[:-1])
                display_file_data(v, f)
