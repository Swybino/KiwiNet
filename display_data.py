import os
from utils.video import Video
import utils.utils as utils
from utils.viewer import Viewer
import config
import argparse
import numpy as np


def display_file_data(video, frame):
    file = "%s_%s.json" % (video, frame)
    data = utils.read_input(os.path.join(config.inputs_dir, file))
    print(os.path.join(config.video_root, "%s.MP4" % video))
    img = Video(os.path.join(config.video_root, "%s.MP4" % video))[int(frame)]
    img_size = img.shape[0]
    viewer = Viewer(img)

    for name, item in data.items():
        if args.anonymize:
            bbox = np.array(item[config.BBOX_KEY]) * img_size / 640
            bbox = bbox.astype(int)
            viewer.blur(bbox)

        if args.bbox:
            bbox = np.array(item[config.BBOX_KEY]) * img_size / 640
            bbox = bbox.astype(int)
            viewer.plt_bbox(bbox, name)

        if args.landmarks:
            landmarks = np.array(item[config.LANDMARKS_KEY]) * img_size / 640
            viewer.plt_landmarks(landmarks)

        if args.pose:
            bbox = np.array(item[config.BBOX_KEY]) * img_size / 640
            pose = item[config.POSE_KEY]
            viewer.plt_axis(pose[0], pose[1], pose[2], bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
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
    parser.add_argument('-f', '--frames', nargs='+', type=str, help='')
    args = parser.parse_args()

    if args.videos is not None and args.frames is not None:
        for v in args.videos:
            for f in args.frames:
                display_file_data(v, f)
    else:
        for file in os.listdir(config.inputs_dir):
            tmp = file[:-5].split("_")
            f = tmp[-1]
            v = "_".join(tmp)
            display_file_data(v, f)
